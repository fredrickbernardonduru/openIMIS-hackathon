"""
feature_engineering.py
Track 3 — AI-Powered Claims Anomaly Detector — Phase 2

Takes the cleaned, anonymised dataset produced by insurance.ipynb
(anonymised_claims_data_cleaned.csv) and engineers the numeric features
fed to the Isolation Forest model in Phase 3.

Design notes:
- Every function is pure (df in -> df out) so it can be unit tested in
  isolation and re-used unchanged inside the Django module later
  (Phase 4 wires this same logic into a post_save signal).
- We do NOT recompute the proxy fraud label from scratch — insurance.ipynb
  already derived it as `is_suspicious` (settled < 80% of invoice). We
  reuse it as-is so the label stays consistent with the cleaning notebook.
- Two extra signal columns already produced by insurance.ipynb
  (`had_pre_audit_adjustment`, `icd_code_missing`) are folded into the
  final feature set because the sanity checks in Step 9 of the plan
  showed they carry real signal (91.6% vs 10.0% suspicious rate).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Known catch-all / vague ICD-10 codes seen in Kenyan insurer data.
# Expand this list if exploration turns up more "unspecified" style codes.
VAGUE_ICD_CODES = [
    "Z51.9",  # Medical care, unspecified
    "Z00.0",  # General medical examination
    "Z76.9",  # Encounter for health services, unspecified circumstances
    "Z71.9",  # Counselling, unspecified
    "Z53.9",  # Procedure/treatment not carried out, unspecified reason
]

# Final feature set fed to the Isolation Forest.
# proxy_fraud_label is included here for convenience but must be dropped
# before training (see model_training.py) — it exists for evaluation only.
FEATURE_COLUMNS = [
    "invoice_inflation_ratio",
    "claim_lag_days",
    "icd_is_vague",
    "provider_avg_inflation",
    "provider_claim_count",
    "member_claim_count",
    "amount_vs_benchmark",
    "had_pre_audit_adjustment",  # already in cleaned data — strong signal
    "icd_code_missing",          # already in cleaned data — weak but real
]

LABEL_COLUMN = "proxy_fraud_label"


# ---------------------------------------------------------------------------
# Individual feature functions — each one is independently testable
# ---------------------------------------------------------------------------

def add_invoice_inflation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """How much the provider billed vs what was actually paid.
    The single strongest signal in the dataset."""
    df = df.copy()
    settled_safe = df["settled_amount"].replace(0, np.nan)
    ratio = df["invoice_amount"] / settled_safe

    # Cap at 10x — beyond that it's already obvious fraud and would
    # distort the model's sense of scale for everything else.
    ratio = ratio.clip(upper=10.0)

    median_ratio = ratio.median()
    df["invoice_inflation_ratio"] = ratio.fillna(median_ratio)
    return df


def add_claim_lag_days(df: pd.DataFrame) -> pd.DataFrame:
    """Days between service delivery and claim submission.
    Long lags (months) are a classic backdating signal."""
    df = df.copy()
    lag = (df["claim_date"] - df["service_date"]).dt.days

    # Negative = filed before service happened -> clearly a data issue,
    # not a real "early" claim. Clip to 0 rather than dropping the row.
    lag = lag.clip(lower=0)
    df["claim_lag_days"] = lag.fillna(lag.median())
    return df


def add_icd_is_vague(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the diagnosis code is a known catch-all code."""
    df = df.copy()
    df["icd_is_vague"] = df["icd_code"].isin(VAGUE_ICD_CODES).astype(int)
    return df


def add_provider_avg_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """Per-provider mean inflation ratio. Catches providers who
    consistently overbill even when any single claim looks mild.
    Requires invoice_inflation_ratio to already be present."""
    df = df.copy()
    provider_avg = (
        df.groupby("provider_name")["invoice_inflation_ratio"]
        .transform("mean")
    )
    df["provider_avg_inflation"] = provider_avg.fillna(
        df["invoice_inflation_ratio"].median()
    )
    return df


def add_provider_claim_count(df: pd.DataFrame) -> pd.DataFrame:
    """Total claims submitted by this provider across the dataset."""
    df = df.copy()
    df["provider_claim_count"] = (
        df.groupby("provider_name")["hash_claim_no"].transform("count")
    )
    return df


def add_member_claim_count(df: pd.DataFrame) -> pd.DataFrame:
    """Total claims per member — unusually high counts can indicate
    ghost beneficiaries or identity fraud."""
    df = df.copy()
    df["member_claim_count"] = (
        df.groupby("hash_membership_no")["hash_claim_no"].transform("count")
    )
    return df


def add_amount_vs_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of this claim's invoice amount to the median invoice for
    its benefit code. A 50K outpatient claim against a 4K median is
    suspicious regardless of anything else."""
    df = df.copy()
    benefit_median = df.groupby("benefit_code")["invoice_amount"].transform("median")
    ratio = df["invoice_amount"] / benefit_median
    df["amount_vs_benchmark"] = ratio.clip(upper=10.0).fillna(1.0)
    return df


def add_proxy_label(df: pd.DataFrame) -> pd.DataFrame:
    """Reuse the proxy fraud label already computed in insurance.ipynb
    (is_suspicious = settled < 80% of invoice). If a dataset is passed
    in that doesn't have it yet, compute it fresh from the same rule."""
    df = df.copy()
    if "is_suspicious" in df.columns:
        df[LABEL_COLUMN] = df["is_suspicious"].astype(int)
    else:
        df[LABEL_COLUMN] = (
            df["settled_amount"] < df["invoice_amount"] * 0.80
        ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "invoice_amount", "settled_amount", "claim_date", "service_date",
    "icd_code", "provider_name", "hash_claim_no", "hash_membership_no",
    "benefit_code",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full Phase 2 feature engineering pipeline.

    Expects the cleaned, anonymised dataframe produced by insurance.ipynb
    (snake_case columns, dates already parsed to datetime). Returns a new
    dataframe — the input is never mutated.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataframe is missing required columns: {missing}. "
            "Did you pass the cleaned output of insurance.ipynb "
            "(anonymised_claims_data_cleaned.csv)?"
        )

    # Dates may still be strings if loaded fresh from CSV — coerce defensively.
    for col in ["claim_date", "service_date"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.copy()
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = add_invoice_inflation_ratio(df)
    df = add_claim_lag_days(df)
    df = add_icd_is_vague(df)
    df = add_provider_avg_inflation(df)
    df = add_provider_claim_count(df)
    df = add_member_claim_count(df)
    df = add_amount_vs_benchmark(df)
    df = add_proxy_label(df)

    # Backfill the two bonus signal columns if they weren't already present
    # (e.g. if someone runs this against a differently-cleaned dataset).
    if "had_pre_audit_adjustment" not in df.columns:
        if "amount_adjustment" in df.columns:
            df["had_pre_audit_adjustment"] = (df["amount_adjustment"] != 0).astype(int)
        else:
            df["had_pre_audit_adjustment"] = 0
    if "icd_code_missing" not in df.columns:
        df["icd_code_missing"] = 0

    return df


def sanity_check(df: pd.DataFrame) -> None:
    """Print fraud-rate-by-bucket checks so you can eyeball whether each
    feature actually separates suspicious from clean claims before it
    goes anywhere near the model. Mirrors Step 9 of the technical plan."""
    print("=" * 60)
    print("FEATURE SANITY CHECK — fraud rate by bucket")
    print("=" * 60)

    lag_bins = pd.cut(
        df["claim_lag_days"], bins=[0, 7, 30, 90, 180, 1e9],
        labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"],
        include_lowest=True,
    )
    print("\nclaim_lag_days:")
    print(df.groupby(lag_bins, observed=True)[LABEL_COLUMN].mean().apply(lambda x: f"{x:.1%}"))

    inf_bins = pd.cut(
        df["invoice_inflation_ratio"], bins=[0, 1, 1.5, 3, 10],
        labels=["=1", "1-1.5x", "1.5-3x", ">3x"], include_lowest=True,
    )
    print("\ninvoice_inflation_ratio:")
    print(df.groupby(inf_bins, observed=True)[LABEL_COLUMN].mean().apply(lambda x: f"{x:.1%}"))

    print("\nicd_is_vague:")
    print(df.groupby("icd_is_vague")[LABEL_COLUMN].mean().apply(lambda x: f"{x:.1%}"))

    print("\nhad_pre_audit_adjustment:")
    print(df.groupby("had_pre_audit_adjustment")[LABEL_COLUMN].mean().apply(lambda x: f"{x:.1%}"))

    print("\nicd_code_missing:")
    print(df.groupby("icd_code_missing")[LABEL_COLUMN].mean().apply(lambda x: f"{x:.1%}"))

    contamination = df[LABEL_COLUMN].mean()
    print(f"\nSuspicious rate overall: {contamination:.1%}")
    print(f"-> Use contamination={contamination:.3f} in the Isolation Forest (Phase 3)")


def save_features(df: pd.DataFrame, out_dir: str = "data") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(out_dir, "claims_features.csv")
    full_path = os.path.join(out_dir, "claims_full.csv")

    null_counts = df[FEATURE_COLUMNS].isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Nulls remain in feature columns, fix before saving:\n{null_counts}")

    df[FEATURE_COLUMNS + [LABEL_COLUMN]].to_csv(features_path, index=False)
    df.to_csv(full_path, index=False)
    return features_path, full_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 feature engineering")
    parser.add_argument(
        "--input", default="anonymised_claims_data_cleaned.csv",
        help="Path to the cleaned dataset produced by insurance.ipynb",
    )
    parser.add_argument("--out-dir", default="data", help="Directory to write outputs to")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False, parse_dates=["claim_date", "service_date"])
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    df = engineer_features(df)
    sanity_check(df)

    features_path, full_path = save_features(df, args.out_dir)
    print(f"\nSaved feature set -> {features_path}")
    print(f"Saved full dataset -> {full_path}")
    print(f"\nFeature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
