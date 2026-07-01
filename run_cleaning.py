"""
run_cleaning.py

Faithful re-run of the data cleaning + anonymisation logic already
committed to the repo in insurance.ipynb ("Data cleaning and preprocessing
of the claims data"). This is NOT a redesign — it mirrors that notebook's
steps 1:1 so the output schema matches what the rest of the team expects.

Input : raw OVERALL_CLAIMS_PAID_ANALYSIS CSV (contains PII — never commit)
Output: data/anonymised_claims_data_cleaned.csv (safe to commit)
"""

import hashlib
import sys

import numpy as np
import pandas as pd

RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "OVERALL_CLAIMS_PAID_ANALYSIS.csv"
OUT_PATH = "data/anonymised_claims_data_cleaned.csv"


def sha_hash(x):
    return hashlib.sha256(str(x).encode()).hexdigest()[:12] if pd.notna(x) else "unknown"


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Loaded raw: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Anonymise -----------------------------------------------------
    df = df.drop(columns=["PATIENT NAME", "PRINCIPAL MEMBER"])
    df["hash_membership_no"] = df["MEMBERSHIP NO"].apply(sha_hash)
    df["hash_invoice_no"] = df["INVOICE NUMBER"].apply(sha_hash)
    df["hash_claim_no"] = df["CLAIM NO"].apply(sha_hash)
    df["hash_policy_code"] = df["POLICY CODE"].apply(sha_hash)
    df = df.drop(columns=["MEMBERSHIP NO", "INVOICE NUMBER", "CLAIM NO", "POLICY CODE"])
    print(f"Anonymised: {df.shape}")

    # --- Normalise column names ----------------------------------------
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- Dates -----------------------------------------------------------
    for col in ["claim_date", "claim_audit_date", "service_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # claim_audit_date: recover Excel-serial values that failed to parse
    mask = df["claim_audit_date"].isna()
    if mask.any():
        raw_audit = pd.read_csv(RAW_PATH, usecols=["CLAIM AUDIT DATE"])["CLAIM AUDIT DATE"]
        excel_serials = pd.to_numeric(raw_audit[mask], errors="coerce")
        excel_dates = pd.Timestamp("1899-12-30") + pd.to_timedelta(excel_serials, unit="D")
        df.loc[mask, "claim_audit_date"] = excel_dates.values

    # service_date: flag rows where the date itself was invalid (e.g. 29 Feb
    # on a non-leap year) BEFORE losing the information, per insurance.ipynb
    df["service_date_invalid"] = df["service_date"].isna().astype(int)
    # Back-fill so downstream features (claim_lag_days, etc.) don't break.
    # Use claim_date as the fallback (lag=0, conservative); the fraud
    # signal itself is preserved in service_date_invalid=1.
    df["service_date"] = df["service_date"].fillna(df["claim_date"])

    # --- Duplicates --------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(df):,} exact duplicate rows")

    # --- Currency filter: keep KES only -------------------------------------
    kes_mask = (df["prod_currency"] == "KES") & (df["inv_currency"] == "KES")
    df = df[kes_mask].reset_index(drop=True)
    print(f"After KES-only filter: {df.shape[0]:,} rows")

    # --- Pre-audit adjustment signal ----------------------------------------
    df["amount_adjustment"] = df["invoice_amount"] - df["inv_amount"]
    df["had_pre_audit_adjustment"] = (df["invoice_amount"] != df["inv_amount"]).astype(int)

    # --- Proxy fraud label ---------------------------------------------------
    df["is_suspicious"] = (df["settled_amount"] < df["invoice_amount"] * 0.80).astype(int)

    # --- Missing values --------------------------------------------------------
    df["icd_code_missing"] = df["icd_code"].isna().astype(int)
    df["plan_name"] = df["plan_name"].fillna("UNKNOWN")
    df["group_name"] = df["group_name"].fillna("INDIVIDUAL")
    df["icd_code"] = df["icd_code"].fillna("UNKNOWN")
    df["diagnosis"] = df["diagnosis"].fillna("UNKNOWN")
    df["payee"] = df["payee"].fillna("UNKNOWN")
    df["notes"] = df["notes"].fillna("")
    df = df[df["benefit_code"].notna()].reset_index(drop=True)
    df["diagnosis_code"] = df["diagnosis_code"].fillna("OTHERS")
    df = df.drop(columns=["doctor"])

    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    print(f"Remaining nulls after cleaning:\n{remaining_nulls if len(remaining_nulls) else '  none'}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {OUT_PATH}  ({df.shape[0]:,} rows x {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
