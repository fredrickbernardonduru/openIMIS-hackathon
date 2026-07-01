"""
tests/test_feature_engineering.py

Unit tests for feature_engineering.py. Each feature function is tested in
isolation with small, hand-built dataframes so the expected output can be
reasoned about by eye — no dependency on the real claims CSV.
"""

import numpy as np
import pandas as pd
import pytest

import feature_engineering as fe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_claims_df():
    """A small, fully-formed dataframe matching the cleaned schema that
    engineer_features() expects."""
    return pd.DataFrame({
        "invoice_amount": [1000.0, 5000.0, 2000.0, 3000.0],
        "settled_amount": [1000.0, 2000.0, 2000.0, 0.0],
        "claim_date": pd.to_datetime(["2025-01-10", "2025-06-01", "2025-01-05", "2025-02-01"]),
        "service_date": pd.to_datetime(["2025-01-08", "2025-01-01", "2025-01-01", "2025-01-30"]),
        "icd_code": ["A01", "Z51.9", "B02", "Z00.0"],
        "provider_name": ["Hosp A", "Hosp A", "Hosp B", "Hosp B"],
        "hash_claim_no": ["c1", "c2", "c3", "c4"],
        "hash_membership_no": ["m1", "m1", "m2", "m3"],
        "benefit_code": ["OUTP", "OUTP", "INP", "OUTP"],
        "had_pre_audit_adjustment": [0, 1, 0, 0],
        "icd_code_missing": [0, 0, 0, 0],
    })


# ---------------------------------------------------------------------------
# invoice_inflation_ratio
# ---------------------------------------------------------------------------

def test_invoice_inflation_ratio_basic_division(minimal_claims_df):
    out = fe.add_invoice_inflation_ratio(minimal_claims_df)
    # row 0: 1000/1000 = 1.0 ; row 1: 5000/2000 = 2.5
    assert out.loc[0, "invoice_inflation_ratio"] == pytest.approx(1.0)
    assert out.loc[1, "invoice_inflation_ratio"] == pytest.approx(2.5)


def test_invoice_inflation_ratio_caps_at_10x():
    df = pd.DataFrame({"invoice_amount": [10000.0], "settled_amount": [10.0]})
    out = fe.add_invoice_inflation_ratio(df)
    assert out.loc[0, "invoice_inflation_ratio"] == 10.0


def test_invoice_inflation_ratio_zero_settled_does_not_crash(minimal_claims_df):
    # row 3 has settled_amount == 0 -> would be a division by zero if not guarded
    out = fe.add_invoice_inflation_ratio(minimal_claims_df)
    assert not out["invoice_inflation_ratio"].isna().any()
    assert np.isfinite(out["invoice_inflation_ratio"]).all()


# ---------------------------------------------------------------------------
# claim_lag_days
# ---------------------------------------------------------------------------

def test_claim_lag_days_basic(minimal_claims_df):
    out = fe.add_claim_lag_days(minimal_claims_df)
    assert out.loc[0, "claim_lag_days"] == 2   # Jan 10 - Jan 8
    assert out.loc[1, "claim_lag_days"] == 151  # Jun 1 - Jan 1 (2025, non-leap)


def test_claim_lag_days_negative_lag_clipped_to_zero():
    df = pd.DataFrame({
        "claim_date": pd.to_datetime(["2025-01-01"]),
        "service_date": pd.to_datetime(["2025-01-15"]),  # filed BEFORE service — bad data
    })
    out = fe.add_claim_lag_days(df)
    assert out.loc[0, "claim_lag_days"] == 0


# ---------------------------------------------------------------------------
# icd_is_vague
# ---------------------------------------------------------------------------

def test_icd_is_vague_flags_known_catchall_codes(minimal_claims_df):
    out = fe.add_icd_is_vague(minimal_claims_df)
    assert out["icd_is_vague"].tolist() == [0, 1, 0, 1]  # Z51.9 and Z00.0 are vague


def test_icd_is_vague_handles_unseen_code_as_specific():
    df = pd.DataFrame({"icd_code": ["Q99.9"]})
    out = fe.add_icd_is_vague(df)
    assert out.loc[0, "icd_is_vague"] == 0


# ---------------------------------------------------------------------------
# provider_avg_inflation
# ---------------------------------------------------------------------------

def test_provider_avg_inflation_groups_correctly(minimal_claims_df):
    df = fe.add_invoice_inflation_ratio(minimal_claims_df)
    out = fe.add_provider_avg_inflation(df)
    # Hosp A: rows 0,1 -> ratios 1.0, 2.5 -> mean 1.75
    assert out.loc[0, "provider_avg_inflation"] == pytest.approx(1.75)
    assert out.loc[1, "provider_avg_inflation"] == pytest.approx(1.75)
    # Hosp B: rows 2,3 -> ratios 1.0, inf-capped... just check both rows match each other
    assert out.loc[2, "provider_avg_inflation"] == pytest.approx(out.loc[3, "provider_avg_inflation"])


# ---------------------------------------------------------------------------
# provider_claim_count / member_claim_count
# ---------------------------------------------------------------------------

def test_provider_claim_count(minimal_claims_df):
    out = fe.add_provider_claim_count(minimal_claims_df)
    assert out.loc[0, "provider_claim_count"] == 2  # Hosp A appears twice
    assert out.loc[2, "provider_claim_count"] == 2  # Hosp B appears twice


def test_member_claim_count(minimal_claims_df):
    out = fe.add_member_claim_count(minimal_claims_df)
    assert out.loc[0, "member_claim_count"] == 2  # m1 appears twice (rows 0,1)
    assert out.loc[2, "member_claim_count"] == 1  # m2 appears once


# ---------------------------------------------------------------------------
# amount_vs_benchmark
# ---------------------------------------------------------------------------

def test_amount_vs_benchmark_ratio(minimal_claims_df):
    out = fe.add_amount_vs_benchmark(minimal_claims_df)
    # OUTP benefit_code rows: 0 (1000), 1 (5000), 3 (3000) -> median = 3000
    assert out.loc[0, "amount_vs_benchmark"] == pytest.approx(1000 / 3000)
    assert out.loc[3, "amount_vs_benchmark"] == pytest.approx(1.0)


def test_amount_vs_benchmark_caps_at_10x():
    df = pd.DataFrame({
        "invoice_amount": [100.0, 100.0, 100000.0],
        "benefit_code": ["X", "X", "X"],
    })
    out = fe.add_amount_vs_benchmark(df)
    assert out["amount_vs_benchmark"].max() <= 10.0


# ---------------------------------------------------------------------------
# proxy label
# ---------------------------------------------------------------------------

def test_proxy_label_reuses_is_suspicious_if_present():
    df = pd.DataFrame({
        "invoice_amount": [1000.0],
        "settled_amount": [1000.0],
        "is_suspicious": [1],  # deliberately inconsistent with the raw amounts
    })
    out = fe.add_proxy_label(df)
    assert out[fe.LABEL_COLUMN].iloc[0] == 1  # trusts is_suspicious, doesn't recompute


def test_proxy_label_computed_fresh_when_missing():
    df = pd.DataFrame({
        "invoice_amount": [1000.0, 1000.0],
        "settled_amount": [500.0, 950.0],  # 50% settled (suspicious) vs 95% (clean)
    })
    out = fe.add_proxy_label(df)
    assert out[fe.LABEL_COLUMN].tolist() == [1, 0]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_engineer_features_produces_all_expected_columns(minimal_claims_df):
    out = fe.engineer_features(minimal_claims_df)
    for col in fe.FEATURE_COLUMNS + [fe.LABEL_COLUMN]:
        assert col in out.columns, f"missing expected column: {col}"


def test_engineer_features_has_no_nulls_in_feature_columns(minimal_claims_df):
    out = fe.engineer_features(minimal_claims_df)
    assert out[fe.FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_engineer_features_does_not_mutate_input(minimal_claims_df):
    original_cols = set(minimal_claims_df.columns)
    fe.engineer_features(minimal_claims_df)
    assert set(minimal_claims_df.columns) == original_cols


def test_engineer_features_raises_on_missing_required_columns():
    df = pd.DataFrame({"invoice_amount": [100.0]})  # missing almost everything
    with pytest.raises(ValueError, match="missing required columns"):
        fe.engineer_features(df)


def test_engineer_features_coerces_string_dates():
    df = pd.DataFrame({
        "invoice_amount": [1000.0],
        "settled_amount": [1000.0],
        "claim_date": ["2025-01-10"],       # string, not datetime
        "service_date": ["2025-01-08"],     # string, not datetime
        "icd_code": ["A01"],
        "provider_name": ["Hosp A"],
        "hash_claim_no": ["c1"],
        "hash_membership_no": ["m1"],
        "benefit_code": ["OUTP"],
    })
    out = fe.engineer_features(df)
    assert out.loc[0, "claim_lag_days"] == 2
