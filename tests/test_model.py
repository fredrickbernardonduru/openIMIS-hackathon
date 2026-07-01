"""
tests/test_model.py

Loads the trained Isolation Forest (models/fraud_model.joblib) and its
scaler, then feeds it two synthetic, hand-built claims:

  - an "obviously fraudulent" claim: heavily inflated invoice, filed
    months late, provider with a history of inflating, pre-audit
    adjustment needed
  - an "obviously clean" claim: invoice matches settlement, filed
    within days, provider with a normal track record

The model was never trained on these exact rows — they're synthetic —
but its job is to separate points like this, so we check it does.

Requires model_training.py to have been run first (models/fraud_model.joblib
etc. must exist). If they don't, these tests are skipped rather than failed,
since "model not trained yet" is a different problem than "model is wrong."
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "fraud_scaler.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)),
    reason="Trained model artifacts not found — run model_training.py first",
)


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def scaler():
    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="module")
def feature_cols():
    with open(FEATURE_COLUMNS_PATH) as f:
        return json.load(f)["full"]


@pytest.fixture(scope="module")
def real_medians(feature_cols):
    """Pull realistic 'normal' values for the weak-signal features straight
    from the actual training data, so the synthetic rows aren't built on
    arbitrary guesses for the features that don't matter to the test's intent."""
    path = "data/claims_features.csv"
    if not os.path.exists(path):
        # Fallback if the CSV isn't present in this environment
        return {
            "provider_claim_count": 12841,
            "member_claim_count": 6,
            "amount_vs_benchmark": 1.0,
        }
    df = pd.read_csv(path)
    return {
        "provider_claim_count": df["provider_claim_count"].median(),
        "member_claim_count": df["member_claim_count"].median(),
        "amount_vs_benchmark": df["amount_vs_benchmark"].median(),
    }


def build_claim(feature_cols, real_medians, fraudulent: bool) -> pd.DataFrame:
    if fraudulent:
        values = {
            "invoice_inflation_ratio": 8.0,        # billed 8x what was settled
            "claim_lag_days": 240,                  # filed 8 months late
            "icd_is_vague": 1,                      # catch-all diagnosis code
            "provider_avg_inflation": 5.0,          # provider has a bad history
            "provider_claim_count": real_medians["provider_claim_count"],
            "member_claim_count": real_medians["member_claim_count"],
            "amount_vs_benchmark": 6.0,             # 6x the median for this benefit
            "had_pre_audit_adjustment": 1,
            "icd_code_missing": 0,
        }
    else:
        values = {
            "invoice_inflation_ratio": 1.0,         # invoice == settlement
            "claim_lag_days": 2,                    # filed within days
            "icd_is_vague": 0,
            "provider_avg_inflation": 1.02,
            "provider_claim_count": real_medians["provider_claim_count"],
            "member_claim_count": real_medians["member_claim_count"],
            "amount_vs_benchmark": real_medians["amount_vs_benchmark"],
            "had_pre_audit_adjustment": 0,
            "icd_code_missing": 0,
        }
    # Preserve the exact column order the scaler/model were fit on
    return pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)


def test_fraudulent_claim_is_flagged_as_anomaly(model, scaler, feature_cols, real_medians):
    claim = build_claim(feature_cols, real_medians, fraudulent=True)
    scaled = scaler.transform(claim)
    prediction = model.predict(scaled)[0]
    assert prediction == -1, "obviously fraudulent claim should be predicted as an anomaly"


def test_clean_claim_is_not_flagged(model, scaler, feature_cols, real_medians):
    claim = build_claim(feature_cols, real_medians, fraudulent=False)
    scaled = scaler.transform(claim)
    prediction = model.predict(scaled)[0]
    assert prediction == 1, "obviously clean claim should be predicted as normal"


def test_fraudulent_claim_scores_more_anomalous_than_clean_claim(model, scaler, feature_cols, real_medians):
    fraud_claim = build_claim(feature_cols, real_medians, fraudulent=True)
    clean_claim = build_claim(feature_cols, real_medians, fraudulent=False)

    fraud_score = model.decision_function(scaler.transform(fraud_claim))[0]
    clean_score = model.decision_function(scaler.transform(clean_claim))[0]

    # decision_function: higher = more normal, lower/more negative = more anomalous
    assert fraud_score < clean_score


def test_model_output_is_deterministic(model, scaler, feature_cols, real_medians):
    """Same input in -> same output out. Matters because this model gets
    called from a Django signal in production; flaky scoring would be a bug."""
    claim = build_claim(feature_cols, real_medians, fraudulent=True)
    scaled = scaler.transform(claim)
    score_1 = model.decision_function(scaled)[0]
    score_2 = model.decision_function(scaled)[0]
    assert score_1 == score_2


def test_scaler_and_model_agree_on_feature_count(model, scaler, feature_cols):
    assert scaler.n_features_in_ == len(feature_cols)
    assert model.n_features_in_ == len(feature_cols)
