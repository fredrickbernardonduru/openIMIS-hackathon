"""
model_training.py
Track 3 — AI-Powered Claims Anomaly Detector — Phase 3

Trains an Isolation Forest on the engineered features (Phase 2 output) and
evaluates it against the proxy fraud label. The proxy label is NEVER shown
to the model during training — it is held out purely for post-hoc scoring,
exactly as required by an unsupervised approach.

Caveat carried over from feature_engineering.py and worth repeating here:
`invoice_inflation_ratio` and `had_pre_audit_adjustment` are structurally
close to how the proxy label itself is defined (settled < 80% of invoice).
That inflates precision/recall against THIS label. We train and report
two variants so that's visible rather than hidden:

  - full_model      : all 9 features (what ships in the demo)
  - conservative_model: drops the two label-adjacent features, to show
                        how much genuine anomaly-detection signal exists
                        in the *other* 7 features on their own
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/claims_features.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

ALL_FEATURES = [
    "invoice_inflation_ratio",
    "claim_lag_days",
    "icd_is_vague",
    "provider_avg_inflation",
    "provider_claim_count",
    "member_claim_count",
    "amount_vs_benchmark",
    "had_pre_audit_adjustment",
    "icd_code_missing",
]

LABEL_ADJACENT_FEATURES = ["invoice_inflation_ratio", "had_pre_audit_adjustment"]
CONSERVATIVE_FEATURES = [f for f in ALL_FEATURES if f not in LABEL_ADJACENT_FEATURES]

LABEL_COLUMN = "proxy_fraud_label"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    return df


def train_and_evaluate(df: pd.DataFrame, feature_cols: list[str], name: str) -> dict:
    """Train one Isolation Forest variant and return its metrics + artifacts."""
    X = df[feature_cols]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    contamination = round(y_train.mean(), 3)
    contamination = min(max(contamination, 0.01), 0.5)  # sklearn's valid range

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)

    test_scores = model.decision_function(X_test_scaled)  # higher = more normal
    test_pred_raw = model.predict(X_test_scaled)           # -1 anomaly, 1 normal
    test_pred = np.where(test_pred_raw == -1, 1, 0)         # 1 = flagged as suspicious

    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    # AUC uses the continuous anomaly score (flip sign: more negative = more anomalous)
    auc = roc_auc_score(y_test, -test_scores)

    cm = confusion_matrix(y_test, test_pred)
    report = classification_report(y_test, test_pred, target_names=["Normal", "Suspicious"], zero_division=0)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Contamination used: {contamination}")
    print(report)
    print("Confusion matrix:")
    print(f"  True Negatives  (correctly cleared): {cm[0][0]:,}")
    print(f"  False Positives (wrongly flagged)  : {cm[0][1]:,}")
    print(f"  False Negatives (missed)           : {cm[1][0]:,}")
    print(f"  True Positives  (correctly caught) : {cm[1][1]:,}")
    print(f"ROC-AUC (score vs proxy label): {auc:.3f}")

    # Permutation importance — how much each feature actually drives the score.
    # IsolationForest has no built-in .score(), so we supply a custom scorer:
    # F1 of the model's flag vs the TRUE proxy label (not its own predictions,
    # which would be circular).
    def iso_forest_f1_scorer(estimator, X, y):
        pred_raw = estimator.predict(X)
        pred = np.where(pred_raw == -1, 1, 0)
        return f1_score(y, pred, zero_division=0)

    perm = permutation_importance(
        model, X_test_scaled, y_test, scoring=iso_forest_f1_scorer,
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importances = dict(zip(feature_cols, perm.importances_mean.round(4)))

    return {
        "name": name,
        "feature_cols": feature_cols,
        "contamination": contamination,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "permutation_importance": importances,
        "model": model,
        "scaler": scaler,
    }


def plot_feature_importance(result: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(result["permutation_importance"].items(), key=lambda kv: kv[1])
    labels, values = zip(*items)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color="#1976D2")
    ax.set_title(f"Feature Importance (permutation) — {result['name']}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df):,} rows for training")

    full_result = train_and_evaluate(df, ALL_FEATURES, "FULL MODEL (9 features)")
    cons_result = train_and_evaluate(df, CONSERVATIVE_FEATURES, "CONSERVATIVE MODEL (7 features, label-adjacent dropped)")

    plot_feature_importance(full_result, f"{PLOTS_DIR}/feature_importance_full.png")
    plot_feature_importance(cons_result, f"{PLOTS_DIR}/feature_importance_conservative.png")

    # Save the full model — this is what ships in the Django module
    joblib.dump(full_result["model"], f"{MODELS_DIR}/fraud_model.joblib")
    joblib.dump(full_result["scaler"], f"{MODELS_DIR}/fraud_scaler.joblib")
    joblib.dump(cons_result["model"], f"{MODELS_DIR}/fraud_model_conservative.joblib")
    joblib.dump(cons_result["scaler"], f"{MODELS_DIR}/fraud_scaler_conservative.joblib")

    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump({"full": ALL_FEATURES, "conservative": CONSERVATIVE_FEATURES}, f, indent=2)

    summary = {
        "full_model": {k: v for k, v in full_result.items() if k not in ("model", "scaler")},
        "conservative_model": {k: v for k, v in cons_result.items() if k not in ("model", "scaler")},
    }
    with open(f"{MODELS_DIR}/evaluation_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model      -> {MODELS_DIR}/fraud_model.joblib")
    print(f"Saved scaler     -> {MODELS_DIR}/fraud_scaler.joblib")
    print(f"Saved report     -> {MODELS_DIR}/evaluation_report.json")


if __name__ == "__main__":
    main()
