# Track 3 — Fraud Detection ML Pipeline

**What this covers:** Feature Engineering → Model Training (Isolation Forest) → Evaluation → Testing.
**What it builds on:** the cleaning work already committed in `insurance.ipynb`.
**Who should read this:** anyone on the team who needs to understand what the model does, why it makes the decisions it makes, or how to run/extend the pipeline before Phase 4 (the Django module).

---

## 1. Where this fits in the project

```
Raw claims CSV (has PII — never commit)
        │
        ▼
run_cleaning.py            ← re-runs insurance.ipynb's cleaning logic as a script
        │  data/anonymised_claims_data_cleaned.csv
        ▼
feature_engineering.py     ← THIS PHASE: turns cleaned data into 9 model features
        │  data/claims_features.csv, data/claims_full.csv
        ▼
model_training.py          ← THIS PHASE: trains + evaluates the Isolation Forest
        │  models/fraud_model.joblib, models/fraud_scaler.joblib
        ▼
tests/                     ← THIS PHASE: pytest suite, 24 tests, all passing
        │
        ▼
(Phase 4, not started) Django module — wires fraud_model.joblib into a
post_save signal so every claim gets scored automatically in openIMIS
```

Each script is runnable on its own and writes its output to `data/` or `models/`, so anyone on the team can re-run just the step they care about without re-running everything upstream.

---

## 2. Feature engineering — what we're feeding the model

9 numeric features, computed per claim:

| Feature | What it measures | Signal strength on real data |
|---|---|---|
| `invoice_inflation_ratio` | invoice ÷ settled amount (capped at 10x) | **Very strong** |
| `had_pre_audit_adjustment` | 1 if the invoice was corrected before audit | **Very strong** |
| `provider_avg_inflation` | this provider's average inflation ratio across all their claims | Moderate |
| `amount_vs_benchmark` | claim amount ÷ median amount for that benefit code | Weak |
| `claim_lag_days` | days between service date and claim filing | Weak/flat |
| `member_claim_count` | how many claims this member has filed in total | None |
| `icd_code_missing` | 1 if the diagnosis code was missing | None |
| `icd_is_vague` | 1 if the ICD code is a known catch-all (Z51.9 etc.) | **Inverted** — see caveat below |
| `provider_claim_count` | total claims this provider has submitted | None (near-zero correlation) |

"Signal strength" = how strongly the feature separates the proxy-suspicious claims from clean ones on the real 418,912-row dataset. Full correlation and permutation-importance numbers are in `models/evaluation_report.json`.

### The proxy fraud label

We don't have a real "fraud = yes/no" column — no insurer labels claims that way. `insurance.ipynb` already derived a stand-in: **a claim is "suspicious" if the insurer settled less than 80% of what was invoiced** (`is_suspicious` in the cleaned data). We reuse that column directly rather than recomputing it, so the label stays consistent with the cleaning notebook. 10.4% of real claims are "suspicious" by this definition.

**Important:** this is a proxy, not ground truth. It's a reasonable stand-in (a big settlement discount usually means the insurer already found something off), but it is not the same as a confirmed fraud label. Frame it that way to judges — "we evaluate against a proxy signal because no labelled fraud data exists" — rather than implying we know true fraud rates.

### Two caveats worth knowing before the demo

1. **Circularity in the two strongest features.** The proxy label is defined as `settled < 0.8 × invoice`. `invoice_inflation_ratio` is `invoice ÷ settled` — almost the same underlying arithmetic. So of course a model using that feature will match the label well; it's close to using the label to predict itself. `had_pre_audit_adjustment` has a similar (weaker) relationship. This doesn't make the features useless — inflated invoices and pre-audit corrections genuinely matter — but it means the headline evaluation numbers are partly measuring "did we recover the rule the label was built from" rather than "did we find fraud no one was looking for." We trained a second **conservative model** with those two features removed specifically to show this honestly (see Section 3).

2. **`icd_is_vague` doesn't behave as hypothesised.** The plan assumed vague ICD codes (Z51.9 etc.) would correlate with fraud. On the real data it's the opposite — vague codes have a *slightly lower* suspicious rate (6.7% vs 10.5%). We kept the feature in because Isolation Forest is unsupervised and can still use unusual code usage as a signal even if it doesn't track this particular proxy label, but don't present it in the demo as a validated fraud indicator — it wasn't, at least not against this label.

---

## 3. Model training — Isolation Forest

Unsupervised anomaly detection — the model never sees the proxy label during training, only at evaluation time.

We trained **two variants** on purpose:

| | Full model (9 features) | Conservative model (7 features — drops `invoice_inflation_ratio` and `had_pre_audit_adjustment`) |
|---|---|---|
| Precision (suspicious class) | 0.32 | 0.13 |
| Recall (suspicious class) | 0.32 | 0.13 |
| F1 | 0.32 | 0.13 |
| ROC-AUC | **0.80** | 0.53 (barely above chance) |
| Contamination parameter | 0.104 | 0.104 |

**The full model is what ships in the demo** (`models/fraud_model.joblib`). The conservative model exists to make the circularity caveat concrete and defensible if a judge pushes on it — it shows how much of the 0.80 AUC comes from the two label-adjacent features versus genuine anomaly detection on the rest.

**Confusion matrix, full model, test set (n=83,783):**

|  | Predicted normal | Predicted suspicious |
|---|---|---|
| **Actually normal** | 69,179 (TN) | 5,879 (FP) |
| **Actually suspicious** | 5,930 (FN) | 2,795 (TP) |

**How to talk about this honestly in the demo:** lead with "we catch about a third of proxy-suspicious claims, and about a third of what we flag is a false alarm — here's the confusion matrix" rather than only quoting the AUC. It's a real, non-trivial result (AUC 0.80 vs 0.50 random baseline) without overselling it.

---

## 4. Testing

`tests/` — 24 pytest tests, all passing.

**`test_feature_engineering.py` (19 tests)** — each feature function tested in isolation with small hand-built dataframes:
- Correct arithmetic for every feature, including edge cases: division by zero on `settled_amount = 0`, negative lag from bad dates, the 10x cap on inflation ratio and benchmark ratio
- Grouped features (`provider_avg_inflation`, `provider_claim_count`, `member_claim_count`) group by the right key
- The proxy label reuses `is_suspicious` when present instead of silently recomputing it
- Full pipeline: produces all expected columns, zero nulls in the feature set, doesn't mutate its input, raises a clear error on malformed input

**`test_model.py` (5 tests)** — loads the actual trained `fraud_model.joblib` and `fraud_scaler.joblib` (not mocks) and checks:
- A synthetic "obviously fraudulent" claim (8x inflation, filed 240 days late, vague ICD, high-inflation provider, pre-audit adjustment) is predicted as an anomaly
- A synthetic "obviously clean" claim (invoice = settlement, filed within days, clean provider history) is predicted as normal
- The fraud claim's anomaly score is lower (more anomalous) than the clean claim's
- Scoring is deterministic — same input always gives the same output, which matters once this runs inside a Django signal in production
- The scaler and model agree on feature count, so a schema drift fails loudly instead of silently corrupting scores

Run with:
```bash
pip install -r requirements.txt   # pandas, scikit-learn, joblib, pytest, matplotlib
pytest tests/ -v
```

---

## 5. Files in this delivery

| File | Purpose |
|---|---|
| `run_cleaning.py` | Re-runs `insurance.ipynb`'s anonymisation/cleaning as a script — same logic, scriptable |
| `feature_engineering.py` | Computes the 9 features + proxy label from cleaned data |
| `model_training.py` | Trains + evaluates both Isolation Forest variants, saves model/scaler/report |
| `tests/conftest.py` | Lets pytest import root-level modules |
| `tests/test_feature_engineering.py` | Unit tests for every feature function |
| `tests/test_model.py` | Loads the real trained model, checks fraud vs clean scoring |
| `models/fraud_model.joblib` | Trained Isolation Forest (full, 9-feature version — this is what ships) |
| `models/fraud_model_conservative.joblib` | The honesty-check variant, 7 features |
| `models/fraud_scaler.joblib` / `fraud_scaler_conservative.joblib` | `StandardScaler`s fit on the training set — required to score any new claim the same way |
| `models/feature_columns.json` | Exact feature order both variants expect — needed at inference time |
| `models/evaluation_report.json` | Full precision/recall/F1/AUC/permutation-importance numbers for both variants |
| `plots/feature_distributions_by_label.png` | Histogram of every feature split by proxy label |
| `plots/feature_importance_full.png` / `_conservative.png` | Permutation importance bar charts |

---

## 6. What's next (Phase 4, not part of this delivery)

Per the technical plan, the next step is wrapping `fraud_model.joblib` in an openIMIS Django module: a `post_save` signal scores each claim when it's saved, storing the result in a `FraudFlag` model and exposing it over GraphQL so the reviewer UI can show the risk badge. `feature_engineering.py`'s functions are written as pure, single-purpose functions specifically so that logic can be lifted into that signal handler without rewriting it.
