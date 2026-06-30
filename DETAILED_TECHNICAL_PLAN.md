# Track 3 — AI-Powered Claims Anomaly Detector
## Full Technical Development Plan

This document lists every technical step required to build and deliver the combined Rules Engine + Isolation Forest fraud detection module for openIMIS, in the exact order they should be done. Each step builds on the one before it. Do not skip steps or reorder them — the later phases assume the earlier ones are complete.

---

## Phase 0 — Environment Setup and Codebase Orientation

Before writing a single line of your own code, you need to understand the platform you are building on. Skipping this phase is the most common reason hackathon teams produce modules that do not integrate properly with openIMIS.

### Step 0.1 — Verify the Docker stack is running

Run the following and confirm all containers are healthy:

```bash
cd openimis-dist_dkr
docker compose up -d
docker compose ps
```

All services should show status `running`. Open `http://localhost` in your browser and confirm the openIMIS login screen appears. Log in with `admin / admin`.

### Step 0.2 — Clone the backend repository alongside the dist repo

The Docker Compose stack mounts the backend as a volume. You need the source code locally so you can read it:

```bash
git clone https://github.com/openimis/openimis-be_py.git
cd openimis-be_py
```

### Step 0.3 — Understand the openIMIS claims database schema

Before building anything that touches claims, you must know how claims are stored. Open a psql shell into the running database:

```bash
docker compose exec db psql -U postgres -d openimis
```

Run the following queries and read the output carefully:

```sql
-- List all claim-related tables
SELECT table_name FROM information_schema.tables
WHERE table_name LIKE '%claim%' OR table_name LIKE '%tbl_Claim%';

-- Inspect the main claims table structure
\d "tbl_Claim"

-- See a few sample claim rows
SELECT * FROM "tbl_Claim" LIMIT 5;

-- Inspect claim items (individual service line items within a claim)
\d "tbl_ClaimItem"

-- Inspect claim services
\d "tbl_ClaimService"

-- See how health facilities (providers) are stored
\d "tbl_HFacility"

-- See how insurees (patients) are stored
\d "tbl_Insuree"
```

Write down the column names you find. You will need to map them to the columns in the custom CSV. The key mappings are:
- `ClaimDate` in openIMIS = `CLAIM DATE` in the CSV
- `DateFrom` / `DateTo` in openIMIS = `SERVICE DATE` in the CSV
- `ClaimedAmount` in openIMIS = `INVOICE AMOUNT` in the CSV
- `ApprovedAmount` in openIMIS = `SETTLED AMOUNT` in the CSV
- `ICDID` / `ICDID1` in openIMIS = `ICD CODE` in the CSV (foreign key to `tbl_ICDCodes`)
- `HFacilityID` in openIMIS = `PROVIDER NAME` in the CSV (foreign key to `tbl_HFacility`)
- `InsureeID` in openIMIS = `MEMBERSHIP NO` in the CSV

### Step 0.4 — Read an existing openIMIS module to understand the pattern

Find an existing simple module in the backend repo and read its structure end to end. The `openimis-be-location_py` or `openimis-be-medical_py` modules are good examples. For each module, read:

- `openimis.json` — how it registers itself
- `apps.py` — the AppConfig class
- `models.py` — the Django database models
- `schema.py` — the GraphQL queries and mutations
- `views.py` — the REST API views
- `signals.py` — if present, how it hooks into core events
- `tests/` directory — how tests are structured

You are not copying this code. You are learning the conventions so your module looks native to the platform, which directly affects the openIMIS Integration score (25 pts).

### Step 0.5 — Read how `openimis.json` works

In the root of `openimis-be_py`, open `openimis.json`. This file lists every module that is loaded at startup. Study its format. Your module will need an entry here. A typical entry looks like:

```json
{
  "name": "fraud_detect",
  "pip": "openimis-be-fraud-detect"
}
```

### Step 0.6 — Install Python dependencies for data science work

Outside Docker, in your local Python environment, install the libraries you will use for data analysis and model training:

```bash
pip install pandas scikit-learn joblib matplotlib seaborn rapidfuzz jupyter
```

---

## Phase 1 — Data Exploration and Anonymisation

This phase turns it into a clean, anonymised, analysis-ready dataset. This must be done before any model training.

### Step 1.1 — Load the CSV and do an initial inspection

Create a Jupyter notebook called `data_exploration.ipynb` (this is for analysis only — it will not be committed to the main repo). Load the data and run basic inspection:

```python
import pandas as pd

df = pd.read_csv("claims_raw.csv")

print(df.shape)           # how many rows and columns
print(df.dtypes)          # data type of each column
print(df.head(10))        # first 10 rows
print(df.isnull().sum())  # how many missing values per column
print(df.describe())      # statistical summary of numeric columns
```

Write down answers to these questions:
- How many total claim rows are there?
- What date range do the claims span?
- How many unique providers (`PROVIDER NAME`) are there?
- How many unique members (`MEMBERSHIP NO`) are there?
- What is the distribution of `CLAIM TYPE` (OP, IP, etc.)?
- What percentage of claims have `SETTLED AMOUNT < INVOICE AMOUNT`?

### Step 1.2 — Anonymise the dataset (A dataset with hashed values already exists [here](https://drive.google.com/file/d/1CAb1zCSEVGtDnnYF4TgEg5CcjiATZ4kn/view?usp=sharing) )

Create a script called `anonymise.py`. It must:

1. Drop columns that directly identify individuals:
   ```python
   columns_to_drop = ["PATIENT NAME", "PRINCIPAL MEMBER"]
   df = df.drop(columns=columns_to_drop)
   ```

2. Hash the membership number so you can still track per-member patterns without exposing the real ID:
   ```python
   import hashlib
   df["MEMBER_HASH"] = df["MEMBERSHIP NO"].apply(
       lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]
   )
   df = df.drop(columns=["MEMBERSHIP NO"])
   ```

3. Save the anonymised version:
   ```python
   df.to_csv("claims_anonymised.csv", index=False)
   ```

The file `claims_raw.csv` must never be committed to GitHub. Add it to `.gitignore` immediately:

```
claims_raw.csv
*.csv.bak
```

Only `claims_anonymised.csv` goes into the repository, and only into a `data/` folder with a clear note in the README about what anonymisation was applied.

### Step 1.3 — Parse and clean date columns

Date columns in the CSV are strings. Convert them to proper datetime objects and fix any inconsistencies:

```python
df["CLAIM DATE"] = pd.to_datetime(df["CLAIM DATE"], dayfirst=True, errors="coerce")
df["SERVICE DATE"] = pd.to_datetime(df["SERVICE DATE"], dayfirst=True, errors="coerce")
df["CLAIM AUDIT DATE"] = pd.to_datetime(df["CLAIM AUDIT DATE"], dayfirst=True, errors="coerce")

# Check how many rows failed to parse (will show as NaT)
print(df["CLAIM DATE"].isna().sum())
print(df["SERVICE DATE"].isna().sum())
```

For rows where SERVICE DATE is null or clearly wrong (e.g. year 1900), decide whether to drop them or impute them. Document your decision — you will need to explain it in the model card.

### Step 1.4 — Parse and clean numeric columns

The INVOICE AMOUNT and SETTLED AMOUNT columns may have commas, currency symbols, or whitespace. Clean them:

```python
for col in ["INVOICE AMOUNT", "SETTLED AMOUNT", "INV AMOUNT", "PAYABLE AMOUNT"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "")
        .str.replace("KES", "")
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

### Step 1.5 — Explore the distribution of key columns

Generate and save plots so you understand what "normal" looks like before you try to find anomalies:

```python
import matplotlib.pyplot as plt

# Distribution of invoice amounts
df["INVOICE AMOUNT"].hist(bins=50)
plt.title("Distribution of Invoice Amounts")
plt.savefig("plots/invoice_distribution.png")

# Distribution of claim lag (days between service and claim)
df["claim_lag_days"] = (df["CLAIM DATE"] - df["SERVICE DATE"]).dt.days
df["claim_lag_days"].hist(bins=50)
plt.title("Distribution of Claim Lag (days)")
plt.savefig("plots/claim_lag_distribution.png")

# Top 10 ICD codes by frequency
df["ICD CODE"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 ICD Codes")
plt.savefig("plots/icd_codes.png")

# Invoice vs Settled scatter
plt.scatter(df["INVOICE AMOUNT"], df["SETTLED AMOUNT"], alpha=0.3)
plt.xlabel("Invoice Amount")
plt.ylabel("Settled Amount")
plt.title("Invoice vs Settled Amount")
plt.savefig("plots/invoice_vs_settled.png")
```

Study these plots. The outliers you see visually are exactly what the Isolation Forest will learn to find programmatically.

---

## Phase 2 — Feature Engineering

Features are the numeric inputs to the machine learning model. This phase calculates all of them from the raw data. Good features = good model. Every feature must have a clear fraud-related meaning.

### Step 2.1 — Feature: Invoice inflation ratio

This is the single most important feature. It measures how much the hospital billed versus how much was actually paid:

```python
# Avoid division by zero — replace 0 settled amounts with NaN first
df["SETTLED AMOUNT"] = df["SETTLED AMOUNT"].replace(0, pd.NA)
df["invoice_inflation_ratio"] = df["INVOICE AMOUNT"] / df["SETTLED AMOUNT"]

# Cap extreme values — a ratio above 10 is already obviously fraudulent
# Capping prevents a single extreme outlier from distorting the model
df["invoice_inflation_ratio"] = df["invoice_inflation_ratio"].clip(upper=10)

# Fill missing values (where settled was 0 or null) with the median
median_ratio = df["invoice_inflation_ratio"].median()
df["invoice_inflation_ratio"] = df["invoice_inflation_ratio"].fillna(median_ratio)
```

### Step 2.2 — Feature: Claim lag in days

The number of days between when the service was delivered and when the claim was filed. A very long lag (like 6–8 months) is a strong fraud signal:

```python
df["claim_lag_days"] = (df["CLAIM DATE"] - df["SERVICE DATE"]).dt.days

# Negative values mean the claim was filed before the service date — clearly wrong
df["claim_lag_days"] = df["claim_lag_days"].clip(lower=0)

# Fill missing with median
df["claim_lag_days"] = df["claim_lag_days"].fillna(df["claim_lag_days"].median())
```

### Step 2.3 — Feature: ICD code specificity score

Vague ICD codes like Z51.9 ("medical care, unspecified") are commonly used to disguise the true nature of a claim. Assign a vagueness score:

```python
# These are known catch-all codes used in the dataset
VAGUE_ICD_CODES = [
    "Z51.9",   # Medical care, unspecified
    "Z00.0",   # General medical examination
    "Z76.9",   # Person encountering health services in unspecified circumstances
    "Z71.9",   # Person encountering health services for unspecified counselling
]

df["icd_is_vague"] = df["ICD CODE"].isin(VAGUE_ICD_CODES).astype(int)
```

You can expand this list as you explore the full dataset — look for ICD codes with descriptions containing "unspecified" or "not elsewhere classified."

### Step 2.4 — Feature: Provider average inflation ratio

A hospital that consistently bills more than it gets paid is a pattern, even if any single claim looks normal. Calculate the average inflation ratio per provider across all their claims:

```python
provider_avg_inflation = (
    df.groupby("PROVIDER NAME")["invoice_inflation_ratio"]
    .mean()
    .rename("provider_avg_inflation")
)
df = df.merge(provider_avg_inflation, on="PROVIDER NAME", how="left")
```

### Step 2.5 — Feature: Provider claim volume (rolling)

A provider submitting an unusually high number of claims in a short period is suspicious. Calculate the total number of claims per provider in the dataset as a proxy (in production you would use a rolling time window):

```python
provider_claim_count = (
    df.groupby("PROVIDER NAME")["CLAIM NO"]
    .count()
    .rename("provider_claim_count")
)
df = df.merge(provider_claim_count, on="PROVIDER NAME", how="left")
```

### Step 2.6 — Feature: Member claim frequency

A member who submits (or has submitted on their behalf) an unusually high number of claims may indicate a ghost beneficiary or identity fraud:

```python
member_claim_count = (
    df.groupby("MEMBER_HASH")["CLAIM NO"]
    .count()
    .rename("member_claim_count")
)
df = df.merge(member_claim_count, on="MEMBER_HASH", how="left")
```

### Step 2.7 — Feature: Amount vs benefit code benchmark

Calculate the median invoice amount per `BENEFIT CODE`. Then measure how many times above that median each individual claim is:

```python
benefit_median = (
    df.groupby("BENEFIT CODE")["INVOICE AMOUNT"]
    .median()
    .rename("benefit_median_amount")
)
df = df.merge(benefit_median, on="BENEFIT CODE", how="left")

df["amount_vs_benchmark"] = df["INVOICE AMOUNT"] / df["benefit_median_amount"]
df["amount_vs_benchmark"] = df["amount_vs_benchmark"].clip(upper=10).fillna(1.0)
```

### Step 2.8 — Define the proxy fraud label for evaluation

You do not have a "fraud = yes/no" column. But you have a meaningful proxy: claims where the insurer settled significantly less than the invoice. This means the insurer already found something wrong:

```python
# A claim is "suspicious" if settled more than 20% below the invoice
df["proxy_fraud_label"] = (
    (df["SETTLED AMOUNT"] < df["INVOICE AMOUNT"] * 0.80)
).astype(int)

print(f"Suspicious claims: {df['proxy_fraud_label'].sum()}")
print(f"Total claims: {len(df)}")
print(f"Suspicious rate: {df['proxy_fraud_label'].mean():.1%}")
```

Write down the suspicious rate. You will use it to set the `contamination` parameter of the Isolation Forest in the next phase.

### Step 2.9 — Save the feature-engineered dataset

```python
FEATURE_COLUMNS = [
    "invoice_inflation_ratio",
    "claim_lag_days",
    "icd_is_vague",
    "provider_avg_inflation",
    "provider_claim_count",
    "member_claim_count",
    "amount_vs_benchmark",
    "proxy_fraud_label",   # only used for evaluation, NOT fed to the model
]

df[FEATURE_COLUMNS].to_csv("data/claims_features.csv", index=False)
```

---

## Phase 3 — Training the Isolation Forest Model

### Step 3.1 — Split the data into training and test sets

Even though Isolation Forest is unsupervised (it does not use the fraud label during training), you hold out a test set to evaluate performance after training:

```python
from sklearn.model_selection import train_test_split

X = df[FEATURE_COLUMNS].drop(columns=["proxy_fraud_label"])
y = df["proxy_fraud_label"]  # only used for post-hoc evaluation

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 3.2 — Scale the features

Isolation Forest works on distances between data points. Features with very different scales (e.g. `claim_lag_days` can be 0–365, while `icd_is_vague` is 0 or 1) need to be brought to a comparable range:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use the SAME scaler, not a new fit
```

Save the scaler — you will need it at inference time to scale new claims the same way:

```python
import joblib
joblib.dump(scaler, "models/fraud_scaler.joblib")
```

### Step 3.3 — Train the Isolation Forest

Set `contamination` to approximately the suspicious rate you measured in Step 2.8:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=200,      # number of trees — more is more accurate but slower
    contamination=0.08,    # replace with the suspicious rate from Step 2.8
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

model.fit(X_train_scaled)
```

### Step 3.4 — Generate predictions and scores on the test set

```python
import numpy as np

# decision_function returns a score — more negative = more anomalous
test_scores = model.decision_function(X_test_scaled)

# predict returns -1 (anomaly) or 1 (normal)
test_predictions = model.predict(X_test_scaled)

# Convert to 0/1 to compare with proxy label (1 = anomaly, 0 = normal)
test_predictions_binary = np.where(test_predictions == -1, 1, 0)
```

### Step 3.5 — Evaluate against the proxy fraud label

This is the performance report required by the hackathon:

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

print("=== Model Performance Report ===")
print(classification_report(y_test, test_predictions_binary,
                             target_names=["Normal", "Suspicious"]))

cm = confusion_matrix(y_test, test_predictions_binary)
print("Confusion Matrix:")
print(f"  True Negatives (correctly cleared): {cm[0][0]}")
print(f"  False Positives (wrongly flagged):  {cm[0][1]}")
print(f"  False Negatives (missed fraud):     {cm[1][0]}")
print(f"  True Positives (correctly caught):  {cm[1][1]}")
```

Save these numbers. They go into the model card section of your README. If precision or recall is below 0.5, go back to Phase 2 and add more features or remove noisy ones.

### Step 3.6 — Plot and save the feature importance

Isolation Forest does not natively provide feature importances, but you can approximate them by measuring the average anomaly score drop when you shuffle each feature (permutation importance):

```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

result = permutation_importance(
    model, X_test_scaled, test_predictions_binary,
    n_repeats=10, random_state=42
)

sorted_idx = result.importances_mean.argsort()
plt.barh(X_train.columns[sorted_idx], result.importances_mean[sorted_idx])
plt.title("Feature Importances (Permutation)")
plt.savefig("plots/feature_importance.png")
```

This plot goes in your README and makes the "how the model output is explained to end users" section concrete.

### Step 3.7 — Save the trained model

```python
joblib.dump(model, "models/fraud_model.joblib")
```

Create a `models/` directory in your project. This directory will be included in the Django module so the model file is available at inference time.

---

## Phase 4 — Creating the Django Module

This is where all the data science work gets wrapped into a proper openIMIS module that lives inside the platform.

### Step 4.1 — Create the module directory structure

Create a new directory (this will eventually become its own Git repository, but start locally):

```
openimis-be-fraud-detect/
├── openimis.json
├── setup.py
├── models/
│   └── fraud_model.joblib       ← copy from Phase 3
│   └── fraud_scaler.joblib      ← copy from Phase 3
├── data/
│   └── claims_features.csv      ← anonymised feature data
├── fraud_detect/
│   ├── __init__.py
│   ├── apps.py
│   ├── models.py
│   ├── rules.py
│   ├── engine.py
│   ├── signals.py
│   ├── schema.py
│   ├── views.py
│   ├── urls.py
│   ├── serializers.py
│   └── tests/
│       ├── __init__.py
│       ├── test_rules.py
│       └── test_model.py
└── migrations/
    └── __init__.py
```

### Step 4.2 — Write `openimis.json`

This file registers your module with the openIMIS platform. Without it, your module is invisible to the system:

```json
{
  "modules": [
    {
      "name": "fraud_detect",
      "pip": "openimis-be-fraud-detect",
      "url": "fraud_detect.urls"
    }
  ]
}
```

### Step 4.3 — Write `apps.py`

The AppConfig tells Django about your app and is where you connect signals:

```python
from django.apps import AppConfig

class FraudDetectConfig(AppConfig):
    name = "fraud_detect"
    verbose_name = "Fraud Detection"

    def ready(self):
        # This import triggers the signal connections defined in signals.py
        import fraud_detect.signals  # noqa: F401
```

### Step 4.4 — Write `models.py`

You need three database tables:

**1. FraudFlag** — stores the result of evaluating a claim (one row per claim):

```python
from django.db import models

class FraudFlag(models.Model):
    claim_id = models.IntegerField(unique=True, db_index=True)
    # Rules engine output
    is_rule_flagged = models.BooleanField(default=False)
    rule_flag_reasons = models.JSONField(default=list)
    # ML model output
    anomaly_score = models.FloatField(null=True)
    is_ml_anomaly = models.BooleanField(default=False)
    # Combined assessment
    overall_risk_level = models.CharField(
        max_length=10,
        choices=[("LOW", "Low"), ("MEDIUM", "Medium"), ("HIGH", "High")],
        default="LOW"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "tbl_FraudFlag"
```

**2. ReviewerOverride** — records when a human reviewer disagrees with the model:

```python
class ReviewerOverride(models.Model):
    claim_id = models.IntegerField(db_index=True)
    fraud_flag = models.ForeignKey(FraudFlag, on_delete=models.CASCADE)
    original_risk_level = models.CharField(max_length=10)
    reviewer_decision = models.CharField(
        max_length=20,
        choices=[("APPROVE", "Approve"), ("REJECT", "Reject"), ("ESCALATE", "Escalate")]
    )
    reviewer_id = models.IntegerField()
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "tbl_ReviewerOverride"
```

**3. ModelVersion** — tracks which version of the ML model is currently active:

```python
class ModelVersion(models.Model):
    version = models.CharField(max_length=50)
    model_file_path = models.CharField(max_length=500)
    scaler_file_path = models.CharField(max_length=500)
    precision_score = models.FloatField(null=True)
    recall_score = models.FloatField(null=True)
    f1_score = models.FloatField(null=True)
    training_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)

    class Meta:
        db_table = "tbl_FraudModelVersion"
```

### Step 4.5 — Create and run migrations

```bash
docker compose exec backend python manage.py makemigrations fraud_detect
docker compose exec backend python manage.py migrate fraud_detect
```

Confirm the three new tables appear in the database:

```bash
docker compose exec db psql -U postgres -d openimis -c "\dt *Fraud*"
```

### Step 4.6 — Write `rules.py`

This file contains the rules list and the function that evaluates a claim against them. The rules are defined as plain Python dictionaries — no special framework needed:

```python
from datetime import date

# -----------------------------------------------------------------------
# RULES CONFIGURATION
# An administrator can add, remove, or change thresholds here without
# touching any other code. Each rule has:
#   - name: human-readable label shown to the reviewer
#   - check: a function that receives a claim dict and returns True if
#             the rule is violated (i.e. True = FLAG this claim)
# -----------------------------------------------------------------------

def _invoice_inflation_ratio(claim):
    """Returns the ratio of invoice to settled amount, or 1.0 if unavailable."""
    settled = claim.get("approved_amount") or claim.get("claimed_amount")
    if not settled or settled == 0:
        return 1.0
    return claim.get("claimed_amount", 0) / settled

def _claim_lag_days(claim):
    """Returns days between service date and claim submission date."""
    service_date = claim.get("date_from")
    claim_date = claim.get("date_claimed")
    if not service_date or not claim_date:
        return 0
    return (claim_date - service_date).days

RULES = [
    {
        "name": "Claim lag exceeds 90 days",
        "description": (
            "The claim was filed more than 90 days after the service was delivered. "
            "This is a strong indicator of backdated or fabricated claims."
        ),
        "check": lambda claim: _claim_lag_days(claim) > 90,
    },
    {
        "name": "Invoice inflation above 3x",
        "description": (
            "The invoiced amount is more than 3 times the amount that was approved "
            "for payment. This suggests deliberate overbilling."
        ),
        "check": lambda claim: _invoice_inflation_ratio(claim) > 3.0,
    },
    {
        "name": "Vague ICD code used",
        "description": (
            "The claim uses a non-specific ICD code (such as Z51.9 — 'medical care, "
            "unspecified') which can be used to disguise the true nature of the visit."
        ),
        "check": lambda claim: claim.get("icd_code") in [
            "Z51.9", "Z00.0", "Z76.9", "Z71.9", "Z53.9"
        ],
    },
    {
        "name": "Claim filed after audit date",
        "description": (
            "The claim submission date is after the audit date, which is logically "
            "impossible and suggests record tampering."
        ),
        "check": lambda claim: (
            claim.get("date_claimed") and
            claim.get("audit_date") and
            claim.get("date_claimed") > claim.get("audit_date")
        ),
    },
    {
        "name": "High-value claim with vague diagnosis",
        "description": (
            "The claim amount is above 20,000 KES but the diagnosis code is "
            "non-specific. High-value claims require precise justification."
        ),
        "check": lambda claim: (
            claim.get("claimed_amount", 0) > 20000 and
            claim.get("icd_code") in ["Z51.9", "Z00.0", "Z76.9"]
        ),
    },
]


def evaluate_rules(claim_dict):
    """
    Runs a claim dictionary through all rules.
    Returns a dict with:
      - is_flagged: True if any rule fired
      - fired_rules: list of rule names and descriptions that fired
    """
    fired = []
    for rule in RULES:
        try:
            if rule["check"](claim_dict):
                fired.append({
                    "name": rule["name"],
                    "description": rule["description"],
                })
        except Exception:
            # Never let a broken rule crash the whole evaluation
            pass

    return {
        "is_flagged": len(fired) > 0,
        "fired_rules": fired,
    }
```

### Step 4.7 — Write `engine.py`

This file wraps both the rules engine and the ML model into a single scoring function:

```python
import os
import numpy as np
import joblib

_MODEL = None
_SCALER = None

def _load_models():
    """Lazy-loads the ML model and scaler on first use."""
    global _MODEL, _SCALER
    if _MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.joblib")
        scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_scaler.joblib")
        _MODEL = joblib.load(model_path)
        _SCALER = joblib.load(scaler_path)


def score_claim_ml(claim_dict):
    """
    Extracts features from a claim dict and returns:
      - anomaly_score: float (more negative = more suspicious)
      - is_anomaly: bool
    """
    _load_models()

    # Extract the same features that were used during training (Phase 2)
    settled = claim_dict.get("approved_amount") or claim_dict.get("claimed_amount") or 1
    claimed = claim_dict.get("claimed_amount", 0)
    inflation_ratio = min(claimed / settled if settled > 0 else 1.0, 10.0)

    service_date = claim_dict.get("date_from")
    claim_date = claim_dict.get("date_claimed")
    lag_days = max((claim_date - service_date).days, 0) if service_date and claim_date else 0

    icd_is_vague = 1 if claim_dict.get("icd_code") in [
        "Z51.9", "Z00.0", "Z76.9", "Z71.9", "Z53.9"
    ] else 0

    # For provider-level features, default to neutral values if unavailable
    # (in production these would be computed from the DB)
    provider_avg_inflation = claim_dict.get("provider_avg_inflation", 1.0)
    provider_claim_count = claim_dict.get("provider_claim_count", 1)
    member_claim_count = claim_dict.get("member_claim_count", 1)
    amount_vs_benchmark = claim_dict.get("amount_vs_benchmark", 1.0)

    features = np.array([[
        inflation_ratio,
        lag_days,
        icd_is_vague,
        provider_avg_inflation,
        provider_claim_count,
        member_claim_count,
        amount_vs_benchmark,
    ]])

    features_scaled = _SCALER.transform(features)
    score = float(_MODEL.decision_function(features_scaled)[0])
    prediction = _MODEL.predict(features_scaled)[0]  # -1 = anomaly, 1 = normal

    return {
        "anomaly_score": score,
        "is_anomaly": prediction == -1,
    }


def compute_risk_level(rules_result, ml_result):
    """
    Combines the rules engine output and ML output into a single risk level.
    """
    rule_flagged = rules_result["is_flagged"]
    ml_anomaly = ml_result["is_anomaly"]
    ml_score = ml_result["anomaly_score"]

    if rule_flagged and ml_anomaly:
        return "HIGH"
    elif rule_flagged or ml_anomaly:
        return "MEDIUM"
    elif ml_score < -0.1:
        return "MEDIUM"
    else:
        return "LOW"
```

### Step 4.8 — Write `signals.py`

This file connects the scoring engine to Django's `post_save` signal on the Claim model. Every time a claim is saved, the engine automatically scores it:

```python
from django.db.models.signals import post_save
from django.dispatch import receiver

# The Claim model lives in the core openIMIS claims module
# The exact import path depends on the version of openIMIS you are using
# Check the claims module's models.py for the correct import
from claim.models import Claim

from .engine import score_claim_ml, compute_risk_level
from .rules import evaluate_rules
from .models import FraudFlag


@receiver(post_save, sender=Claim)
def evaluate_claim_on_save(sender, instance, created, **kwargs):
    """
    Automatically scores a claim for fraud risk whenever it is saved.
    This fires for both new claims (created=True) and updates (created=False).
    """
    # Build a claim dict from the Django model instance
    # Map openIMIS field names to the dict keys expected by the engine
    claim_dict = {
        "claimed_amount": float(instance.claimed or 0),
        "approved_amount": float(instance.approved or 0),
        "date_from": instance.date_from,
        "date_claimed": instance.date_claimed,
        "icd_code": instance.icd.code if instance.icd else None,
        "icd_code_1": instance.icd1.code if instance.icd1 else None,
    }

    # Run both layers
    rules_result = evaluate_rules(claim_dict)
    ml_result = score_claim_ml(claim_dict)
    risk_level = compute_risk_level(rules_result, ml_result)

    # Write or update the FraudFlag record for this claim
    FraudFlag.objects.update_or_create(
        claim_id=instance.id,
        defaults={
            "is_rule_flagged": rules_result["is_flagged"],
            "rule_flag_reasons": rules_result["fired_rules"],
            "anomaly_score": ml_result["anomaly_score"],
            "is_ml_anomaly": ml_result["is_anomaly"],
            "overall_risk_level": risk_level,
        }
    )
```

### Step 4.9 — Write `serializers.py`

Serializers convert your Django model instances into JSON for the REST API:

```python
from rest_framework import serializers
from .models import FraudFlag, ReviewerOverride


class FraudFlagSerializer(serializers.ModelSerializer):
    class Meta:
        model = FraudFlag
        fields = [
            "id", "claim_id", "is_rule_flagged", "rule_flag_reasons",
            "anomaly_score", "is_ml_anomaly", "overall_risk_level",
            "created_at", "updated_at"
        ]


class ReviewerOverrideSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewerOverride
        fields = [
            "id", "claim_id", "original_risk_level", "reviewer_decision",
            "reviewer_id", "notes", "created_at"
        ]
```

### Step 4.10 — Write `views.py`

These are the REST API endpoints your frontend will call:

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404

from .models import FraudFlag, ReviewerOverride
from .serializers import FraudFlagSerializer, ReviewerOverrideSerializer
from .engine import score_claim_ml, compute_risk_level
from .rules import evaluate_rules


class ClaimFraudFlagView(APIView):
    """
    GET /api/fraud/flags/{claim_id}/
    Returns the fraud flag assessment for a specific claim.
    """
    def get(self, request, claim_id):
        flag = get_object_or_404(FraudFlag, claim_id=claim_id)
        serializer = FraudFlagSerializer(flag)
        return Response(serializer.data)


class FraudFlagListView(APIView):
    """
    GET /api/fraud/flags/?risk_level=HIGH
    Returns all fraud flags, optionally filtered by risk level.
    """
    def get(self, request):
        risk_level = request.query_params.get("risk_level")
        queryset = FraudFlag.objects.all()
        if risk_level:
            queryset = queryset.filter(overall_risk_level=risk_level)
        serializer = FraudFlagSerializer(queryset, many=True)
        return Response(serializer.data)


class ReviewerOverrideView(APIView):
    """
    POST /api/fraud/override/
    Records a reviewer's decision to override the model's assessment.
    Body: { claim_id, reviewer_decision, reviewer_id, notes }
    """
    def post(self, request):
        flag = get_object_or_404(FraudFlag, claim_id=request.data.get("claim_id"))
        override = ReviewerOverride.objects.create(
            claim_id=request.data["claim_id"],
            fraud_flag=flag,
            original_risk_level=flag.overall_risk_level,
            reviewer_decision=request.data["reviewer_decision"],
            reviewer_id=request.data["reviewer_id"],
            notes=request.data.get("notes", ""),
        )
        serializer = ReviewerOverrideSerializer(override)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
```

### Step 4.11 — Write `urls.py`

```python
from django.urls import path
from .views import ClaimFraudFlagView, FraudFlagListView, ReviewerOverrideView

urlpatterns = [
    path("fraud/flags/", FraudFlagListView.as_view(), name="fraud-flag-list"),
    path("fraud/flags/<int:claim_id>/", ClaimFraudFlagView.as_view(), name="fraud-flag-detail"),
    path("fraud/override/", ReviewerOverrideView.as_view(), name="reviewer-override"),
]
```

### Step 4.12 — Write `schema.py` (GraphQL)

OpenIMIS uses GraphQL for frontend-to-backend communication. You need at least one query so the frontend React module can fetch fraud flags:

```python
import graphene
from graphene_django import DjangoObjectType
from .models import FraudFlag, ReviewerOverride


class FraudFlagType(DjangoObjectType):
    class Meta:
        model = FraudFlag
        fields = "__all__"


class ReviewerOverrideType(DjangoObjectType):
    class Meta:
        model = ReviewerOverride
        fields = "__all__"


class Query(graphene.ObjectType):

    fraud_flag = graphene.Field(FraudFlagType, claim_id=graphene.Int(required=True))
    fraud_flags = graphene.List(
        FraudFlagType,
        risk_level=graphene.String(),
        first=graphene.Int(),
        skip=graphene.Int(),
    )

    def resolve_fraud_flag(self, info, claim_id):
        return FraudFlag.objects.filter(claim_id=claim_id).first()

    def resolve_fraud_flags(self, info, risk_level=None, first=None, skip=None):
        qs = FraudFlag.objects.all().order_by("-created_at")
        if risk_level:
            qs = qs.filter(overall_risk_level=risk_level)
        if skip:
            qs = qs[skip:]
        if first:
            qs = qs[:first]
        return qs


class Mutation(graphene.ObjectType):
    pass  # overrides are handled via REST; add GraphQL mutations here if time permits
```

---

## Phase 5 — FHIR ClaimResponse Integration

The hackathon specifically requires that FHIR R4 `ClaimResponse` resources are correctly populated. This is worth 25 points on the openIMIS integration criterion and is something most teams will skip or do superficially.

### Step 5.1 — Understand the existing FHIR module

The openIMIS FHIR R4 module (`openimis-be-api_fhir_r4_py`) already handles converting openIMIS claims to FHIR `Claim` resources. Read its converter classes to understand the pattern:

```bash
find openimis-be_py -name "*claim*converter*" -o -name "*ClaimResponse*"
```

### Step 5.2 — Understand the FHIR ClaimResponse structure

A FHIR `ClaimResponse` resource represents the outcome of adjudicating a claim. The relevant fields for your module are:

- `outcome` — overall adjudication result: `complete`, `partial`, `error`, `queued`
- `adjudication` — array of adjudication items, each with a category code and amount
- `extension` — where you add custom fields not in the base FHIR spec (your fraud score goes here)

### Step 5.3 — Add the fraud score as a FHIR extension

FHIR extensions allow you to add custom data to standard resources without breaking the spec. Create a file `fhir_extensions.py`:

```python
FRAUD_SCORE_EXTENSION_URL = "https://openimis.org/fhir/StructureDefinition/fraud-score"
FRAUD_RISK_LEVEL_EXTENSION_URL = "https://openimis.org/fhir/StructureDefinition/fraud-risk-level"
FRAUD_RULES_FIRED_EXTENSION_URL = "https://openimis.org/fhir/StructureDefinition/fraud-rules-fired"


def build_fraud_extensions(fraud_flag):
    """
    Builds a list of FHIR extension objects from a FraudFlag instance.
    These are added to the ClaimResponse resource.
    """
    if not fraud_flag:
        return []

    return [
        {
            "url": FRAUD_SCORE_EXTENSION_URL,
            "valueDecimal": round(fraud_flag.anomaly_score, 4),
        },
        {
            "url": FRAUD_RISK_LEVEL_EXTENSION_URL,
            "valueString": fraud_flag.overall_risk_level,
        },
        {
            "url": FRAUD_RULES_FIRED_EXTENSION_URL,
            "valueString": "; ".join(
                r["name"] for r in fraud_flag.rule_flag_reasons
            ) if fraud_flag.rule_flag_reasons else "None",
        },
    ]
```

### Step 5.4 — Hook into the FHIR ClaimResponse converter

Find the ClaimResponse converter in the FHIR module and add your extensions to its output. The exact method to override depends on the openIMIS FHIR module version — look for a method called `build_fhir_claim_response` or `to_fhir_obj`. Add:

```python
from fraud_detect.models import FraudFlag
from fraud_detect.fhir_extensions import build_fraud_extensions

# Inside the ClaimResponse converter, after building the base resource:
flag = FraudFlag.objects.filter(claim_id=claim.id).first()
fhir_claim_response["extension"] = build_fraud_extensions(flag)
```

---

## Phase 6 — The Feedback Loop (Retraining Pipeline)

This is the bonus deliverable that the hackathon handbook specifically calls out. It makes your model adaptive.

### Step 6.1 — Write a management command to retrain the model

Django management commands are scripts you run with `python manage.py <command_name>`. Create `management/commands/retrain_fraud_model.py`:

```python
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

from fraud_detect.models import ReviewerOverride, ModelVersion

FEATURE_COLUMNS = [
    "invoice_inflation_ratio", "claim_lag_days", "icd_is_vague",
    "provider_avg_inflation", "provider_claim_count",
    "member_claim_count", "amount_vs_benchmark",
]


class Command(BaseCommand):
    help = "Retrains the fraud detection model using reviewer overrides as feedback"

    def handle(self, *args, **options):
        self.stdout.write("Loading base training data...")
        base_data_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "claims_features.csv"
        )
        df = pd.read_csv(base_data_path)

        self.stdout.write("Loading reviewer overrides...")
        # Get all overrides where a reviewer approved a claim the model flagged
        # These are "false positives" — the model was too aggressive
        overrides = ReviewerOverride.objects.filter(
            reviewer_decision="APPROVE",
            original_risk_level__in=["HIGH", "MEDIUM"]
        ).values(
            "claim_id", "fraud_flag__anomaly_score",
            "fraud_flag__is_rule_flagged"
        )

        self.stdout.write(f"Found {overrides.count()} override records")

        self.stdout.write("Retraining model...")
        X = df[FEATURE_COLUMNS].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=200,
            contamination=0.08,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled)

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "fraud_model_new.joblib"
        )
        scaler_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "fraud_scaler_new.joblib"
        )
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Record the new model version in the database
        ModelVersion.objects.filter(is_active=True).update(is_active=False)
        ModelVersion.objects.create(
            version="retrained",
            model_file_path=model_path,
            scaler_file_path=scaler_path,
            is_active=True,
        )

        self.stdout.write(self.style.SUCCESS("Model retrained and saved successfully"))
```

---

## Phase 7 — Unit Tests

Tests are worth points in the Technical criterion and also protect you from breaking your own code as you develop. Write them as you go, not at the end.

### Step 7.1 — Test the rules engine

Create `tests/test_rules.py`:

```python
from django.test import TestCase
from datetime import date
from fraud_detect.rules import evaluate_rules


class RulesEngineTestCase(TestCase):

    def _make_claim(self, **kwargs):
        """Creates a clean claim dict with sensible defaults, overriding with kwargs."""
        defaults = {
            "claimed_amount": 5000,
            "approved_amount": 5000,
            "date_from": date(2025, 3, 1),
            "date_claimed": date(2025, 3, 3),
            "icd_code": "J06.9",  # non-vague code
        }
        defaults.update(kwargs)
        return defaults

    def test_clean_claim_is_not_flagged(self):
        claim = self._make_claim()
        result = evaluate_rules(claim)
        self.assertFalse(result["is_flagged"])
        self.assertEqual(len(result["fired_rules"]), 0)

    def test_claim_lag_rule_fires_at_91_days(self):
        claim = self._make_claim(
            date_from=date(2025, 1, 1),
            date_claimed=date(2025, 4, 2)  # 91 days later
        )
        result = evaluate_rules(claim)
        self.assertTrue(result["is_flagged"])
        rule_names = [r["name"] for r in result["fired_rules"]]
        self.assertIn("Claim lag exceeds 90 days", rule_names)

    def test_claim_lag_rule_does_not_fire_at_89_days(self):
        claim = self._make_claim(
            date_from=date(2025, 1, 1),
            date_claimed=date(2025, 3, 31)  # 89 days later
        )
        result = evaluate_rules(claim)
        self.assertFalse(result["is_flagged"])

    def test_vague_icd_rule_fires_for_z519(self):
        claim = self._make_claim(icd_code="Z51.9")
        result = evaluate_rules(claim)
        self.assertTrue(result["is_flagged"])
        rule_names = [r["name"] for r in result["fired_rules"]]
        self.assertIn("Vague ICD code used", rule_names)

    def test_inflation_rule_fires_when_invoice_is_3x_settled(self):
        claim = self._make_claim(claimed_amount=9000, approved_amount=3000)
        result = evaluate_rules(claim)
        self.assertTrue(result["is_flagged"])

    def test_multiple_rules_can_fire_on_same_claim(self):
        claim = self._make_claim(
            claimed_amount=30000,
            approved_amount=5000,
            icd_code="Z51.9",
            date_from=date(2025, 1, 1),
            date_claimed=date(2025, 6, 1)
        )
        result = evaluate_rules(claim)
        self.assertTrue(result["is_flagged"])
        self.assertGreater(len(result["fired_rules"]), 1)
```

### Step 7.2 — Test the ML scoring engine

Create `tests/test_model.py`:

```python
from django.test import TestCase
from datetime import date
from fraud_detect.engine import score_claim_ml, compute_risk_level


class MLEngineTestCase(TestCase):

    def test_score_claim_returns_required_keys(self):
        claim = {
            "claimed_amount": 5000,
            "approved_amount": 5000,
            "date_from": date(2025, 3, 1),
            "date_claimed": date(2025, 3, 3),
            "icd_code": "J06.9",
        }
        result = score_claim_ml(claim)
        self.assertIn("anomaly_score", result)
        self.assertIn("is_anomaly", result)
        self.assertIsInstance(result["anomaly_score"], float)
        self.assertIsInstance(result["is_anomaly"], bool)

    def test_risk_level_is_high_when_both_layers_flag(self):
        rules_result = {"is_flagged": True, "fired_rules": [{"name": "test"}]}
        ml_result = {"is_anomaly": True, "anomaly_score": -0.5}
        level = compute_risk_level(rules_result, ml_result)
        self.assertEqual(level, "HIGH")

    def test_risk_level_is_low_for_clean_claim(self):
        rules_result = {"is_flagged": False, "fired_rules": []}
        ml_result = {"is_anomaly": False, "anomaly_score": 0.3}
        level = compute_risk_level(rules_result, ml_result)
        self.assertEqual(level, "LOW")
```

Run all tests with:

```bash
docker compose exec backend python manage.py test fraud_detect
```

All tests must pass before moving to Phase 8.

---

## Phase 8 — Frontend Integration

The minimum viable submission requires a React UI component that surfaces the fraud score within the existing claims list view. This does not require building a new React module from scratch — it requires adding a column and a badge to an existing page.

### Step 8.1 — Locate the existing claims list component

In `openimis-fe_js`, find the claims list component. It is typically in the `openimis-fe-claim_js` module. Look for a file named something like `ClaimSearcher.js` or `ClaimList.js`.

### Step 8.2 — Add a fraud risk badge column

In the claims table, add a new column that calls your REST endpoint and displays the risk level as a coloured badge:

```jsx
// A small component that fetches and displays the fraud risk badge for one claim
import React, { useEffect, useState } from "react";

const RISK_COLOURS = {
  HIGH: { background: "#d32f2f", color: "white" },
  MEDIUM: { background: "#f57c00", color: "white" },
  LOW: { background: "#388e3c", color: "white" },
};

function FraudRiskBadge({ claimId }) {
  const [flag, setFlag] = useState(null);

  useEffect(() => {
    fetch(`/api/fraud/flags/${claimId}/`)
      .then((res) => res.ok ? res.json() : null)
      .then((data) => setFlag(data))
      .catch(() => setFlag(null));
  }, [claimId]);

  if (!flag) return <span style={{ color: "#999" }}>—</span>;

  const style = RISK_COLOURS[flag.overall_risk_level] || {};

  return (
    <span
      title={
        flag.rule_flag_reasons.length > 0
          ? flag.rule_flag_reasons.map((r) => r.name).join("; ")
          : `Anomaly score: ${flag.anomaly_score.toFixed(3)}`
      }
      style={{
        ...style,
        padding: "2px 8px",
        borderRadius: "12px",
        fontSize: "0.75rem",
        fontWeight: "bold",
        cursor: "help",
      }}
    >
      {flag.overall_risk_level}
    </span>
  );
}

export default FraudRiskBadge;
```

The `title` attribute makes the badge show the reason when hovered — this is the "surfaces scoring output to a claims reviewer" requirement without needing a full modal.

### Step 8.3 — Add the badge column to the claims table

In the claims table component, import `FraudRiskBadge` and add it as a new column:

```jsx
// Inside the column definitions array of the claims table
{
  id: "fraud_risk",
  label: "Fraud Risk",
  render: (claim) => <FraudRiskBadge claimId={claim.id} />,
  sortable: false,
}
```

---

## Phase 9 — Generate the Performance Report

This is a required deliverable for Track 3.

### Step 9.1 — Run the full evaluation script

Using the test set held out in Phase 3 (Step 3.1), generate the final numbers:

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assumes X_test_scaled and y_test from Phase 3 are still in memory
test_predictions = model.predict(X_test_scaled)
test_predictions_binary = (test_predictions == -1).astype(int)

print(classification_report(y_test, test_predictions_binary,
                             target_names=["Normal", "Suspicious"]))

# Plot confusion matrix
cm = confusion_matrix(y_test, test_predictions_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Predicted Normal", "Predicted Suspicious"],
            yticklabels=["Actual Normal", "Actual Suspicious"])
plt.title("Fraud Detection Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
```

Save the printed classification report as plain text. Both the confusion matrix image and the text report go into the README under a "Model Performance" section.

---

## Phase 10 — Documentation

Documentation is worth 10 points but is also the thing judges use to understand what you built before the demo. A poor README creates doubt before you even start speaking.

### Step 10.1 — Write the README

The README must contain these sections in this order:

1. **Problem Statement** — one paragraph explaining why claim fraud matters in Kenya / LMICs and what the current manual process costs
2. **Solution Overview** — the two-layer approach (rules + ML), with the architecture diagram
3. **Architecture Diagram** — a text diagram showing the data flow from claim submission to fraud flag
4. **Features Used by the Model** — a table of all 7 features with descriptions (copy from Phase 2 with results)
5. **Installation Steps** — exact commands to add the module to the Docker stack
6. **Model Performance Report** — the precision, recall, F1 table and confusion matrix image from Phase 9
7. **Responsible AI Section** — cover:
   - What data was used for training (anonymised Kenyan insurer claims, date range, row count)
   - What biases were considered (the model was trained on one insurer's data and may not generalise to other providers)
   - How model output is explained to end users (the rules engine shows exact rule names; the anomaly score is shown with feature importance context)
   - Known failure modes (new fraud patterns not present in training data will not be caught until the model is retrained)
8. **Known Limitations**
9. **Link to Draft PR** on the official openIMIS GitHub repo

### Step 10.2 — Write a model card

A model card is a one-page document (can be a section of the README or a separate `MODEL_CARD.md`) that summarises the ML model for non-technical readers. It should cover:

- Model type: Isolation Forest (unsupervised anomaly detection)
- Training data: anonymised claims data, N rows, date range
- Features: list all 7
- Performance: precision, recall, F1 from Phase 9
- Intended use: flagging claims for human review — NOT for automatic rejection
- Out-of-scope use: should not be used to make final adjudication decisions without human review
- How to update: retrain using `python manage.py retrain_fraud_model`

### Step 10.3 — Write a Docker Compose override file

This is the bonus deliverable that lets anyone add your module to the stack in one command:

```yaml
# compose.fraud-detect.yml
services:
  backend:
    build:
      context: ./openimis-be-fraud-detect
      dockerfile: Dockerfile
    environment:
      - FRAUD_DETECT_ENABLED=true
    volumes:
      - ./openimis-be-fraud-detect/models:/app/fraud_detect/models
```

### Step 10.4 — Open a draft PR on the official openIMIS repository

This is **required** for the Documentation criterion. Go to `github.com/openimis/openimis-be_py`, fork the repo if not already done, and open a draft pull request with:

- Title: `[Hackathon] feat: Add AI-powered claims fraud detection module`
- Body: link to your fork, brief description, screenshot of the fraud badge in the claims list
- Mark it as a Draft PR — you are not asking for a merge, just demonstrating intent to contribute

---

## Phase 11 — Demo Preparation

### Step 11.1 — Seed the demo database with interesting test claims

Write a Django management command `seed_demo_claims.py` that creates a set of test claims specifically designed to show the system working:

- Claim A: clean lab test, 3,000 KES, filed same day → should be LOW risk, auto-cleared
- Claim B: consultant, Z51.9 ICD code, 15,000 KES, filed 120 days after service → should be HIGH risk, both layers fire
- Claim C: invoice 25,000 KES, settled 3,000 KES → should trigger inflation rule
- Claim D: normal spectacle frame, H52.1, filed 3 days after service → LOW risk

### Step 11.2 — Prepare the 8-minute demo script

Structure the demo in exactly this order:

1. **(1 min)** Open the claims list. Point out that 3 of the 4 seeded claims have coloured badges — without any human having reviewed them yet.
2. **(2 min)** Click on Claim B (HIGH risk). Show the flag details: which rules fired, the anomaly score, and what each means.
3. **(2 min)** Click "Reject" on Claim B. Show that the override is logged and will feed back into the next retraining cycle.
4. **(1 min)** Show Claim A (LOW risk) was auto-cleared — no human needed to touch it.
5. **(1 min)** Show the performance report numbers: "On real Kenyan insurer data, our model achieves X% precision and Y% recall."
6. **(1 min)** Show the architecture diagram and the FHIR ClaimResponse extension field in a raw API response.

### Step 11.3 — Commit, tag, and open the final PR

```bash
git add .
git commit -m "feat: complete Track 3 fraud detection module v1.0"
git tag v1.0-hackathon
git push origin main --tags
```

---

## Implementation Order Summary

| Phase | What you build | Why this order |
|-------|----------------|----------------|
| 0 | Environment + codebase reading | Must understand before building |
| 1 | Data cleaning + anonymisation | Must have clean data before engineering features |
| 2 | Feature engineering | Must have features before training the model |
| 3 | Model training + evaluation | Must have the model before building the API |
| 4 | Django module (models, signals, API) | Core integration — everything else hangs off this |
| 5 | FHIR ClaimResponse extensions | Builds on Phase 4's module structure |
| 6 | Feedback loop / retraining command | Builds on Phase 4's override model |
| 7 | Unit tests | Written alongside Phase 4, run to verify before proceeding |
| 8 | Frontend badge | Last — depends on the API from Phase 4 being stable |
| 9 | Performance report | Depends on Phase 3 model being final |
| 10 | Documentation | Written last when implementation is stable |
| 11 | Demo preparation | Final step — assumes everything works |
