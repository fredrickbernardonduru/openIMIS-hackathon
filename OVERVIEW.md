# AI-Powered Claims Anomaly Detector

## Problem Statement

Health insurance fraud is a systemic problem in Kenya and across low- and middle-income countries. When a patient visits a hospital, the hospital submits a claim to the insurer describing what services were provided and how much they cost. The insurer then reviews the claim and decides how much to pay. This process is almost entirely manual — a human reviewer reads each claim and makes a judgement call.

This creates two serious problems:

**1. It is slow.** With hundreds or thousands of claims arriving daily, reviewers cannot keep up. Claims sit in queues for days or weeks before being processed. Providers do not get paid on time, which erodes trust in the insurance scheme. Beneficiaries sometimes lose coverage while their claims are in limbo.

**2. It misses fraud.** A common form of fraud is hospital claim inflation — a hospital bills for services that were not delivered, upcodes a minor visit to a more expensive procedure, or submits the same claim multiple times. A reviewer looking at one claim in isolation cannot easily detect that the same hospital has been systematically overbilling for months. Another common pattern, visible in real Kenyan insurer data, is claims filed many months after the service date — a strong signal of backdated or fabricated records.

Real-world evidence from a Kenyan insurer dataset shows these patterns clearly: claims filed 6–8 months after the service date, invoices settled at a fraction of the billed amount (indicating the insurer already detected something wrong), and repeated use of vague ICD diagnostic codes such as Z51.9 ("medical care, unspecified") that provide no clinical justification for high-value services.

The openIMIS platform, used by 38.8 million beneficiaries across 14 countries, currently has no automated fraud detection layer. Every claim must be manually adjudicated.

## What We Are Building

A fraud detection module for openIMIS that operates as two complementary layers:

- **Layer 1 — Rules Engine**: A configurable set of rules derived directly from known fraud patterns in the dataset. Any claim that violates a rule is immediately flagged with a plain-language explanation of why. This catches the obvious cases deterministically and at zero computational cost.

- **Layer 2 — Isolation Forest ML Model**: A machine learning model trained on real claim data that scores every claim for statistical anomaly. It catches fraud patterns that no one thought to write a rule for — the subtle, evolving schemes that slip past the rulebook.

The two layers run automatically every time a claim is saved in openIMIS. A claims reviewer sees a colour-coded risk badge (LOW / MEDIUM / HIGH) on each claim in the list, with a hover tooltip explaining the reason. High-risk claims are prioritised for human review. Low-risk claims can be processed faster. When a reviewer overrides the model's decision, that action feeds back into the next model retraining cycle, making the system progressively smarter.

The result: faster claim processing, fewer missed fraud cases, and an auditable record of every automated decision.

---

## Technical Overview

The combined idea has two layers that work together. Think of them as two security guards standing at a gate. The first guard has a rulebook and checks everyone against known rules. The second guard uses instinct — they've seen thousands of people walk through and can sense when something feels "off" even if it doesn't break any written rule. Together they catch more problems than either would alone.

---

## Layer 1 — The Rules Engine (Idea #1)

### What problem does it solve?

Right now in openIMIS (and in real Kenyan insurers like Equity Afia), when a hospital submits a claim, a human reviewer has to read it and decide: approve it, partially approve it, or reject it. This is slow — it can take days or weeks. It is also inconsistent — two reviewers might make different decisions about the same claim.

The rules engine automates the obvious cases. It is a piece of Python code that reads each incoming claim and checks it against a list of rules you define. If the claim passes all rules cleanly, it is auto-approved. If it fails a rule, it is flagged for a human to look at. The human only ever sees the tricky ones — the clear ones are handled automatically.

### What does "configurable" mean?

It means the rules are not hardcoded. They live in a configuration file (a Python dictionary or a database table) that an administrator can edit without touching the code. For example:

```python
RULES = [
    {
        "name": "Claim lag too long",
        "description": "Flag claims filed more than 90 days after the service date",
        "field": "claim_to_service_lag_days",
        "operator": "greater_than",
        "threshold": 90,
        "action": "FLAG"
    },
    {
        "name": "Invoice inflation",
        "description": "Flag claims where the invoice is more than 3x the benchmark for that service",
        "field": "invoice_vs_benchmark_ratio",
        "operator": "greater_than",
        "threshold": 3.0,
        "action": "FLAG"
    },
    {
        "name": "Vague ICD code",
        "description": "Flag claims using Z51.9 or other catch-all codes for high-value services",
        "field": "icd_code",
        "operator": "in_list",
        "threshold": ["Z51.9", "Z00.0", "Z76.9"],
        "action": "FLAG"
    }
]
```

You can look at your dataset and directly translate what you observe into rules. You already found three rules in the sample data alone:
- Claims filed 8 months after the service date → FLAG
- Invoice 83% higher than what was settled → FLAG
- ICD code Z51.9 (vague "medical care, unspecified") used repeatedly → FLAG

### How does it plug into openIMIS technically?

OpenIMIS is built in Django (a Python web framework). Django has a feature called **signals** — these are like event listeners. You write a function and say "run this function every time a claim is saved to the database." So your rules engine lives inside one of these signals:

```
Hospital submits claim
        ↓
openIMIS saves the claim to the database
        ↓
Django fires a "post_save" signal (like a notification: "hey, a claim was just saved")
        ↓
Your rules engine function wakes up, receives the claim
        ↓
It loops through every rule in your RULES list
        ↓
If any rule triggers → sets a "fraud_flag" field on the claim to True
        ↓
Adds a "flag_reason" field explaining which rule fired
        ↓
The claims reviewer sees the flag in the UI and knows exactly why it was flagged
```

The reviewer never had to open the claim themselves — your code told them: "This claim was flagged because it was filed 247 days after the service date and uses a vague ICD code."

### What does the Django module skeleton look like?

An openIMIS module is just a Django app with a specific structure. At minimum yours would have:

```
openimis-be-fraud-detect/
├── openimis.json              ← registers your module with the platform
├── fraud_detect/
│   ├── models.py              ← FraudFlag model (stores flag results per claim)
│   ├── rules.py               ← the RULES list and the engine function
│   ├── signals.py             ← wires the engine to Django's post_save signal
│   ├── schema.py              ← GraphQL query so the frontend can read flags
│   ├── views.py               ← REST endpoint for the flags
│   └── tests.py               ← unit tests (feed a test claim, check the right rule fires)
```

### What are unit tests in this context?

A unit test is a small script that checks your code does what you think it does. For the rules engine, a test looks like:

```python
def test_lag_rule_fires():
    claim = FakeClaim(service_date="2025-03-03", claim_date="2025-11-25")
    result = run_rules_engine(claim)
    assert result.is_flagged == True
    assert "Claim lag too long" in result.flag_reasons
```

You create a fake claim with known properties, run it through your engine, and confirm the right flag came out. This is what gives you the "test coverage" that judges look for in the Technical criterion.

---

## Layer 2 — The Anomaly Scoring Model (Idea #3)

### What problem does it solve?

Rules catch what you already know is suspicious. But fraud is creative — fraudsters learn the rules and design claims that technically pass every check while still being wrong. The ML model catches the patterns you didn't think to write a rule for.

The specific algorithm is called **Isolation Forest**. Here is how it works in plain English:

Imagine you have thousands of claims plotted as points on a map. Each claim is a dot, and its position is determined by its properties: invoice amount, how long after service it was filed, how many claims this provider submitted this month, etc. Most claims cluster together in a big dense group — these are "normal" claims. Fraudulent claims tend to sit far from the group — they are outliers, isolated points.

Isolation Forest finds these isolated points. It does so by randomly drawing dividing lines across the map (like a game of divide-and-conquer). Normal claims in the dense group take many cuts to isolate — they have lots of neighbours. Anomalous claims get isolated with very few cuts because they are already far from everyone else. The fewer cuts needed, the higher the anomaly score.

### Why "unsupervised"?

A supervised ML model needs labelled training data: thousands of claims with a column saying "fraud = yes" or "fraud = no." You almost never have this because insurers rarely officially label claims as fraud — they just partially settle them or reject them.

Isolation Forest is **unsupervised** — it needs no labels. It just looks at the distribution of claim properties and identifies the ones that are weirdly different from the rest. This is perfect for your dataset because you don't have a "fraud" column, but you do have patterns (that 83% invoice reduction is an implicit signal that something was wrong).

### What features (columns) does the model use?

A "feature" is just a number you calculate from your data and feed to the model. From your dataset specifically:

| Feature name | How you calculate it | What it captures |
|---|---|---|
| `invoice_inflation_ratio` | `INVOICE AMOUNT ÷ SETTLED AMOUNT` | How much the hospital inflated vs what was paid |
| `claim_lag_days` | `CLAIM DATE − SERVICE DATE` in days | Suspiciously late filing |
| `icd_specificity_score` | 1 if ICD is a catch-all code, 0 if specific | Vague diagnosis on expensive service |
| `provider_avg_inflation` | Average `invoice_inflation_ratio` for this provider across all their claims | Consistently inflating provider |
| `member_claim_frequency` | Number of claims this member filed in the last 90 days | Unusually frequent claimant |
| `amount_vs_benefit_benchmark` | `INVOICE AMOUNT ÷ typical amount for this BENEFIT CODE` | Claiming far above the norm for this service type |

You extract these using **pandas** — a Python library for working with tabular data (like Excel, but in code). For example:

```python
import pandas as pd

df = pd.read_csv("claims.csv")

# Calculate invoice inflation ratio
df["invoice_inflation_ratio"] = df["INVOICE AMOUNT"] / df["SETTLED AMOUNT"]

# Calculate claim lag in days
df["CLAIM DATE"] = pd.to_datetime(df["CLAIM DATE"], dayfirst=True)
df["SERVICE DATE"] = pd.to_datetime(df["SERVICE DATE"], dayfirst=True)
df["claim_lag_days"] = (df["CLAIM DATE"] - df["SERVICE DATE"]).dt.days

# Flag vague ICD codes
VAGUE_CODES = ["Z51.9", "Z00.0", "Z76.9"]
df["icd_is_vague"] = df["ICD CODE"].isin(VAGUE_CODES).astype(int)
```

### Training and using the model

Training means showing the model all the claims so it learns what "normal" looks like:

```python
from sklearn.ensemble import IsolationForest
import joblib

features = ["invoice_inflation_ratio", "claim_lag_days", "icd_is_vague",
            "provider_avg_inflation", "member_claim_frequency"]

X = df[features].fillna(0)  # fill any missing values with 0

model = IsolationForest(contamination=0.05)  # assume ~5% of claims are anomalous
model.fit(X)  # learns what normal looks like

# Score every claim — more negative = more anomalous
df["anomaly_score"] = model.decision_function(X)
df["is_anomaly"] = model.predict(X)  # -1 = anomaly, 1 = normal

# Save the trained model to a file so it can be loaded later without retraining
joblib.dump(model, "fraud_model.joblib")
```

When a new claim arrives in openIMIS, you load the saved model, extract features from the new claim, and get its anomaly score in milliseconds — no retraining needed:

```python
model = joblib.load("fraud_model.joblib")
score = model.decision_function([new_claim_features])
```

### The feedback loop (bonus deliverable)

When a human reviewer looks at a flagged claim and decides "actually this is fine, approve it" — that override action is recorded. Over time you feed those overrides back into the training data and retrain the model. This means the model gets smarter the more it is used. This is what the hackathon handbook calls "when a reviewer overrides the model decision, that action re-enters the training pipeline."

---

## How the two layers work together in the demo

This is the 8-minute demo you will show the judges:

1. You open the openIMIS claims list page. A new claim appears from "Equity Afia" for a consultant visit costing 45,000 KES with ICD code Z51.9, filed 6 months after the service date.

2. Within a second, two things happen automatically:
   - The **rules engine** fires a flag: "FLAGGED — Vague ICD code (Z51.9) AND claim lag 187 days"
   - The **ML model** scores it: anomaly score -0.41 (high risk — shown as a red badge in the UI)

3. The claim reviewer sees both signals side by side in the UI. They review it, decide it is fraudulent, and click "Reject."

4. That rejection is logged. The claim with its features is added to the training dataset. The model will be smarter next time.

5. You then show a second claim — a normal lab test for 3,000 KES with a specific ICD code filed 2 days after the service date. Rules engine: no flags. ML model: score +0.38 (normal). It is auto-approved. No human touched it.

That contrast is your demo. The judges see time saved, fraud caught, and a system that learns.

---

## What the performance report looks like

Because you have real data where `INVOICE AMOUNT ≠ SETTLED AMOUNT` (the insurer already partially rejected those claims), you can use that discrepancy as a proxy label to evaluate your model:

- Claims where `SETTLED AMOUNT < INVOICE AMOUNT` by more than 20% → treat as "known suspicious"
- Run Isolation Forest → count how many of those it correctly flags

This gives you three metrics:

| Metric | What it means in plain English |
|--------|-------------------------------|
| **Precision** | Of all the claims your model flagged, what percentage were actually suspicious? (High precision = few false alarms) |
| **Recall** | Of all the suspicious claims in the dataset, what percentage did your model catch? (High recall = few missed frauds) |
| **F1 Score** | The balance between precision and recall — a single number summarising both |

This is a required deliverable for Track 3 and your data makes it possible, since most teams will not have real labelled data to evaluate against.

---

## Privacy note on our dataset

The raw CSV contains real patient full names (`MS ROSE RACHEL ATIENO OPIYO`, etc.) and membership numbers. Before committing anything to the public GitHub repository:

1. Strip or hash the `PATIENT NAME`, `PRINCIPAL MEMBER`, and `MEMBERSHIP NO` columns
2. Document in your README that the dataset was anonymised before use
3. This anonymised dataset then becomes a strength — real Kenyan insurer data makes your model card far more credible to judges than the generic openIMIS demo database
