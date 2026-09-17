# Banking Transaction Anomaly Detection & Fraud Risk Prediction

Leakage-safe feature engineering and temporal evaluation for card fraud
detection on the [Altman credit-card transactions dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions).

## Status

Built and tested: schema normalisation, temporal splitting, causal feature
engineering, four models with an ablation, cost-based thresholding, and a
24-test suite.

Also built: rolling-origin backtest, SHAP explanations, MLflow tracking,
FastAPI scoring service, Docker image. 34 tests.

## The label is fraud, not chargeback

The dataset carries `Is Fraud?`. Chargebacks and fraud overlap but are not the
same thing — friendly fraud produces a chargeback with no fraud flag, and fraud
caught before settlement often never becomes one. This predicts fraud.

## Why the split is temporal

A random split inflates fraud scores rather than lowering them, which is why it
survives into so many write-ups. Two mechanisms leak:

- **Rolling features look forwards.** A test row placed before a training row
  from the same card means the model was fitted on aggregates containing the
  test transaction.
- **Fraud is bursty.** A compromised card produces a run of transactions in
  minutes. Scatter them across train and test and the model memorises the burst.

So the data is cut by time, with a **7-day embargo** between parts — the longest
rolling window — removing the last path for information to cross the boundary.

## Every feature is causal

Each feature answers *what could a scoring system have known when this
transaction arrived?*

- Rolling windows are shifted one row within each card, so no window contains
  its own row.
- Target encodings (merchant risk, MCC risk) are expanding past-only means,
  smoothed toward the global prior so a merchant's first transaction does not
  read 0.0 or 1.0.
- Global statistics are fitted on train and reused, never recomputed on test.

`tests/test_leakage.py` verifies this **empirically**: it recomputes each probe
row's features from a truncated frame containing nothing after that row and
asserts the values are unchanged. Anything peeking forward fails.

```
test_features_are_causal                  recompute on truncated data
test_merchant_risk_excludes_own_row       first sighting == prior
test_rolling_windows_exclude_current_row  card's first row has zero history
test_statistics_are_fitted_on_train_only  transform() must not mutate fit state
test_no_feature_is_a_perfect_predictor    single-feature AUC 1.0 means a leak
test_random_split_would_leak              documents the failure mode
```

That last-but-one test earned its place immediately: it failed on the first run
because the synthetic generator made every fraudulent transaction "Online", so
channel alone separated the classes. The features were fine; the data was not.

## Quick start

```bash
make setup     # install dependencies
make smoke     # generate data, train, test — no Kaggle download needed
```

Or step by step:

```bash
pip install -r requirements.txt
python scripts/make_sample.py data/sample_transactions.csv
python -m src.train --data data/sample_transactions.csv --out artifacts
python -m pytest -q tests/
```

### With the real dataset

Download `credit_card_transactions-ibm_v2.csv` (24M rows, ~2.4 GB) into `data/`:

```bash
make train DATA="data/credit_card_transactions-ibm_v2.csv" NROWS=2000000
```

Start at 2M rows. A full pass needs roughly 6-8 GB for the raw frame plus a
2.6 GB feature matrix.

Expect PR-AUC well below the synthetic figures below — real fraud is far
subtler than a planted burst, and a large drop is the correct outcome rather
than a bug. The dataset also ships `sd254_users.csv` and `sd254_cards.csv`
(age, income, credit limit, card type); joining them adds genuine signal,
since amount relative to credit limit is a strong fraud feature.

In code:

```python
from src.schema import load, summarise
from src.split import temporal_split, assert_no_time_overlap
from src.features import build

df = load("data/transactions.csv")          # 24M rows; pass nrows= to sample
split = temporal_split(df, embargo_days=7)
assert_no_time_overlap(split)
fb, (Xtr, Xva, Xte) = build(split.train, split.valid, split.test)
```

## Results

Synthetic data, 4,032 test rows, 17 positives, **0.42% base rate**.

| model | PR-AUC | lift | ROC-AUC | P@100 | R@100 |
|---|---|---|---|---|---|
| baseline (logistic) | 0.5525 | 131x | 0.9707 | 0.140 | 0.824 |
| isolation forest alone | 0.1310 | 31x | 0.9247 | 0.060 | 0.353 |
| xgboost | 0.8543 | 203x | 0.9588 | 0.150 | 0.882 |
| xgboost + anomaly score | 0.8550 | 203x | 0.9632 | 0.150 | 0.882 |

PR-AUC is the headline; its baseline is the positive rate itself, so the lift
column matters more than the raw figure. Note that ROC-AUC is *highest for the
worst model* — the logistic baseline reads 0.9707 against XGBoost's 0.9588
while having barely a third of the PR-AUC. At a 0.4% base rate ROC-AUC is
dominated by the negatives and is close to useless for ranking these models.
That contrast is the reason PR-AUC leads.

These figures come from generated data with a deliberately learnable pattern.
They demonstrate the pipeline runs; they are not expected real-world numbers.

### The anomaly-score ablation

The plan assumed an Isolation Forest score would improve the supervised model.
Measured rather than assumed:

| | PR-AUC |
|---|---|
| xgboost | 0.8543 |
| xgboost + anomaly score | 0.8550 |
| **delta** | **+0.0007 (+0.1%)** |

**It does not measurably help.** The score earns 0.031 feature importance —
non-zero but minor — and alone it manages 0.131 PR-AUC against XGBoost's 0.854.

That is the expected result once labels exist, and worth stating plainly.
"Unusual" and "fraudulent" are different properties: most legitimate outliers
are a customer buying a fridge. Unsupervised anomaly detection earns its place
when labels are scarce or attacks are novel, not as a free accuracy boost on a
labelled problem. Keeping the ablation in the pipeline means the claim stays
honest if the data changes.

### Operating point

Threshold chosen by **cost**, not F1. The errors are not symmetric: a missed
fraud loses the transaction amount, while a false positive costs a few minutes
of review plus the risk of wrongly declining a good customer.

| | |
|---|---|
| Flags | 0.52% of transactions (5.2 reviews per 1,000) |
| Precision | 0.714 |
| Recall | 0.882 |
| Cost | $372 vs $4,035 with no model |
| Avoided | **$3,663** |

Optimising F1 would implicitly price a missed fraud and a wasted review
equally, which is false and expensive.

Top features: `amount_z_vs_card`, `amount`, `amount_abs`, `is_online`,
`anomaly_score`, `amount_sum_24h`. The card-relative z-score dominating is the
expected shape — fraud is unusual *for that card*, not unusual in absolute
terms. If raw amount had led instead, the card-relative features would not be
working.

## Layout

```
src/schema.py      currency/timestamp parsing, type normalisation, sorting
src/split.py       temporal split with embargo, rolling-origin backtest folds
src/features.py    causal features: velocity, card behaviour, merchant risk
src/models.py      logistic baseline, Isolation Forest scorer, XGBoost
src/evaluate.py    PR-AUC, precision@k, cost curve, threshold selection
src/train.py       runs all four models and the ablation
scripts/make_sample.py  synthetic data with the real schema and planted bursts
src/backtest.py    rolling-origin folds with variance
serve/api.py       FastAPI scoring service, history-aware
Dockerfile         slim runtime image, non-root, healthcheck
tests/test_leakage.py   13 tests — causality, splitting, target encoding
tests/test_models.py    12 tests — score direction, cost model, metric sanity
tests/test_api.py        9 tests — history handling, batch, validation
```

```bash
python -m src.train --data data/sample_transactions.csv --out artifacts
python -m pytest -q tests/        # 24 tests
```

### A note on sampling

No SMOTE or random oversampling. Synthesising minority rows by interpolating
between fraud cases invents transactions that never happened, and with causal
time-series features it can interpolate across the time boundary.
`scale_pos_weight` reweights the loss without fabricating data.


## Explanations

```bash
python -m src.explain --artifacts artifacts --data data/sample_transactions.csv
```

Two audiences, two outputs, and conflating them is the usual mistake.

**Global** (`reports/shap_global.csv`) is a sanity check for the modeller. If
`merchant_risk` dominated, the target encoding would probably be leaking; if
`hour` dominated, the model has latched onto a scheduling artefact.

**Local** (`reports/shap_cases.json`) is for the fraud analyst opening an alert.
Rendered as sentences, not SHAP values — nobody reviewing a hundred alerts a day
reads a force plot:

```
score 1.000  $444.85  [FRAUD]
  - amount vs this card's normal spend (17.2 standard deviations above)
  - overall unusualness vs normal traffic (0.62)
  - transaction amount (444.85)
  - card-not-present (online)
```

"Amount is 17 standard deviations above this card's normal spend" is
investigable. A risk score of 0.87 is not. In several jurisdictions an adverse
decision also has to be explainable, so this is a compliance requirement rather
than a nicety.

## Experiment tracking

```bash
python -m src.train --data data/sample_transactions.csv --mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

One parent run per training job with a nested run per model. The nesting
matters: the four models share a split and a feature set, so logging them as
separate top-level runs would lose the fact that they are one experiment. The
ablation delta sits on the parent, because it is a property of the comparison
rather than of either model.

SQLite rather than the default file store — MLflow 3 put the filesystem backend
into maintenance mode and refuses it without an opt-out.

## Serving

```bash
uvicorn serve.api:app --port 8000        # or: docker build -t fraud-api . && docker run -p 8000:8000 fraud-api
curl localhost:8000/health
```

### The history problem

Every useful feature here is causal — transactions in the last hour, amount
relative to the card's past average, whether this merchant is new for the card.
**None of it can be computed from a single incoming transaction.** A stateless
`POST /predict` taking one JSON object is therefore either lying or scoring on
a crippled feature set, which is the quiet failure in most fraud-API demos.

The service handles it three ways and reports which applied:

| `history_source` | meaning |
|---|---|
| `request` | caller supplied the card's recent activity — preferred |
| `cache` | service used its own in-memory recent-transaction store |
| `none` | cold start; transaction-level features only |

Every response carries `history_depth` and, when history is thin, a
`confidence_note`. A score computed with no history is not equivalent to one
computed with fifty prior transactions, and the API says so rather than
returning a number that looks equally authoritative.

The in-process cache is explicitly a development convenience: it does not
survive a restart and is not shared across replicas. Production wants Redis or
DynamoDB with a TTL.

## Two bugs the tests caught

**Zero variance collapsed the z-score.** Nine identical $50 charges followed by
$2,500 produced `amount_z_vs_card = 0.00` — the most anomalous transaction on
the card scored as perfectly normal. A card with perfectly regular spending (a
recurring subscription) has a prior standard deviation of exactly zero, and
dividing by it gave NaN, which was then filled with 0. Flooring the denominator
at 10% of the card's mean fixed it: the same transaction now reads z = 490.
Found while testing the batch endpoint, not by reading the code.

**The baseline beat XGBoost.** After that fix, logistic regression scored 0.762
PR-AUC against plain XGBoost's 0.748. With only ~86 training positives the
boosted model overfits while a regularised linear model on a well-scaled
feature holds up. A test asserting "XGBoost wins" was removed rather than
worked around — pinning that ordering would make the suite fail whenever the
data legitimately favours the simpler model. This is what baselines are for.

## Variance, not single numbers

The anomaly-score ablation read +0.0007 on one run and +0.0206 on the next —
same code, different seed. With ~17 positives in a test window, neither is a
finding.

`src/backtest.py` retrains on an expanding window and evaluates on the block
that follows, several times:

```bash
python -m src.backtest --data data/sample_transactions.csv --folds 4
```

| model | PR-AUC across 4 folds |
|---|---|
| baseline | 0.5417 ± 0.0511 |
| isolation forest | 0.2466 ± 0.0155 |
| xgboost | 0.8319 ± 0.0782 |
| xgboost + anomaly | 0.8334 ± 0.0700 |

Ablation: **Δ +0.0015 ± 0.0111, 3/4 folds improved — inconclusive.** The
difference is smaller than its own variation, and the script says so rather
than reporting the mean as a result.

## Performance

Feature building was rewritten after benchmarking showed it would not finish on
the real dataset. The original used `groupby.apply` with Python lambdas, whose
cost scales with the number of *groups* — and the data has roughly 6,000 cards.

| cards | groupby.apply | prefix sums + searchsorted |
|---|---|---|
| 600 | 34 min | 1.1 min |
| 3,000 | 172 min | 2.1 min |
| 6,000 | ~6 hours | **~1 min** |

Projected to 24M rows. The rewrite computes the same windows with one binary
search per group and vectorised operations inside, and all 24 tests still pass
— which is what the causality suite is for: it made a large refactor safe.
