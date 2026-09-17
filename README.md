# Banking Transaction Anomaly Detection & Fraud Risk Prediction

Leakage-safe feature engineering and temporal evaluation for card fraud
detection on the [Altman credit-card transactions dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions).

## Status

Built and tested: schema normalisation, temporal splitting, causal feature
engineering, and a 12-test leakage suite.

Next: Isolation Forest + XGBoost with the anomaly-score ablation, cost-based
threshold selection, SHAP, FastAPI and Docker.

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
pip install pandas numpy scikit-learn xgboost pytest
python scripts/make_sample.py data/sample_transactions.csv   # no download needed
python -m pytest -q tests/
```

With the real data at `data/transactions.csv`:

```python
from src.schema import load, summarise
from src.split import temporal_split, assert_no_time_overlap
from src.features import build

df = load("data/transactions.csv")          # 24M rows; pass nrows= to sample
split = temporal_split(df, embargo_days=7)
assert_no_time_overlap(split)
fb, (Xtr, Xva, Xte) = build(split.train, split.valid, split.test)
```

## Baseline on synthetic data

| | |
|---|---|
| Test rows | 4,032 (17 positives, 0.42% base rate) |
| PR-AUC | 0.8435 |
| ROC-AUC | 0.9553 |

PR-AUC against a 0.0042 base rate is the number that matters; ROC-AUC looks
flattering at this imbalance and should not be led with. These figures come
from generated data with a deliberately learnable pattern — they demonstrate
the pipeline runs, not expected real-world performance.

Top features: `amount_z_vs_card` (0.42), `amount` (0.19), `is_online` (0.06) —
the card-relative z-score dominating is the expected shape, since fraud is
unusual *for that card* rather than unusual in absolute terms.

## Layout

```
src/schema.py      currency/timestamp parsing, type normalisation, sorting
src/split.py       temporal split with embargo, rolling-origin backtest folds
src/features.py    causal features: velocity, card behaviour, merchant risk
scripts/make_sample.py  synthetic data with the real schema and planted bursts
tests/test_leakage.py   12 tests, the core of the project
```
