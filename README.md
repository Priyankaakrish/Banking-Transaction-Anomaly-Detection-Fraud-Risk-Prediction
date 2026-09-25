# Banking Transaction Anomaly Detection & Fraud Risk Prediction

Fraud detection on 24M credit-card transactions, built around the thing that
decides whether a fraud model is real or not: **whether the features could have
been computed at the moment the transaction arrived.**

Leakage-safe feature engineering, temporal validation with an embargo, four
models with an ablation, cost-based threshold selection, SHAP explanations,
MLflow tracking, and a FastAPI scoring service.

```
Transactions ─▶ causal features ─▶ Isolation Forest ─▶ XGBoost ─▶ cost-based threshold ─▶ decision
                      │                                                    │
               temporal split                                     SHAP explanation
```

---

## Results on real data

2,000,000 transactions (1995–2020), 300,000 held-out test rows, **341 fraud
cases at a 0.11% base rate**.

| model | PR-AUC | lift | ROC-AUC | P@100 | R@100 |
|---|---|---|---|---|---|
| **logistic baseline** | **0.1480** | **130x** | 0.9459 | 0.340 | 0.100 |
| isolation forest | 0.0125 | 11x | 0.8685 | 0.010 | 0.003 |
| xgboost | 0.0756 | 66x | 0.9380 | 0.210 | 0.062 |
| xgboost + anomaly score | 0.0699 | 62x | 0.9443 | 0.160 | 0.047 |

PR-AUC is the headline and its baseline is the positive rate itself, so **lift
matters more than the raw figure**: 0.148 against a 0.0011 base rate is 130x
better than chance.

Note that ROC-AUC ranks the models almost identically (0.94 for three of them)
while PR-AUC separates them 2:1. At this imbalance ROC-AUC is dominated by the
negatives and is close to useless for model selection — which is why it is
reported but not led with.

---

## Four findings

### 1. A similarity-style threshold cannot detect what the model has not seen

Out-of-domain and unusual-but-legitimate transactions are not separable by score
alone. The same applies to the fraud/normal boundary: the cost of being wrong is
asymmetric and score thresholds have to be chosen against that cost, not against
F1.

### 2. The logistic baseline beat XGBoost, 2:1

0.1480 against 0.0756. That is not noise at 341 positives.

The likely cause is `scale_pos_weight ≈ 943` making the boosted model
over-aggressive, compounded by early stopping on a validation window whose fraud
rate (0.131%) differs materially from test (0.114%). A regularised linear model
on well-scaled causal features is simply harder to destabilise at this level of
imbalance.

This is why the baseline is in the pipeline rather than assumed away. A test
asserting "XGBoost wins" was deleted rather than worked around — pinning that
ordering would make the suite fail whenever the data legitimately favours the
simpler model.

### 3. A made-up cost assumption inverted the conclusion

The first real-data run reported a **negative** saving: $26,287 with the model
against $21,493 without, with the optimiser pushing the threshold to 1.0000 —
"flag as little as possible".

The model was not the problem. The assumed **$25 false-decline penalty** was,
against an actual median fraud of **$79** (mean $124, p90 $247). Wrongly
declining three good customers cost about as much as the fraud caught, so
refusing to act really was cheapest under those numbers.

`--review-cost` and `--false-decline-cost` now make the assumption explicit, and
every run prints a sensitivity table:

```
 review  decline   flagged   prec  recall      avoided
$  0.50 $   2.00    2.53%  0.147   0.882 $      5,219
$  1.50 $   5.00    0.52%  0.667   0.824 $      5,166
$  3.00 $  10.00    0.52%  0.667   0.824 $      5,099
$  3.00 $  25.00    0.52%  0.667   0.824 $      4,994
```

Reporting a single operating point invites the reader to treat it as fact. The
defensible claim is the shape of the table, not one number inside it.

### 4. The anomaly score did not help

The plan assumed an Isolation Forest score would improve the supervised model.
Measured instead of assumed, on real data: **ΔPR-AUC −0.0057 (−7.6%)**. Alone,
the anomaly score manages 0.0125 PR-AUC against the baseline's 0.148.

That is the expected result once labels exist. "Unusual" and "fraudulent" are
different properties — most legitimate outliers are a customer buying a fridge.
Unsupervised anomaly detection earns its place when labels are scarce or attacks
are novel, not as a free accuracy boost on a labelled problem.

On synthetic data the same ablation returned +0.0007, +0.0206 and +0.0475 across
three runs. Any single number here is noise, which is why `src/backtest.py`
reports the delta with a standard deviation across four time windows.

---

## Why the validation is temporal

A random split inflates fraud scores rather than lowering them, which is why it
survives into so many write-ups. Two mechanisms leak:

- **Rolling features look forwards.** A test row placed before a training row
  from the same card means the model was fitted on aggregates that already
  contain the test transaction.
- **Fraud is bursty.** A compromised card produces a run of transactions in
  minutes. Scatter them across train and test and the model memorises the burst;
  recall looks excellent and collapses in production.

The data is cut by time with a **7-day embargo** between parts — the longest
rolling window — removing the last path for information to cross the boundary.

### Every feature is causal

Each answers *what could a scoring system have known when this transaction
arrived?*

- Rolling windows are shifted one row within each card, so no window contains
  its own row.
- Target encodings (merchant risk, MCC risk) are expanding past-only means,
  smoothed toward the global prior so a merchant's first transaction does not
  read 0.0 or 1.0.
- Global statistics are fitted on train and reused, never recomputed on test.

`tests/test_leakage.py` verifies this **empirically**: it recomputes each probe
row's features from a truncated frame containing nothing after that row and
asserts the values are unchanged. Anything peeking forward fails.

| test | what it catches |
|---|---|
| `test_features_are_causal` | any feature that changes when the future is removed |
| `test_merchant_risk_excludes_own_row` | target encoding containing its own label |
| `test_rolling_windows_exclude_current_row` | a card's first row showing history |
| `test_no_feature_is_a_perfect_predictor` | single-feature AUC of 1.0 |
| `test_enrichment_excludes_leaky_fields` | `Card on Dark Web` reaching the model |
| `test_constant_history_still_flags_a_deviation` | zero-variance z-score collapse |

Two of those earned their place immediately. `test_no_feature_is_a_perfect_predictor`
failed on the first run because the synthetic generator made every fraudulent
transaction "Online" — the features were fine, the data was not.
`test_constant_history_still_flags_a_deviation` was written after the batch
endpoint revealed that nine identical $50 charges followed by $2,500 produced
`amount_z_vs_card = 0.00`: a card with perfectly regular spending has zero prior
variance, and dividing by it marked the most anomalous transaction on that card
as perfectly normal.

---

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python scripts/make_sample.py data/sample_transactions.csv   # no download needed
python -m src.train --data data/sample_transactions.csv
python -m pytest -q tests/                                    # 37 tests
```

With the [real dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)
(`credit_card_transactions-ibm_v2.csv`, 24M rows, 2.4 GB) in `data/`:

```bash
python -m src.train --data "data/credit_card_transactions-ibm_v2.csv" \
  --nrows 2000000 --review-cost 1.5 --false-decline-cost 5.0 --mlflow
python -m src.backtest --data "data/credit_card_transactions-ibm_v2.csv" --folds 4
python -m src.explain  --artifacts artifacts --data "data/credit_card_transactions-ibm_v2.csv"
```

The dataset also ships `sd254_users.csv` and `sd254_cards.csv`. `src/enrich.py`
joins them for `amount_vs_limit`, `fico`, `debt_to_income` and others — **with
two fields excluded by default**. `Card on Dark Web` is a present-day status
flag with no timestamp, recorded *because* a card was compromised, often after
the fraud it would predict. `Current Age` is age today, not age at transaction
time. Both produce excellent offline scores and unusable models.

---

## Serving

```bash
uvicorn serve.api:app --port 8000     # then http://127.0.0.1:8000/docs
docker build -t fraud-api . && docker run -p 8000:8000 fraud-api
```

### The history problem

Every useful feature here is causal — transactions in the last hour, amount
relative to the card's past average, whether this merchant is new. **None of it
can be computed from a single incoming transaction.** A stateless `POST /predict`
taking one JSON object is therefore either lying or scoring on a crippled
feature set, which is the quiet failure in most fraud-API demos.

The service resolves it three ways and reports which applied:

| `history_source` | meaning |
|---|---|
| `request` | caller supplied the card's recent activity — preferred |
| `cache` | service used its own in-memory recent-transaction store |
| `none` | cold start; transaction-level features only |

Every response carries `history_depth` and, when history is thin, a
`confidence_note`. A score computed with no history is not equivalent to one
computed with fifty prior transactions, and the API says so.

The in-process cache is a development convenience: it does not survive a restart
and is not shared across replicas. Production wants Redis or DynamoDB with a TTL.

---

## Performance

Feature building was rewritten after benchmarking showed it would not finish on
the real dataset. The original used `groupby.apply` with Python lambdas, whose
cost scales with the number of *groups* — and the data has roughly 6,000 cards.

| cards | groupby.apply | prefix sums + searchsorted |
|---|---|---|
| 600 | 34 min | 1.1 min |
| 3,000 | 172 min | 2.1 min |
| 6,000 | ~6 hours | **~1 min** |

Projected to 24M rows. All 37 tests passed unchanged afterwards — which is what
the causality suite is for: it made a large refactor safe.

Actual: 2M rows load in 19.2s, 27 features build in 5.2s.

---

## Layout

```
src/schema.py       currency/timestamp parsing, type normalisation, sorting
src/split.py        temporal split with embargo, rolling-origin backtest folds
src/features.py     causal features: velocity, card behaviour, merchant risk
src/enrich.py       user/card reference join, with leaky fields excluded
src/models.py       logistic baseline, Isolation Forest scorer, XGBoost
src/evaluate.py     PR-AUC, precision@k, cost curve, threshold selection
src/train.py        four models, the ablation, cost sensitivity, MLflow
src/backtest.py     rolling-origin folds with variance
src/explain.py      SHAP: global drivers and plain-language per-transaction reasons
serve/api.py        FastAPI scoring service, history-aware
Dockerfile          slim runtime image, non-root, healthcheck
tests/              37 tests
```

---

## Known limitations

**Trained on 2M of 24M rows.** The first 2M are weighted toward the early years
when volume was lower; sampling a recent slice would be more representative.

**No hyperparameter tuning.** XGBoost runs on sensible defaults. Given finding
2, tuning `scale_pos_weight` is the obvious first experiment.

**The Docker image has not been built.** It is written carefully — slim base,
layer caching, non-root user, healthcheck — but untested, unlike everything else
here.

**Fraud, not chargeback.** The dataset labels `Is Fraud?`. Chargebacks and fraud
overlap but differ: friendly fraud produces a chargeback with no fraud flag, and
fraud caught before settlement often never becomes one.

---

## Data

The dataset is not redistributed here — download it from Kaggle under its
own licence.
