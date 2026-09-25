"""Leakage tests.

These are the tests that matter most in this project. A fraud model with
leaking features reports excellent offline numbers and fails in production,
and the failure is invisible unless you check for it deliberately.

The central test (`test_features_are_causal`) is empirical rather than
inspective: it recomputes each row's features from a *truncated* copy of the
data containing nothing after that row, and asserts the values are identical.
If any feature peeks forward, truncation changes it and the test fails.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureBuilder, build  # noqa: E402
from src.schema import load  # noqa: E402
from src.split import (  # noqa: E402
    assert_no_time_overlap,
    rolling_origin_folds,
    temporal_split,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "sample_transactions.csv")

# Features that are pure functions of the current row cannot leak by
# construction; the causality check targets everything that uses history.
HISTORY_FEATURES = [
    "seconds_since_prev", "card_txn_seq",
    "txn_count_1h", "amount_sum_1h",
    "txn_count_24h", "amount_sum_24h",
    "txn_count_7d", "amount_sum_7d",
    "card_amount_mean_prior", "card_amount_std_prior", "amount_z_vs_card",
    "merchant_risk",
]


@pytest.fixture(scope="module")
def df():
    if not os.path.exists(DATA):
        pytest.skip("run scripts/make_sample.py first")
    return load(DATA)


@pytest.fixture(scope="module")
def split(df):
    return temporal_split(df)


# ----------------------------------------------------------------- split --
def test_split_is_ordered_and_disjoint(split):
    assert_no_time_overlap(split)


def test_embargo_leaves_a_real_gap(split):
    gap = split.valid["timestamp"].min() - split.train["timestamp"].max()
    assert gap >= pd.Timedelta(days=6), f"embargo gap too small: {gap}"


def test_every_part_contains_fraud(split):
    """A test set with no positives cannot measure anything."""
    for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        assert part["is_fraud"].sum() > 0, f"{name} has no fraud rows"


def test_rolling_origin_folds_are_ordered(df):
    folds = rolling_origin_folds(df, n_folds=3)
    assert len(folds) >= 2
    for tr, te in folds:
        assert tr["timestamp"].max() < te["timestamp"].min()


def test_random_split_would_leak(df):
    """Documents why the temporal split exists.

    Under a random split the same card appears on both sides with interleaved
    timestamps, which is exactly the condition that lets rolling features see
    the future. Asserting it here makes the failure mode explicit rather than
    a comment nobody reads.
    """
    shuffled = df.sample(frac=1.0, random_state=0)
    cut = int(len(shuffled) * 0.8)
    tr, te = shuffled.iloc[:cut], shuffled.iloc[cut:]
    shared = set(tr["card_id"]) & set(te["card_id"])
    assert len(shared) > 0
    # and at least one test row precedes a train row on the same card
    overlap = tr["timestamp"].max() > te["timestamp"].min()
    assert overlap, "expected a random split to interleave in time"


# -------------------------------------------------------------- causality --
def test_features_are_causal(split):
    """Recompute each probe row's features with the future deleted.

    If a feature used any information from after the row, truncating the frame
    would change its value. Identical values means the feature is computable at
    scoring time, which is the property that has to hold in production.
    """
    train = split.train
    fb = FeatureBuilder().fit(train)
    full = fb.transform(train)

    rng = np.random.default_rng(0)
    probes = sorted(rng.choice(np.arange(200, len(train)), size=12, replace=False))

    for i in probes:
        truncated = train.iloc[: i + 1].reset_index(drop=True)
        # Same fitted statistics — only the rows available change.
        partial = fb.transform(truncated)
        a = full.loc[i, HISTORY_FEATURES].to_numpy(dtype=float)
        b = partial.loc[i, HISTORY_FEATURES].to_numpy(dtype=float)
        bad = [HISTORY_FEATURES[j] for j in range(len(a))
               if not np.isclose(a[j], b[j], rtol=1e-5, atol=1e-6, equal_nan=True)]
        assert not bad, f"row {i}: these features changed when the future was removed: {bad}"


def test_merchant_risk_excludes_own_row(split):
    """A target encoding that includes its own label hands the model the answer."""
    train = split.train
    fb = FeatureBuilder().fit(train)
    X = fb.transform(train)

    # First transaction at any merchant has no prior history, so its encoding
    # must sit exactly at the smoothed global prior regardless of its label.
    first = ~train.duplicated(subset=["merchant"], keep="first")
    firsts = X.loc[first, "merchant_risk"]
    assert np.allclose(firsts, fb.fraud_prior_, atol=1e-6), (
        "first-sighting merchant_risk should equal the prior, not the row's own label"
    )
    # and a fraudulent first sighting must not read higher than a clean one
    lab = train.loc[first, "is_fraud"].to_numpy()
    if lab.sum() and (1 - lab).sum():
        assert abs(firsts[lab == 1].mean() - firsts[lab == 0].mean()) < 1e-6


def test_rolling_windows_exclude_current_row(split):
    """The first transaction on a card has no history, so counts must be zero."""
    train = split.train
    fb = FeatureBuilder().fit(train)
    X = fb.transform(train)
    first_on_card = train.groupby("card_id", sort=False).head(1).index
    for col in ("txn_count_1h", "txn_count_24h", "txn_count_7d",
                "amount_sum_1h", "amount_sum_24h", "amount_sum_7d"):
        assert (X.loc[first_on_card, col] == 0).all(), f"{col} nonzero on a card's first row"
    assert (X.loc[first_on_card, "seconds_since_prev"] == -1).all()


def test_constant_history_still_flags_a_deviation():
    """Zero variance must not collapse the z-score.

    A card with perfectly regular spending — a recurring subscription — has a
    prior standard deviation of exactly zero. Dividing by it sends the z-score
    to NaN and then to 0, which marked the single most unusual transaction on
    that card as perfectly normal. Found while testing the batch endpoint:
    nine identical $50 charges followed by $2,500 scored z = 0.00.
    """
    rows = []
    for i in range(1, 13):
        rows.append({
            "card_id": "c1", "user": 1, "card": 1,
            "timestamp": pd.Timestamp("2019-05-01") + pd.Timedelta(days=i),
            "amount": 50.0 if i < 10 else 2500.0,
            "merchant": "m1", "merchant_state": "CA", "merchant_city": "c",
            "zip": "1", "mcc": 5411, "errors": "", "use_chip": "Swipe Transaction",
            "is_fraud": 0,
        })
    df = pd.DataFrame(rows)
    fb = FeatureBuilder().fit(df)
    X = fb.transform(df)
    assert (X.loc[:8, "card_amount_std_prior"] == 0).all(), "expected zero-variance history"
    assert X.loc[9, "amount_z_vs_card"] > 10, (
        "a 50x deviation on a constant-history card must not read as normal"
    )


def test_statistics_are_fitted_on_train_only(split):
    """Test-set statistics must not influence the encoding applied to test."""
    fb = FeatureBuilder().fit(split.train)
    prior_before = fb.fraud_prior_
    mcc_before = dict(fb.mcc_stats_)
    fb.transform(split.test)
    assert fb.fraud_prior_ == prior_before
    assert fb.mcc_stats_ == mcc_before, "transform() mutated fitted statistics"


def test_unseen_category_falls_back_to_prior(split):
    """An MCC absent from training must not produce NaN at scoring time."""
    fb = FeatureBuilder().fit(split.train)
    probe = split.test.head(50).copy()
    probe["mcc"] = 9999                       # never seen in training
    X = fb.transform(probe)
    assert X["mcc_risk"].notna().all()
    assert np.allclose(X["mcc_risk"], fb.fraud_prior_)


def test_no_feature_is_a_perfect_predictor(split):
    """A single feature separating the classes perfectly means leakage.

    Real fraud signals are noisy and overlapping. An AUC of 1.0 from one column
    is a bug, not a breakthrough.
    """
    from sklearn.metrics import roc_auc_score

    fb = FeatureBuilder().fit(split.train)
    X = fb.transform(split.train)
    y = split.train["is_fraud"].to_numpy()
    if y.sum() == 0:
        pytest.skip("no positives")
    for col in X.columns:
        v = X[col].to_numpy(dtype=float)
        if np.all(np.isfinite(v)) and len(np.unique(v)) > 1:
            auc = roc_auc_score(y, v)
            assert max(auc, 1 - auc) < 0.999, f"{col} separates the classes perfectly — leak"


def test_build_applies_one_fitted_builder_to_all_splits(split):
    fb, (Xtr, Xva, Xte) = build(split.train, split.valid, split.test)
    assert list(Xtr.columns) == list(Xva.columns) == list(Xte.columns)
    assert len(Xtr) == len(split.train) and len(Xte) == len(split.test)
    assert Xte.notna().all().all(), "NaNs reaching the model from the test split"


# ------------------------------------------------------------- enrichment --
def test_enrichment_excludes_leaky_fields_by_default():
    """`Card on Dark Web` is a present-day status flag with no timestamp.

    A card is listed on the dark web *because* it was compromised, often after
    the fraud it would be used to predict. It is the best-looking feature in
    the dataset and must not be in the default feature set.
    """
    from src.enrich import enrich, enriched_columns

    users = os.path.join(os.path.dirname(DATA), "sample_users.csv")
    cards = os.path.join(os.path.dirname(DATA), "sample_cards.csv")
    if not (os.path.exists(users) and os.path.exists(cards)):
        pytest.skip("reference files not generated")

    df = load(DATA)
    safe = enrich(df, users, cards)
    assert "card_on_dark_web" not in enriched_columns(safe)

    unsafe = enrich(df, users, cards, include_unsafe=True)
    assert "card_on_dark_web" in enriched_columns(unsafe, include_unsafe=True)


def test_age_is_computed_per_transaction_not_taken_as_current():
    """'Current Age' is as-of-today; in a multi-year dataset that leaks time."""
    from src.enrich import enrich

    users = os.path.join(os.path.dirname(DATA), "sample_users.csv")
    if not os.path.exists(users):
        pytest.skip("reference files not generated")
    df = load(DATA)
    e = enrich(df, users_path=users)
    assert "age_at_txn" in e.columns
    assert "Current Age" not in e.columns and "current_age" not in e.columns
    # age must track the transaction year, so it varies within one user
    per_user = e.groupby("user")["age_at_txn"].nunique()
    assert per_user.max() > 1, "age_at_txn is constant per user — it is not time-aware"


def test_join_does_not_drop_or_duplicate_rows():
    """A silently shrinking join changes the base rate and the evaluation."""
    from src.enrich import enrich

    users = os.path.join(os.path.dirname(DATA), "sample_users.csv")
    cards = os.path.join(os.path.dirname(DATA), "sample_cards.csv")
    if not (os.path.exists(users) and os.path.exists(cards)):
        pytest.skip("reference files not generated")
    df = load(DATA)
    e = enrich(df, users, cards)
    assert len(e) == len(df)
    assert e["is_fraud"].sum() == df["is_fraud"].sum()
