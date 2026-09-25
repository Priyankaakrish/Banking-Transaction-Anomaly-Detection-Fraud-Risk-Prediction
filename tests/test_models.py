"""Tests for the modelling and evaluation layer.

These guard the things that are easy to get subtly wrong and impossible to
notice from a metrics table: the sign of the anomaly score, the direction of
the cost trade-off, and whether "best" is being chosen on a meaningful metric.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import (  # noqa: E402
    confusion_at, cost_curve, pick_threshold, ranking_metrics, risk_bands,
)
from src.features import build  # noqa: E402
from src.models import AnomalyScorer, fit_baseline, fit_xgboost, with_anomaly  # noqa: E402
from src.schema import load  # noqa: E402
from src.split import temporal_split  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "sample_transactions.csv")


@pytest.fixture(scope="module")
def prepared():
    if not os.path.exists(DATA):
        pytest.skip("run scripts/make_sample.py first")
    sp = temporal_split(load(DATA))
    fb, (Xtr, Xva, Xte) = build(sp.train, sp.valid, sp.test)
    return sp, Xtr, Xva, Xte


# ----------------------------------------------------------- anomaly ------
def test_anomaly_score_points_the_right_way(prepared):
    """sklearn's score_samples is higher-is-normal; the wrapper must invert it.

    A sign error here would silently feed the model an inverted feature and
    still train without complaint.
    """
    sp, Xtr, _, Xte = prepared
    iso = AnomalyScorer(n_estimators=50, seed=0).fit(Xtr)
    s = iso.score(Xte)
    raw = iso.model_.score_samples(Xte)
    assert np.allclose(s, -raw)
    # extreme amounts should look more anomalous than typical ones
    amt = Xte["amount_abs"].to_numpy()
    assert s[np.argmax(amt)] > np.median(s)


def test_anomaly_scorer_requires_fit():
    with pytest.raises(RuntimeError):
        AnomalyScorer().score(pd.DataFrame({"a": [1.0]}))


def test_with_anomaly_does_not_mutate_input(prepared):
    _, Xtr, _, _ = prepared
    before = list(Xtr.columns)
    out = with_anomaly(Xtr, np.zeros(len(Xtr)))
    assert list(Xtr.columns) == before, "with_anomaly mutated the caller's frame"
    assert "anomaly_score" in out.columns and len(out.columns) == len(before) + 1


# ------------------------------------------------------------ metrics ----
def test_pr_auc_baseline_is_the_base_rate():
    """A random scorer should land near the positive rate, not near 0.5."""
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.005).astype(int)
    m = ranking_metrics(y, rng.random(20000))
    assert abs(m["pr_auc"] - y.mean()) < 0.01
    assert 0.45 < m["roc_auc"] < 0.55


def test_perfect_scorer_scores_one():
    y = np.array([0] * 990 + [1] * 10)
    m = ranking_metrics(y, y.astype(float))
    assert m["pr_auc"] > 0.99 and m["roc_auc"] > 0.99
    assert m["precision@50"] == pytest.approx(10 / 50)
    assert m["recall@50"] == 1.0


def test_lift_is_relative_to_base_rate():
    y = np.array([0] * 999 + [1])
    m = ranking_metrics(y, y.astype(float), ks=(10,))
    assert m["pr_auc_lift"] > 100     # 1.0 PR-AUC over a 0.001 base rate


# --------------------------------------------------------------- cost ----
def test_cost_falls_then_rises():
    """The curve must have an interior minimum, or thresholding is pointless.

    Flag nothing and every fraud is lost; flag everything and review costs
    dominate. If the optimum sat at an extreme, the cost model would be wrong.
    """
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.random(n) < 0.01).astype(int)
    scores = np.clip(rng.normal(0.2, 0.15, n) + y * 0.5, 0, 1)
    amounts = rng.lognormal(4, 1, n)
    curve = cost_curve(y, scores, amounts)
    best = pick_threshold(curve)
    assert 0 < best["flagged_pct"] < 1, "optimum sits at an extreme"
    assert best["total_cost"] < best["cost_if_no_model"]
    assert best["cost_avoided"] > 0


def test_missed_fraud_is_valued_per_transaction():
    """A missed $5,000 fraud must cost more than a missed $50 one."""
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.1, 0.9, 0.9])      # both frauds missed
    cheap = cost_curve(y, scores, np.array([50, 50, 10, 10]))
    dear = cost_curve(y, scores, np.array([5000, 5000, 10, 10]))
    assert dear["fraud_missed_value"].max() > cheap["fraud_missed_value"].max() * 50


def test_confusion_matches_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    c = confusion_at(y, s, 0.5)
    assert c == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_risk_bands_are_quantile_based():
    s = np.linspace(0, 1, 1000)
    b = risk_bands(s, low=0.90, high=0.99)
    assert (b == "high").sum() == pytest.approx(10, abs=2)
    assert (b == "medium").sum() == pytest.approx(90, abs=3)
    assert (b == "low").sum() == pytest.approx(900, abs=5)


# -------------------------------------------------------------- models ---
def test_models_beat_random_on_held_out_data(prepared):
    sp, Xtr, Xva, Xte = prepared
    ytr, yva, yte = (d["is_fraud"].to_numpy() for d in (sp.train, sp.valid, sp.test))
    if yte.sum() < 3:
        pytest.skip("too few positives to assess")

    base = fit_baseline(Xtr, ytr)
    xgb = fit_xgboost(Xtr, ytr, Xva, yva, n_estimators=120)
    m_base = ranking_metrics(yte, base.predict_proba(Xte)[:, 1])
    m_xgb = ranking_metrics(yte, xgb.predict_proba(Xte)[:, 1])

    # Both must be clearly better than chance, which at this base rate means
    # many times the positive rate rather than "above 0.5".
    assert m_base["pr_auc"] > yte.mean() * 5, "baseline no better than guessing"
    assert m_xgb["pr_auc"] > yte.mean() * 5, "xgboost no better than guessing"

    # Deliberately NOT asserting that xgboost beats the baseline. On this
    # sample it does not: with ~86 training positives the boosted model
    # overfits while a regularised linear model on well-scaled features holds
    # up. That is the point of keeping a baseline, and pinning the ordering
    # here would make the suite fail whenever the data legitimately favours
    # the simpler model.
    assert m_xgb["pr_auc"] < 0.999, "PR-AUC ~1.0 on held-out data suggests a leak"
    assert m_base["pr_auc"] < 0.999, "PR-AUC ~1.0 on held-out data suggests a leak"


def test_xgboost_handles_validation_free_fitting(prepared):
    """Early stopping must be optional, or refitting on train+valid breaks."""
    sp, Xtr, _, _ = prepared
    y = sp.train["is_fraud"].to_numpy()
    m = fit_xgboost(Xtr, y, n_estimators=30)
    assert m.predict_proba(Xtr).shape == (len(Xtr), 2)
