"""Evaluation for a rare positive class.

Accuracy is meaningless here: predicting "never fraud" scores 99.9%. Even
ROC-AUC flatters, because with ~1 positive per 1,000 rows the false-positive
rate barely moves as thresholds change. **PR-AUC is the headline**, and its
baseline is the positive rate itself — 0.0042 PR-AUC on a 0.42% base rate is
no better than guessing, so the base rate is always reported beside it.

The operational metrics matter more than either. A fraud team reviews a fixed
number of alerts per day, so `precision@k` and `recall@k` answer the question
they actually ask: *of the 100 riskiest transactions today, how many are fraud,
and what share of the day's fraud did we catch?*

Threshold selection is by **cost**, not by F1. The two errors are not
symmetric: missing fraud loses the transaction amount, while a false positive
costs a few minutes of review. Optimising F1 implicitly assumes they are equal,
which is false and expensive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

REVIEW_COST = 3.0      # analyst time for one manual review, USD
FALSE_DECLINE_COST = 25.0   # goodwill cost of wrongly blocking a good customer


def ranking_metrics(y: np.ndarray, scores: np.ndarray, ks=(50, 100, 500, 1000)) -> dict:
    """Threshold-free quality, plus precision/recall at review-queue sizes."""
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }
    # PR-AUC is only meaningful relative to the base rate.
    out["pr_auc_lift"] = round(out["pr_auc"] / max(out["base_rate"], 1e-12), 1)

    order = np.argsort(scores)[::-1]
    total_pos = max(1, int(y.sum()))
    for k in ks:
        if k > len(y):
            continue
        top = y[order[:k]]
        out[f"precision@{k}"] = float(top.sum() / k)
        out[f"recall@{k}"] = float(top.sum() / total_pos)
    return out


def cost_curve(
    y: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_cost: float = REVIEW_COST,
    false_decline_cost: float = FALSE_DECLINE_COST,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """Total expected cost across thresholds.

    A missed fraud costs the transaction amount, so the cost of a false negative
    varies per row — catching a $2,000 fraud is worth far more than catching a
    $20 one, and a single averaged cost hides that. A false positive costs a
    review plus the risk of wrongly declining a genuine customer.
    """
    amounts = np.abs(np.asarray(amounts, dtype=float))
    grid = np.quantile(scores, np.linspace(0.5, 0.9999, n_thresholds))
    rows = []
    for t in np.unique(grid):
        flagged = scores >= t
        tp = flagged & (y == 1)
        fp = flagged & (y == 0)
        fn = (~flagged) & (y == 1)
        rows.append({
            "threshold": float(t),
            "flagged": int(flagged.sum()),
            "flagged_pct": float(flagged.mean()),
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
            "precision": float(tp.sum() / max(1, flagged.sum())),
            "recall": float(tp.sum() / max(1, (y == 1).sum())),
            "fraud_caught_value": float(amounts[tp].sum()),
            "fraud_missed_value": float(amounts[fn].sum()),
            "review_cost": float(flagged.sum() * review_cost
                                 + fp.sum() * false_decline_cost),
        })
    df = pd.DataFrame(rows)
    df["total_cost"] = df["fraud_missed_value"] + df["review_cost"]
    return df


def pick_threshold(curve: pd.DataFrame) -> dict:
    """The cheapest operating point, with what it costs to be there."""
    best = curve.loc[curve["total_cost"].idxmin()]
    naive = curve["fraud_missed_value"].max()   # flag nothing: lose all fraud
    return {
        "threshold": float(best["threshold"]),
        "flagged_pct": float(best["flagged_pct"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "total_cost": float(best["total_cost"]),
        "cost_if_no_model": float(naive),
        "cost_avoided": float(naive - best["total_cost"]),
        "reviews_per_1000_txns": round(float(best["flagged_pct"]) * 1000, 1),
    }


def risk_bands(scores: np.ndarray, low: float = 0.90, high: float = 0.99) -> np.ndarray:
    """Map scores to low / medium / high for a triage queue.

    Bands are score quantiles rather than fixed probabilities, because a
    calibrated 0.7 on one model is not a calibrated 0.7 on another, while
    "the riskiest 1%" is stable and is what a review team actually staffs for.
    """
    lo, hi = np.quantile(scores, low), np.quantile(scores, high)
    return np.where(scores >= hi, "high", np.where(scores >= lo, "medium", "low"))


def confusion_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (scores >= threshold).astype(int),
                                      labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def pr_points(y: np.ndarray, scores: np.ndarray, n: int = 60) -> pd.DataFrame:
    """Thinned precision-recall curve, for plotting without 100k rows."""
    p, r, t = precision_recall_curve(y, scores)
    idx = np.linspace(0, len(t) - 1, min(n, len(t))).astype(int)
    return pd.DataFrame({"threshold": t[idx], "precision": p[idx], "recall": r[idx]})
