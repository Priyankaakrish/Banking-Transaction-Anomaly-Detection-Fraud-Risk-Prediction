"""Rolling-origin backtest.

A single train/test cut gives one number from one period, and with a ~0.1%
positive rate that number moves a lot. During development the anomaly-score
ablation read +0.0007 on one run and +0.0206 on the next — same code, different
seed. Neither is a finding; the spread is.

This retrains on an expanding window and evaluates on the block that follows,
several times, so every metric arrives with a mean and a standard deviation.
A delta smaller than its own standard deviation is noise, and the script says
so rather than leaving it to be misread.

    python -m src.backtest --data data/transactions.csv --folds 4 --nrows 2000000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .evaluate import ranking_metrics
from .features import build
from .models import AnomalyScorer, fit_baseline, fit_xgboost, with_anomaly
from .schema import load
from .split import rolling_origin_folds


def _fold_scores(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    """Fit every model on one fold and score the held-out block."""
    fb, (Xtr, Xte) = build(train, test)
    ytr, yte = train["is_fraud"].to_numpy(), test["is_fraud"].to_numpy()

    if ytr.sum() < 5 or yte.sum() < 5:
        return {}      # too few positives for the metrics to mean anything

    out = {}
    base = fit_baseline(Xtr, ytr, seed=seed)
    out["baseline"] = ranking_metrics(yte, base.predict_proba(Xte)[:, 1])

    iso = AnomalyScorer(seed=seed).fit(Xtr)
    iso_te = iso.score(Xte)
    out["isoforest"] = ranking_metrics(yte, iso_te)

    xgb = fit_xgboost(Xtr, ytr, seed=seed)
    out["xgboost"] = ranking_metrics(yte, xgb.predict_proba(Xte)[:, 1])

    xgb_iso = fit_xgboost(with_anomaly(Xtr, iso.score(Xtr)), ytr, seed=seed)
    out["xgboost+iso"] = ranking_metrics(
        yte, xgb_iso.predict_proba(with_anomaly(Xte, iso_te))[:, 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--embargo-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    os.makedirs(args.reports, exist_ok=True)

    df = load(args.data, nrows=args.nrows)
    folds = rolling_origin_folds(df, n_folds=args.folds,
                                 test_frac=args.test_frac,
                                 embargo_days=args.embargo_days)
    print(f"{len(df):,} rows | {len(folds)} folds | fraud {df['is_fraud'].mean():.4%}\n")

    rows = []
    for i, (tr, te) in enumerate(folds, 1):
        print(f"  fold {i}: train {len(tr):>9,} ({tr['is_fraud'].sum():>4} fraud) "
              f"-> test {len(te):>8,} ({te['is_fraud'].sum():>4} fraud)  "
              f"{str(te['timestamp'].min())[:10]} to {str(te['timestamp'].max())[:10]}")
        res = _fold_scores(tr, te, args.seed)
        if not res:
            print("         skipped — too few positives")
            continue
        for model, m in res.items():
            rows.append({"fold": i, "model": model, **m})

    if not rows:
        raise SystemExit("No fold had enough positives. Increase --nrows.")

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(args.reports, "backtest_folds.csv"), index=False)

    agg = (per_fold.groupby("model")[["pr_auc", "roc_auc", "base_rate"]]
           .agg(["mean", "std"]).round(4))
    print("\nAcross folds (mean ± sd):")
    print(f"  {'model':14} {'PR-AUC':>18} {'ROC-AUC':>18}")
    for model in agg.index:
        pm, ps = agg.loc[model, ("pr_auc", "mean")], agg.loc[model, ("pr_auc", "std")]
        rm, rs = agg.loc[model, ("roc_auc", "mean")], agg.loc[model, ("roc_auc", "std")]
        print(f"  {model:14} {pm:>8.4f} ± {0 if pd.isna(ps) else ps:<7.4f} "
              f"{rm:>8.4f} ± {0 if pd.isna(rs) else rs:<7.4f}")

    # --- the ablation, with an honest verdict ----------------------------
    wide = per_fold.pivot(index="fold", columns="model", values="pr_auc")
    verdict = {}
    if {"xgboost", "xgboost+iso"} <= set(wide.columns):
        deltas = (wide["xgboost+iso"] - wide["xgboost"]).dropna()
        mean_d, sd_d = float(deltas.mean()), float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        wins = int((deltas > 0).sum())
        # A difference smaller than its own spread is not evidence.
        decisive = abs(mean_d) > max(sd_d, 1e-9) and wins in (0, len(deltas))
        verdict = {
            "mean_delta": round(mean_d, 4),
            "sd_delta": round(sd_d, 4),
            "folds_improved": f"{wins}/{len(deltas)}",
            "per_fold": [round(d, 4) for d in deltas.tolist()],
            "conclusion": (
                f"anomaly score {'helps' if mean_d > 0 else 'hurts'} consistently"
                if decisive else
                "inconclusive — the difference is smaller than its variation across folds"
            ),
        }
        print(f"\nAblation across folds: Δ PR-AUC {mean_d:+.4f} ± {sd_d:.4f} "
              f"({wins}/{len(deltas)} folds improved)")
        print(f"  per fold: {verdict['per_fold']}")
        print(f"  {verdict['conclusion']}")

    with open(os.path.join(args.reports, "backtest.json"), "w") as fh:
        json.dump({"folds": len(folds),
                   "per_fold": rows,
                   "ablation": verdict}, fh, indent=2)
    print(f"\nWrote {args.reports}/backtest.json and backtest_folds.csv")


if __name__ == "__main__":
    main()
