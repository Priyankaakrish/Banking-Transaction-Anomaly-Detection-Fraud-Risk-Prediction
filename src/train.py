"""End-to-end training and the anomaly-score ablation.

    python -m src.train --data data/sample_transactions.csv --out artifacts

Runs four models on identical splits and features:

    baseline       logistic regression
    isoforest      Isolation Forest score used alone as a predictor
    xgboost        supervised only
    xgboost+iso    supervised, with the anomaly score as one extra feature

The last two exist to answer a question the project plan takes for granted:
*does the unsupervised score actually help once labels are available?* Running
it as an ablation means the answer is measured. A negligible delta is a
legitimate finding and a more honest one than asserting the hybrid is better.

Everything is fitted on train, tuned on validation, and reported once on test.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from .evaluate import cost_curve, pick_threshold, pr_points, ranking_metrics
from .features import build
from .models import AnomalyScorer, fit_baseline, fit_xgboost, with_anomaly
from .schema import load, summarise
from .split import assert_no_time_overlap, temporal_split


def _scores(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--nrows", type=int, default=None,
                    help="Cap rows read. The full dataset is ~24M; start smaller.")
    ap.add_argument("--embargo-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow", action="store_true",
                    help="Log run, params, metrics and artifacts to MLflow.")
    ap.add_argument("--experiment", default="fraud-detection")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                    help="MLflow backend. The file store is deprecated in "
                         "MLflow 3, so sqlite is the default.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)

    # ---- data ------------------------------------------------------------
    t0 = time.time()
    df = load(args.data, nrows=args.nrows)
    info = summarise(df)
    print(f"Loaded {info['rows']:,} rows | fraud {info['fraud_rate']:.4%} | "
          f"{info['from'][:10]} to {info['to'][:10]} | {time.time()-t0:.1f}s")

    split = temporal_split(df, embargo_days=args.embargo_days)
    assert_no_time_overlap(split)          # cheap, and catches a silent disaster
    s = split.summary()
    for part in ("train", "valid", "test"):
        print(f"  {part:5} {s[part+'_rows']:>9,} rows  "
              f"{s[part+'_fraud']:>5} fraud ({s[part+'_fraud_rate']:.4%})")

    # ---- features --------------------------------------------------------
    t0 = time.time()
    fb, (Xtr, Xva, Xte) = build(split.train, split.valid, split.test)
    ytr, yva, yte = (d["is_fraud"].to_numpy() for d in
                     (split.train, split.valid, split.test))
    amounts_te = split.test["amount"].to_numpy()
    print(f"Built {Xtr.shape[1]} features in {time.time()-t0:.1f}s")

    results, models = {}, {}

    # ---- 1. baseline -----------------------------------------------------
    t0 = time.time()
    base = fit_baseline(Xtr, ytr, seed=args.seed)
    results["baseline"] = ranking_metrics(yte, _scores(base, Xte))
    results["baseline"]["fit_seconds"] = round(time.time() - t0, 1)
    models["baseline"] = base

    # ---- 2. isolation forest, alone --------------------------------------
    t0 = time.time()
    iso = AnomalyScorer(seed=args.seed).fit(Xtr)
    iso_te = iso.score(Xte)
    results["isoforest"] = ranking_metrics(yte, iso_te)
    results["isoforest"]["fit_seconds"] = round(time.time() - t0, 1)
    models["isoforest"] = iso

    # ---- 3. xgboost, supervised only -------------------------------------
    t0 = time.time()
    xgb_plain = fit_xgboost(Xtr, ytr, Xva, yva, seed=args.seed)
    p_plain = _scores(xgb_plain, Xte)
    results["xgboost"] = ranking_metrics(yte, p_plain)
    results["xgboost"]["fit_seconds"] = round(time.time() - t0, 1)
    models["xgboost"] = xgb_plain

    # ---- 4. xgboost + anomaly score --------------------------------------
    t0 = time.time()
    Xtr_a = with_anomaly(Xtr, iso.score(Xtr))
    Xva_a = with_anomaly(Xva, iso.score(Xva))
    Xte_a = with_anomaly(Xte, iso_te)
    xgb_hybrid = fit_xgboost(Xtr_a, ytr, Xva_a, yva, seed=args.seed)
    p_hybrid = _scores(xgb_hybrid, Xte_a)
    results["xgboost+iso"] = ranking_metrics(yte, p_hybrid)
    results["xgboost+iso"]["fit_seconds"] = round(time.time() - t0, 1)
    models["xgboost+iso"] = xgb_hybrid

    # ---- the ablation ----------------------------------------------------
    delta = results["xgboost+iso"]["pr_auc"] - results["xgboost"]["pr_auc"]
    ablation = {
        "pr_auc_xgboost": results["xgboost"]["pr_auc"],
        "pr_auc_xgboost_plus_iso": results["xgboost+iso"]["pr_auc"],
        "delta": delta,
        "relative_change_pct": round(100 * delta / max(results["xgboost"]["pr_auc"], 1e-9), 2),
        "anomaly_score_importance": None,
        "verdict": None,
    }
    imp = dict(zip(Xtr_a.columns, xgb_hybrid.feature_importances_))
    ablation["anomaly_score_importance"] = round(float(imp.get("anomaly_score", 0.0)), 4)
    ablation["verdict"] = (
        "anomaly score helps" if delta > 0.01 else
        "anomaly score does not measurably help" if abs(delta) <= 0.01 else
        "anomaly score hurts"
    )

    # ---- pick the best and tune its threshold on cost ---------------------
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_scores = {"baseline": _scores(base, Xte), "isoforest": iso_te,
                   "xgboost": p_plain, "xgboost+iso": p_hybrid}[best_name]

    curve = cost_curve(yte, best_scores, amounts_te)
    operating = pick_threshold(curve)
    curve.to_csv(os.path.join(args.reports, "cost_curve.csv"), index=False)
    pr_points(yte, best_scores).to_csv(
        os.path.join(args.reports, "pr_curve.csv"), index=False)

    importances = sorted(imp.items(), key=lambda kv: -kv[1])[:15]

    report = {
        "data": info,
        "split": s,
        "n_features": int(Xtr.shape[1]),
        "models": results,
        "ablation": ablation,
        "best_model": best_name,
        "operating_point": operating,
        "top_features": [{"feature": k, "importance": round(float(v), 4)}
                         for k, v in importances],
    }
    with open(os.path.join(args.reports, "metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    joblib.dump({"feature_builder": fb, "anomaly": iso,
                 "model": models[best_name], "model_name": best_name,
                 "threshold": operating["threshold"],
                 "columns": list(Xte_a.columns if "iso" in best_name else Xte.columns)},
                os.path.join(args.out, "pipeline.joblib"))

    # ---- print -----------------------------------------------------------
    print("\nTest-set ranking quality (base rate {:.4%}):".format(float(yte.mean())))
    print(f"  {'model':14} {'PR-AUC':>8} {'lift':>7} {'ROC-AUC':>8} {'P@100':>7} {'R@100':>7}")
    for name, m in results.items():
        print(f"  {name:14} {m['pr_auc']:>8.4f} {m['pr_auc_lift']:>6.0f}x "
              f"{m['roc_auc']:>8.4f} {m.get('precision@100', float('nan')):>7.3f} "
              f"{m.get('recall@100', float('nan')):>7.3f}")

    print(f"\nAblation: {ablation['verdict']} "
          f"(ΔPR-AUC {delta:+.4f}, {ablation['relative_change_pct']:+.1f}%, "
          f"importance {ablation['anomaly_score_importance']})")

    print(f"\nBest: {best_name}. Cost-optimal operating point:")
    print(f"  threshold {operating['threshold']:.4f} → flags "
          f"{operating['flagged_pct']:.2%} of transactions "
          f"({operating['reviews_per_1000_txns']} reviews per 1,000)")
    print(f"  precision {operating['precision']:.3f}  recall {operating['recall']:.3f}")
    print(f"  cost ${operating['total_cost']:,.0f} vs ${operating['cost_if_no_model']:,.0f} "
          f"with no model → ${operating['cost_avoided']:,.0f} avoided")
    if args.mlflow:
        _log_to_mlflow(args, report, results, models, best_name)

    print(f"\nWrote {args.reports}/metrics.json and {args.out}/pipeline.joblib")


def _metric_name(key: str) -> str:
    """MLflow allows alphanumerics, _ - . space : / only."""
    return key.replace("@", "_at_")


def _log_to_mlflow(args, report, results, models, best_name) -> None:
    """Record the run so results are reproducible and comparable.

    One parent run per training job, with a nested run per model. Nesting
    matters: the four models share a split and a feature set, so comparing
    them across separate top-level runs would lose the fact that they are one
    experiment. The ablation delta is logged on the parent, because it is a
    property of the comparison rather than of either model.
    """
    import mlflow

    # MLflow 3 put the filesystem backend into maintenance mode and refuses to
    # use it without an opt-out env var. sqlite needs no server, works on one
    # machine, and is what `mlflow ui` reads.
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"train-{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params({
            "rows": report["data"]["rows"],
            "fraud_rate": report["data"]["fraud_rate"],
            "n_features": report["n_features"],
            "embargo_days": args.embargo_days,
            "seed": args.seed,
            "nrows_cap": args.nrows or "all",
        })
        for name, m in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                # "precision@100" is not a legal MLflow metric name — @ is
                # rejected. Rename rather than drop: precision at a fixed
                # review-queue size is one of the more operationally useful
                # numbers here.
                mlflow.log_metrics({
                    _metric_name(k): float(v) for k, v in m.items()
                    if isinstance(v, (int, float)) and k != "n"
                })
        mlflow.log_metrics({
            "best_pr_auc": results[best_name]["pr_auc"],
            "ablation_delta": report["ablation"]["delta"],
            "operating_recall": report["operating_point"]["recall"],
            "operating_precision": report["operating_point"]["precision"],
            "cost_avoided": report["operating_point"]["cost_avoided"],
        })
        mlflow.set_tags({
            "best_model": best_name,
            "ablation_verdict": report["ablation"]["verdict"],
        })
        mlflow.log_artifact(os.path.join(args.reports, "metrics.json"))
        mlflow.log_artifact(os.path.join(args.out, "pipeline.joblib"))
        print(f"Logged to MLflow experiment '{args.experiment}'")


if __name__ == "__main__":
    main()
