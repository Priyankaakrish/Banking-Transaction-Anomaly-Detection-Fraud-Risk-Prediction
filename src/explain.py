"""SHAP explanations.

Two audiences, two outputs, and conflating them is the usual mistake.

**Global** — which features drive the model overall. A sanity check for the
modeller: if `merchant_risk` dominates, the target encoding is probably leaking;
if `hour` dominates, the model has latched onto a scheduling artefact rather
than fraud.

**Local** — why *this* transaction was flagged. A fraud analyst opening an alert
needs a reason they can act on: "amount is 40x this card's average and the
merchant is new" is investigable. A risk score of 0.87 is not. In several
jurisdictions an adverse decision also has to be explainable, so this is a
compliance requirement rather than a nicety.

Local explanations are rendered as short sentences, not SHAP values. An analyst
reviewing a hundred alerts a day will not read a force plot.

    python -m src.explain --artifacts artifacts --data data/sample_transactions.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

# How each feature reads in a sentence, and whether a high value is the
# suspicious direction. Anything absent falls back to the raw column name.
READABLE = {
    "amount_z_vs_card": ("amount vs this card's normal spend", "high"),
    "amount": ("transaction amount", "high"),
    "amount_abs": ("transaction amount", "high"),
    "log_amount": ("transaction amount", "high"),
    "txn_count_1h": ("transactions in the last hour", "high"),
    "txn_count_24h": ("transactions in the last 24 hours", "high"),
    "txn_count_7d": ("transactions in the last 7 days", "high"),
    "amount_sum_1h": ("amount spent in the last hour", "high"),
    "amount_sum_24h": ("amount spent in the last 24 hours", "high"),
    "amount_sum_7d": ("amount spent in the last 7 days", "high"),
    "seconds_since_prev": ("time since the previous transaction", "low"),
    "is_online": ("card-not-present (online)", "high"),
    "is_swipe": ("magnetic stripe used", "high"),
    "is_night": ("made overnight", "high"),
    "is_weekend": ("made at a weekend", "high"),
    "has_error": ("an error was recorded (e.g. bad PIN)", "high"),
    "new_merchant_for_card": ("first time this card has used this merchant", "high"),
    "new_state_for_card": ("first transaction in this state", "high"),
    "state_changed": ("different state from the previous transaction", "high"),
    "merchant_risk": ("this merchant's historical fraud rate", "high"),
    "mcc_risk": ("this merchant category's fraud rate", "high"),
    "anomaly_score": ("overall unusualness vs normal traffic", "high"),
    "card_amount_mean_prior": ("this card's usual spend", "low"),
    "is_refund": ("a refund", "high"),
}


def global_importance(model, X: pd.DataFrame, sample: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Mean |SHAP| per feature.

    Sampled because SHAP over millions of rows is slow and adds nothing — the
    ranking stabilises quickly.
    """
    import shap

    X = X.sample(min(sample, len(X)), random_state=seed)
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):          # older API returns one array per class
        values = values[1]
    return (pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def explain_row(model, X: pd.DataFrame, i: int, top_k: int = 4) -> list[str]:
    """Plain-language reasons this row scored as it did.

    Only features pushing *towards* fraud are returned: an analyst wants the
    case for investigating, not a balanced ledger. Ordered by contribution.
    """
    import shap

    row = X.iloc[[i]]
    values = shap.TreeExplainer(model).shap_values(row)
    if isinstance(values, list):
        values = values[1]
    contrib = pd.Series(values[0], index=X.columns).sort_values(ascending=False)

    reasons: list[str] = []
    # Several columns share a label (amount / amount_abs / log_amount all read
    # as "transaction amount"), so dedupe on the label rather than the column —
    # repeating the same reason wastes an analyst's attention.
    seen_labels: set[str] = set()
    for feature, v in contrib.head(top_k * 3).items():
        if v <= 0:
            break
        label, suspicious_high = READABLE.get(feature, (feature.replace("_", " "), "high"))
        if label in seen_labels:
            continue
        seen_labels.add(label)
        value = row[feature].iloc[0]
        if feature.startswith("is_") or feature.startswith("new_") or feature.startswith("has_") \
                or feature == "state_changed":
            if value >= 0.5:
                reasons.append(label)
        elif feature == "amount_z_vs_card":
            reasons.append(f"{label} ({value:.1f} standard deviations above)")
        elif feature == "seconds_since_prev" and 0 <= value < 300:
            reasons.append(f"{label} was {int(value)}s")
        else:
            reasons.append(f"{label} ({value:,.2f})")
        if len(reasons) >= top_k:
            break
    return reasons or ["no single feature dominates; the score comes from a combination"]


def main() -> None:
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", required=True)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    from .features import build
    from .models import with_anomaly
    from .schema import load
    from .split import temporal_split

    os.makedirs(args.reports, exist_ok=True)
    bundle = joblib.load(os.path.join(args.artifacts, "pipeline.joblib"))
    model = bundle["model"]

    if not hasattr(model, "get_booster"):
        raise SystemExit(f"SHAP TreeExplainer needs a tree model; got {bundle['model_name']}. "
                         "Retrain so xgboost wins, or use KernelExplainer (much slower).")

    df = load(args.data, nrows=args.nrows)
    split = temporal_split(df)
    _, (_, _, Xte) = build(split.train, split.valid, split.test)
    if "anomaly_score" in bundle["columns"]:
        Xte = with_anomaly(Xte, bundle["anomaly"].score(Xte))
    Xte = Xte[bundle["columns"]]

    # ---- global -----------------------------------------------------------
    imp = global_importance(model, Xte, sample=args.sample)
    imp.to_csv(os.path.join(args.reports, "shap_global.csv"), index=False)
    print("Global drivers (mean |SHAP|):")
    for _, r in imp.head(10).iterrows():
        label = READABLE.get(r["feature"], (r["feature"], ""))[0]
        print(f"  {r['mean_abs_shap']:.4f}  {r['feature']:<24} {label}")

    # ---- local: the riskiest transactions ---------------------------------
    scores = model.predict_proba(Xte)[:, 1]
    top = np.argsort(scores)[::-1][: args.examples]
    y = split.test["is_fraud"].to_numpy()

    print(f"\nHighest-risk transactions in the test period:")
    cases = []
    for i in top:
        reasons = explain_row(model, Xte, int(i))
        amount = split.test["amount"].iloc[int(i)]
        case = {
            "risk_score": round(float(scores[i]), 4),
            "actually_fraud": bool(y[i]),
            "amount": float(amount),
            "timestamp": str(split.test["timestamp"].iloc[int(i)]),
            "reasons": reasons,
        }
        cases.append(case)
        mark = "FRAUD" if y[i] else "legit"
        print(f"\n  score {case['risk_score']:.3f}  ${amount:,.2f}  [{mark}]")
        for r in reasons:
            print(f"    - {r}")

    with open(os.path.join(args.reports, "shap_cases.json"), "w") as fh:
        json.dump(cases, fh, indent=2)
    print(f"\nWrote {args.reports}/shap_global.csv and shap_cases.json")


if __name__ == "__main__":
    main()
