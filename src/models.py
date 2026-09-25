"""Models, in the order they earn their place.

The point of keeping all three is the comparison. A gradient-boosted model that
beats nothing is not evidence; a gradient-boosted model that beats a regularised
linear baseline by a stated margin is.

**Baseline** — logistic regression on scaled features. Fast, hard to get wrong,
and it sets the bar. If XGBoost cannot clear it meaningfully, the extra
complexity is not paying for itself.

**Isolation Forest** — unsupervised, fitted without labels. Its score becomes a
*candidate* feature for the supervised model, not a predictor in its own right.
Whether it helps is an open question: when labels exist, an anomaly score often
adds little, because "unusual" and "fraudulent" are different properties. Most
legitimate outliers are just a customer buying a fridge. `train.py` runs the
ablation rather than assuming.

**XGBoost** — handles the tabular mix, the imbalance (via `scale_pos_weight`)
and the non-linear interactions that matter here, such as high amount *and* new
merchant *and* night-time.

One deliberate omission: no SMOTE or random oversampling. Synthesising minority
rows by interpolating between fraud cases invents transactions that never
happened, and with causal time-series features it can interpolate across the
time boundary. `scale_pos_weight` reweights the loss without fabricating data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------- baseline ---
def fit_baseline(X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    """Regularised logistic regression with balanced class weights."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5,
            solver="lbfgs", random_state=seed,
        ),
    )
    return model.fit(X, y)


# ------------------------------------------------------ anomaly scoring ---
@dataclass
class AnomalyScorer:
    """Isolation Forest wrapper producing a single feature.

    Fitted on the training split only — including the evaluation period would
    let the contamination estimate and the tree structure absorb information
    from the future, which is the same leak the temporal split exists to stop.

    Fitted on *all* training rows rather than legitimate ones only. Filtering to
    negatives would need the labels, which turns an unsupervised model into a
    weakly supervised one and quietly changes what the ablation is measuring.
    """

    n_estimators: int = 200
    max_samples: int = 50_000
    contamination: float | str = "auto"
    seed: int = 42
    model_: IsolationForest | None = None

    def fit(self, X: pd.DataFrame) -> "AnomalyScorer":
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(X)),
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        ).fit(X)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher means more anomalous.

        sklearn's `score_samples` returns higher-is-more-normal, which is the
        opposite of the intuition and an easy sign error to carry into a
        feature. Negated here once, at the boundary.
        """
        if self.model_ is None:
            raise RuntimeError("AnomalyScorer.fit must be called first")
        return -self.model_.score_samples(X)


# -------------------------------------------------------------- xgboost ---
def fit_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    seed: int = 42,
    **overrides,
):
    """Gradient boosting tuned for a rare positive class.

    `aucpr` is the evaluation metric because with a ~0.1% positive rate the
    ROC curve is dominated by the negatives and barely moves; early stopping on
    it would stop on noise.
    """
    import xgboost as xgb

    pos = max(1, int((y == 1).sum()))
    neg = int((y == 0).sum())

    params = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    params.update(overrides)

    fit_kwargs = {}
    if X_valid is not None and y_valid is not None and len(np.unique(y_valid)) > 1:
        params["early_stopping_rounds"] = 50
        fit_kwargs = {"eval_set": [(X_valid, y_valid)], "verbose": False}

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, **fit_kwargs)
    return model


def with_anomaly(X: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Attach the anomaly score as one more column, leaving the original intact."""
    out = X.copy()
    out["anomaly_score"] = scores.astype("float32")
    return out
