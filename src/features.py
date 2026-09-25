"""Feature engineering, computed causally.

Every feature here answers the question *what could a scoring system have known
at the moment this transaction arrived?* That constraint is what separates a
model that works in production from one that scores 0.99 offline and fails.

Three rules, applied everywhere:

1. **Shift before aggregating.** A rolling mean that includes the current row
   leaks the row into its own feature. Every window is computed on the series
   shifted by one within each card.
2. **Target encodings use past rows only.** A merchant's fraud rate computed
   over the whole dataset tells the model the answer. Here it is an expanding
   mean over prior rows, smoothed toward the global prior so a merchant's first
   few transactions do not get an extreme value.
3. **Fit statistics on train, apply to test.** Anything global — the fraud
   prior, category vocabularies — is learned in `fit()` and reused in
   `transform()`, never recomputed on the evaluation data.

`tests/test_leakage.py` verifies rule 1 empirically by recomputing features on
truncated data and checking the values are identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7D"}

# Denominator floor for the card-relative z-score. Without it, a card with
# perfectly constant spending has zero variance and its first unusual
# transaction scores as unremarkable — see the comment in transform().
STD_FLOOR_FRACTION = 0.10      # 10% of the card's prior mean
STD_FLOOR_ABSOLUTE = 1.0       # and never less than $1


def _past_window_stats(
    codes: np.ndarray, times: np.ndarray, values: np.ndarray, window_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Count and sum of prior rows within a time window, per group.

    `groupby.apply` with a Python lambda was the original implementation and it
    does not scale: cost grows with the number of groups, and at ~6,000 cards
    and 24M rows it projected to roughly three hours. This does the same work
    with a prefix sum and two binary searches per group — the inner operations
    are vectorised, so only the group loop is Python.

    Both outputs exclude the current row, which is what makes the feature
    causal: the window covers [t - window, t), open at the right.
    """
    n = len(codes)
    counts = np.zeros(n, dtype=np.float32)
    sums = np.zeros(n, dtype=np.float32)

    order = np.argsort(codes, kind="stable")
    grouped_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, grouped_codes[1:] != grouped_codes[:-1]])
    ends = np.r_[starts[1:], n]

    for a, b in zip(starts, ends):
        idx = order[a:b]                      # rows of one card, already time-sorted
        t = times[idx]
        v = values[idx]
        prefix = np.concatenate(([0.0], np.cumsum(v)))
        # first index whose timestamp is >= t - window
        left = np.searchsorted(t, t - window_ns, side="left")
        here = np.arange(len(idx))            # rows strictly before the current one
        counts[idx] = (here - left).astype(np.float32)
        sums[idx] = (prefix[here] - prefix[left]).astype(np.float32)
    return counts, sums


def _expanding_prior(codes: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of all prior rows in each group, vectorised.

    Computed from running sums of x and x^2 rather than pandas' expanding(),
    which allocates per group. The current row is subtracted out so the
    statistic describes the card's behaviour strictly before now.
    """
    n = len(codes)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)

    order = np.argsort(codes, kind="stable")
    gc = codes[order]
    starts = np.flatnonzero(np.r_[True, gc[1:] != gc[:-1]])
    ends = np.r_[starts[1:], n]

    for a, b in zip(starts, ends):
        idx = order[a:b]
        v = values[idx].astype(np.float64)
        k = np.arange(len(idx))               # number of prior rows
        s1 = np.concatenate(([0.0], np.cumsum(v)))[k]
        s2 = np.concatenate(([0.0], np.cumsum(v * v)))[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(k > 0, s1 / np.maximum(k, 1), np.nan)
            var = np.where(k > 1, (s2 - k * m * m) / np.maximum(k - 1, 1), np.nan)
        mean[idx] = m
        std[idx] = np.sqrt(np.maximum(var, 0))
    return mean, std


@dataclass
class FeatureBuilder:
    """Stateful so that train-fitted statistics can be reused at inference."""

    smoothing: float = 50.0          # prior weight for target encodings
    fraud_prior_: float = 0.0
    mcc_stats_: dict = field(default_factory=dict)
    fitted_: bool = False

    # ---------------------------------------------------------------- fit --
    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        """Learn only the global constants. Everything else is per-row causal."""
        self.fraud_prior_ = float(train["is_fraud"].mean())
        # MCC risk from the training period only, smoothed toward the prior.
        g = train.groupby("mcc")["is_fraud"].agg(["sum", "count"])
        self.mcc_stats_ = {
            int(k): float((r["sum"] + self.smoothing * self.fraud_prior_)
                          / (r["count"] + self.smoothing))
            for k, r in g.iterrows()
        }
        self.fitted_ = True
        return self

    # ---------------------------------------------------------- transform --
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeatureBuilder.fit must be called on the training split first")
        if not df["timestamp"].is_monotonic_increasing:
            df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        out = pd.DataFrame(index=df.index)

        # --- transaction-level (no history needed, so no leakage risk) -----
        out["amount"] = df["amount"]
        out["amount_abs"] = df["amount"].abs()
        out["is_refund"] = (df["amount"] < 0).astype("int8")
        out["log_amount"] = np.log1p(df["amount"].abs())
        out["hour"] = df["timestamp"].dt.hour.astype("int8")
        out["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
        out["is_night"] = df["timestamp"].dt.hour.isin([0, 1, 2, 3, 4, 5]).astype("int8")
        out["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype("int8")
        out["has_error"] = (df["errors"].fillna("").str.len() > 0).astype("int8")
        out["is_online"] = df["use_chip"].astype(str).str.contains("Online", case=False).astype("int8")
        out["is_swipe"] = df["use_chip"].astype(str).str.contains("Swipe", case=False).astype("int8")
        out["mcc_risk"] = df["mcc"].map(self.mcc_stats_).fillna(self.fraud_prior_)

        # --- per-card history: every window shifted by one row -------------
        by_card = df.groupby("card_id", sort=False, group_keys=False)

        out["seconds_since_prev"] = (
            by_card["timestamp"].diff().dt.total_seconds().fillna(-1)
        )
        out["card_txn_seq"] = by_card.cumcount().astype("int32")

        codes = df["card_id"].astype("category").cat.codes.to_numpy()
        times = df["timestamp"].to_numpy("datetime64[ns]").astype("int64")
        absamt = df["amount"].abs().to_numpy("float64")

        for name, win in WINDOWS.items():
            window_ns = int(pd.Timedelta(win).value)
            cnt, tot = _past_window_stats(codes, times, absamt, window_ns)
            out[f"txn_count_{name}"] = cnt
            out[f"amount_sum_{name}"] = tot

        # The card's behaviour strictly before this transaction.
        prev_mean, prev_std = _expanding_prior(codes, absamt)
        out["card_amount_mean_prior"] = np.nan_to_num(prev_mean)
        out["card_amount_std_prior"] = np.nan_to_num(prev_std)

        # Floor the denominator. A card with perfectly regular history — a
        # recurring $15.99 subscription, say — has zero variance, and dividing
        # by it would send the z-score to NaN and then to 0, marking the single
        # most unusual transaction on that card as perfectly normal. Flooring
        # at a fraction of the card's mean (and at $1 absolute, for very small
        # averages) keeps a large deviation reading as large.
        denom = np.maximum.reduce([
            np.nan_to_num(prev_std),
            STD_FLOOR_FRACTION * np.nan_to_num(prev_mean),
            np.full_like(absamt, STD_FLOOR_ABSOLUTE),
        ])
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (absamt - prev_mean) / denom
        # Rows with no history at all keep z = 0: there is nothing to compare to.
        z = np.where(np.isnan(prev_mean), 0.0, z)
        out["amount_z_vs_card"] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # --- novelty: has this card seen this merchant / state before? -----
        out["new_merchant_for_card"] = (
            ~df.duplicated(subset=["card_id", "merchant"], keep="first")
        ).astype("int8")
        out["new_state_for_card"] = (
            ~df.duplicated(subset=["card_id", "merchant_state"], keep="first")
        ).astype("int8")
        out["state_changed"] = (
            by_card["merchant_state"].shift(1).ne(df["merchant_state"])
            & by_card["merchant_state"].shift(1).notna()
        ).astype("int8")

        # --- merchant risk: expanding past-only mean, smoothed -------------
        out["merchant_risk"] = self._expanding_target_rate(df, "merchant")

        return out.astype("float32", errors="ignore")

    # ------------------------------------------------------------ helpers --
    def _expanding_target_rate(self, df: pd.DataFrame, key: str) -> pd.Series:
        """Smoothed fraud rate for `key`, using only rows strictly before each row.

        cumsum minus the current value gives the prior sum; the prior count is
        the running position within the group. Smoothing toward the global
        prior keeps a merchant's first transaction from reading 0.0 or 1.0.
        """
        g = df.groupby(key, sort=False)
        prior_sum = g["is_fraud"].cumsum() - df["is_fraud"]
        prior_count = g.cumcount()
        return ((prior_sum + self.smoothing * self.fraud_prior_)
                / (prior_count + self.smoothing)).astype("float32")


def build(train: pd.DataFrame, *others: pd.DataFrame, smoothing: float = 50.0):
    """Fit on train, transform train and every other split with the same statistics."""
    fb = FeatureBuilder(smoothing=smoothing).fit(train)
    return fb, [fb.transform(d) for d in (train, *others)]
