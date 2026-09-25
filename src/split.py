"""Time-based splitting.

A random split is wrong here, and wrong in a way that inflates scores rather
than lowering them — which is why it survives into so many fraud write-ups.

Two mechanisms leak under a random split:

* **Rolling features look forwards.** A test row placed before a training row
  from the same card means the model was fitted on aggregates that already
  contain the test transaction.
* **Fraud is bursty.** A compromised card produces a run of fraudulent
  transactions in minutes. Scatter those rows across train and test and the
  model can memorise the burst rather than learn the pattern. Recall looks
  excellent and collapses in production.

So: sort by time, cut by time, and leave an **embargo** gap between train and
test. The embargo matters because the longest rolling window (7 days) means a
training row within 7 days of the boundary has aggregates partially computed
from the test period. Dropping that window costs a little data and removes the
last path for information to cross the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict

    def summary(self) -> dict:
        def part(name, d):
            return {
                f"{name}_rows": int(len(d)),
                f"{name}_fraud": int(d["is_fraud"].sum()),
                f"{name}_fraud_rate": round(float(d["is_fraud"].mean()), 6) if len(d) else 0.0,
                f"{name}_from": str(d["timestamp"].min()) if len(d) else None,
                f"{name}_to": str(d["timestamp"].max()) if len(d) else None,
            }
        out = {}
        for n, d in (("train", self.train), ("valid", self.valid), ("test", self.test)):
            out.update(part(n, d))
        out["boundaries"] = self.boundaries
        return out


def temporal_split(
    df: pd.DataFrame,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    embargo_days: int = 7,
    time_col: str = "timestamp",
) -> Split:
    """Split chronologically into train / valid / test with an embargo gap.

    Fractions are of the *time-ordered rows*, not of the calendar span, so
    each part holds a comparable number of transactions even though volume
    grows over the years.

    The embargo is dropped from the END of the earlier part rather than the
    start of the later one. The later part must stay intact: it is the thing
    being measured, and removing its first week would quietly change what
    "test" means.
    """
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    n = len(df)
    train_end = int(n * (1 - valid_frac - test_frac))
    valid_end = int(n * (1 - test_frac))

    t_train_cut = df[time_col].iloc[train_end]
    t_valid_cut = df[time_col].iloc[valid_end]
    embargo = pd.Timedelta(days=embargo_days)

    train = df.iloc[:train_end]
    train = train[train[time_col] < t_train_cut - embargo]

    valid = df.iloc[train_end:valid_end]
    valid = valid[valid[time_col] < t_valid_cut - embargo]

    test = df.iloc[valid_end:]

    boundaries = {
        "train_cut": str(t_train_cut),
        "valid_cut": str(t_valid_cut),
        "embargo_days": embargo_days,
        "rows_dropped_to_embargo": int(n - len(train) - len(valid) - len(test)),
    }
    return Split(train.reset_index(drop=True), valid.reset_index(drop=True),
                 test.reset_index(drop=True), boundaries)


def assert_no_time_overlap(split: Split, time_col: str = "timestamp") -> None:
    """Hard check that the parts are ordered and disjoint in time.

    Cheap to run and worth running: a silently overlapping split produces
    results that look good and mean nothing.
    """
    for earlier, later, names in (
        (split.train, split.valid, ("train", "valid")),
        (split.valid, split.test, ("valid", "test")),
        (split.train, split.test, ("train", "test")),
    ):
        if len(earlier) == 0 or len(later) == 0:
            continue
        hi, lo = earlier[time_col].max(), later[time_col].min()
        if hi >= lo:
            raise AssertionError(
                f"{names[0]} ends at {hi} but {names[1]} starts at {lo} — the split overlaps"
            )


def rolling_origin_folds(
    df: pd.DataFrame, n_folds: int = 4, test_frac: float = 0.15,
    embargo_days: int = 7, time_col: str = "timestamp",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window backtest: train on everything up to t, test on what follows.

    A single split gives one number from one period. Fraud patterns shift —
    new attack methods, seasonal volume — so a model can look strong on one
    quarter and weak on the next. Several folds show whether performance is
    stable or whether you got lucky with the cut point.
    """
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    n = len(df)
    test_size = int(n * test_frac)
    embargo = pd.Timedelta(days=embargo_days)
    folds = []
    for k in range(n_folds, 0, -1):
        test_start = n - k * test_size
        test_end = test_start + test_size
        if test_start <= 0:
            continue
        cut = df[time_col].iloc[test_start]
        tr = df.iloc[:test_start]
        tr = tr[tr[time_col] < cut - embargo]
        te = df.iloc[test_start:test_end]
        if len(tr) and len(te):
            folds.append((tr.reset_index(drop=True), te.reset_index(drop=True)))
    return folds
