"""Load and normalise the credit-card transactions dataset.

Source: kaggle.com/datasets/ealtman2019/credit-card-transactions
(~24M rows). Columns arrive as:

    User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
    Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?

Two things need fixing before anything else can be trusted:

1. **Amount is a currency string** ("$134.09", sometimes negative for refunds),
   not a number. Silently coercing it with `errors="coerce"` would turn every
   refund into NaN, so it is parsed explicitly and refunds are kept — a refund
   immediately after a charge is itself a fraud signal.

2. **There is no single timestamp.** Year/Month/Day/Time are separate columns.
   Every causal feature and the train/test split depend on a correct ordering,
   so the timestamp is built once here and the frame is sorted by it.

The label is `Is Fraud?`, not chargeback. They overlap but differ: friendly
fraud produces a chargeback with no fraud flag, and fraud caught before
settlement often never becomes one. Calling this fraud prediction is the
accurate description.
"""
from __future__ import annotations

import pandas as pd

RAW_COLUMNS = {
    "user": "User",
    "card": "Card",
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "time": "Time",
    "amount": "Amount",
    "use_chip": "Use Chip",
    "merchant": "Merchant Name",
    "merchant_city": "Merchant City",
    "merchant_state": "Merchant State",
    "zip": "Zip",
    "mcc": "MCC",
    "errors": "Errors?",
    "label": "Is Fraud?",
}

# Read as strings where pandas would otherwise guess badly: merchant ids are
# large integers that become floats, zips lose leading zeros.
DTYPES = {
    "Merchant Name": "string",
    "Zip": "string",
    "Merchant State": "string",
    "Merchant City": "string",
    "Errors?": "string",
    "Use Chip": "string",
    "Is Fraud?": "string",
    "Amount": "string",
}


def parse_amount(series: pd.Series) -> pd.Series:
    """'$134.09' -> 134.09, '$-77.00' -> -77.00.

    Refunds are negative and are kept, not dropped: a refund following a
    charge within minutes is a recognised fraud pattern.
    """
    cleaned = (
        series.astype("string")
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
    )
    out = pd.to_numeric(cleaned, errors="coerce")
    if out.isna().any():
        bad = series[out.isna()].head(3).tolist()
        raise ValueError(f"Could not parse {int(out.isna().sum())} amounts, e.g. {bad}")
    return out.astype("float64")


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Combine Year/Month/Day/Time into one datetime column."""
    stamp = (
        df["Year"].astype(str).str.zfill(4) + "-"
        + df["Month"].astype(str).str.zfill(2) + "-"
        + df["Day"].astype(str).str.zfill(2) + " "
        + df["Time"].astype(str)
    )
    ts = pd.to_datetime(stamp, format="%Y-%m-%d %H:%M", errors="coerce")
    if ts.isna().any():
        bad = stamp[ts.isna()].head(3).tolist()
        raise ValueError(f"{int(ts.isna().sum())} timestamps failed to parse, e.g. {bad}")
    return ts


def load(path: str, nrows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read the CSV, normalise types, and sort by time.

    Sorting here rather than in the feature code means every downstream step
    can assume chronological order, which is what makes the causal rolling
    features correct by construction.
    """
    df = pd.read_csv(path, dtype=DTYPES, nrows=nrows, usecols=usecols)
    missing = [c for c in RAW_COLUMNS.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "user": df["User"].astype("int32"),
        "card": df["Card"].astype("int16"),
        "timestamp": build_timestamp(df),
        "amount": parse_amount(df["Amount"]),
        "use_chip": df["Use Chip"].fillna("unknown").astype("category"),
        "merchant": df["Merchant Name"].fillna("unknown"),
        "merchant_city": df["Merchant City"].fillna("unknown"),
        "merchant_state": df["Merchant State"].fillna("unknown"),
        "zip": df["Zip"].fillna("unknown"),
        "mcc": df["MCC"].astype("int32"),
        "errors": df["Errors?"].fillna(""),
        "is_fraud": (df["Is Fraud?"].str.strip().str.lower() == "yes").astype("int8"),
    })

    # A card is identified by (user, card) — card numbers repeat across users.
    out["card_id"] = out["user"].astype(str) + "-" + out["card"].astype(str)

    out = out.sort_values(["timestamp", "user", "card"], kind="mergesort").reset_index(drop=True)
    return out


def summarise(df: pd.DataFrame) -> dict:
    fraud = int(df["is_fraud"].sum())
    return {
        "rows": int(len(df)),
        "cards": int(df["card_id"].nunique()),
        "users": int(df["user"].nunique()),
        "merchants": int(df["merchant"].nunique()),
        "from": str(df["timestamp"].min()),
        "to": str(df["timestamp"].max()),
        "fraud_rows": fraud,
        "fraud_rate": round(fraud / max(1, len(df)), 6),
        "refunds": int((df["amount"] < 0).sum()),
    }
