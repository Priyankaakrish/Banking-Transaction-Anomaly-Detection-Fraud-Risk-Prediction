"""Join the user and card reference files.

The dataset ships three files. `credit_card_transactions-ibm_v2.csv` is the
event log; `sd254_users.csv` and `sd254_cards.csv` describe the people and the
plastic. The transaction file alone cannot answer *is this amount large for
this customer's credit limit?*, which is one of the stronger fraud signals
available, so the join is worth doing.

    sd254_users.csv   FICO, yearly income, total debt, age, num credit cards
    sd254_cards.csv   credit limit, card brand/type, has chip, account open date

**Two fields here are traps, and both are excluded by default.**

`Card on Dark Web` looks like the best feature in the dataset. It is a
present-day status flag, and a card is listed on the dark web *because* it was
compromised — often recorded after the fraud it would be used to predict. There
is no timestamp on it, so there is no way to establish it was known beforehand.
Including it would produce an excellent offline score and a model that cannot
work in production.

`Current Age` is the person's age now, not their age at transaction time. In a
dataset spanning years that is a subtle time leak; `Birth Year` is used instead
and age is computed per transaction.

Pass `include_unsafe=True` to include them anyway — the ablation is interesting,
and seeing how much the dark-web flag inflates PR-AUC is a good illustration of
what leakage looks like.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

MONEY_COLUMNS = (
    "Per Capita Income - Zipcode", "Yearly Income - Person",
    "Total Debt", "Credit Limit",
)

# Excluded unless explicitly requested. See the module docstring.
UNSAFE_COLUMNS = {"Card on Dark Web", "Current Age"}


def _money(series: pd.Series) -> pd.Series:
    """'$59696' -> 59696.0"""
    return pd.to_numeric(
        series.astype("string").str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def load_users(path: str) -> pd.DataFrame:
    """One row per user. The row index is the User id in the transaction file."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "user": np.arange(len(df), dtype="int32"),
        "birth_year": pd.to_numeric(df.get("Birth Year"), errors="coerce"),
        "gender": df.get("Gender", pd.Series(["unknown"] * len(df))).astype("category"),
        "fico": pd.to_numeric(df.get("FICO Score"), errors="coerce"),
        "num_cards": pd.to_numeric(df.get("Num Credit Cards"), errors="coerce"),
        "yearly_income": _money(df["Yearly Income - Person"])
                         if "Yearly Income - Person" in df else np.nan,
        "total_debt": _money(df["Total Debt"]) if "Total Debt" in df else np.nan,
        "zip_income": _money(df["Per Capita Income - Zipcode"])
                      if "Per Capita Income - Zipcode" in df else np.nan,
    })
    return out


def load_cards(path: str, include_unsafe: bool = False) -> pd.DataFrame:
    """One row per (user, card)."""
    df = pd.read_csv(df_path := path)
    out = pd.DataFrame({
        "user": pd.to_numeric(df["User"], errors="coerce").astype("int32"),
        "card": pd.to_numeric(df["CARD INDEX"], errors="coerce").astype("int16"),
        "card_brand": df.get("Card Brand", "unknown").astype("category"),
        "card_type": df.get("Card Type", "unknown").astype("category"),
        "has_chip": (df.get("Has Chip", "NO").astype(str).str.upper() == "YES").astype("int8"),
        "cards_issued": pd.to_numeric(df.get("Cards Issued"), errors="coerce"),
        "credit_limit": _money(df["Credit Limit"]) if "Credit Limit" in df else np.nan,
        "acct_open": pd.to_datetime(df.get("Acct Open Date"), format="%m/%Y", errors="coerce"),
    })
    if include_unsafe and "Card on Dark Web" in df.columns:
        out["card_on_dark_web"] = (
            df["Card on Dark Web"].astype(str).str.upper() == "YES").astype("int8")
    return out


def enrich(
    transactions: pd.DataFrame,
    users_path: str | None = None,
    cards_path: str | None = None,
    include_unsafe: bool = False,
) -> pd.DataFrame:
    """Left-join reference data and derive the features it makes possible.

    A left join, deliberately: a transaction whose card is missing from the
    reference file must survive with NaNs rather than vanish. Silently dropping
    rows would change the base rate and the evaluation along with it.
    """
    df = transactions.copy()

    if cards_path and os.path.exists(cards_path):
        cards = load_cards(cards_path, include_unsafe=include_unsafe)
        before = len(df)
        df = df.merge(cards, on=["user", "card"], how="left")
        assert len(df) == before, "card join duplicated rows — check (user, card) uniqueness"

        if "credit_limit" in df:
            # The headline feature: a $900 charge is unremarkable on a $20,000
            # limit and extraordinary on a $1,000 one.
            df["amount_vs_limit"] = (df["amount"].abs()
                                     / df["credit_limit"].replace(0, np.nan))
            df["credit_limit_log"] = np.log1p(df["credit_limit"])
        if "acct_open" in df:
            # Age of the account at the time of the transaction, not today.
            df["card_age_days"] = (df["timestamp"] - df["acct_open"]).dt.days
            df["card_is_new"] = (df["card_age_days"] < 90).astype("int8")

    if users_path and os.path.exists(users_path):
        users = load_users(users_path)
        before = len(df)
        df = df.merge(users, on="user", how="left")
        assert len(df) == before, "user join duplicated rows"

        if "birth_year" in df:
            # Computed per transaction rather than taken from 'Current Age',
            # which is as-of-today and leaks time in a multi-year dataset.
            df["age_at_txn"] = df["timestamp"].dt.year - df["birth_year"]
        if {"total_debt", "yearly_income"} <= set(df.columns):
            df["debt_to_income"] = (df["total_debt"]
                                    / df["yearly_income"].replace(0, np.nan))
        if "yearly_income" in df:
            df["amount_vs_income"] = (df["amount"].abs()
                                      / df["yearly_income"].replace(0, np.nan))

    return df


ENRICHED_FEATURES = [
    "amount_vs_limit", "credit_limit_log", "card_age_days", "card_is_new",
    "has_chip", "cards_issued", "fico", "num_cards", "age_at_txn",
    "debt_to_income", "amount_vs_income", "zip_income",
]


def enriched_columns(df: pd.DataFrame, include_unsafe: bool = False) -> list[str]:
    """Which derived columns are present and safe to feed the model."""
    cols = [c for c in ENRICHED_FEATURES if c in df.columns]
    if include_unsafe and "card_on_dark_web" in df.columns:
        cols.append("card_on_dark_web")
    return cols


def coverage(df: pd.DataFrame) -> dict:
    """How much of the join actually landed.

    A join that silently matches 3% of rows produces features that are almost
    all NaN, which a tree model will happily ignore while you wonder why the
    new features did nothing.
    """
    out = {"rows": int(len(df))}
    for col in ENRICHED_FEATURES:
        if col in df.columns:
            out[f"{col}_coverage"] = round(float(df[col].notna().mean()), 4)
    return out
