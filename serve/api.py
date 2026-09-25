"""Real-time scoring API.

**The hard part is not the model, it is the history.**

Every useful feature in this project is causal: transactions in the last hour,
amount relative to the card's past average, whether this merchant is new for
the card. None of that can be computed from a single incoming transaction.
A stateless `POST /predict` that receives one JSON object and returns a score
is therefore either lying or scoring on a crippled feature set.

Three honest ways to resolve it:

1. **Caller supplies the history.** The payment platform usually already has
   the card's recent activity in hand. Accepting it makes the dependency
   explicit and keeps the service stateless.
2. **Service keeps a recent-transaction cache.** What is implemented here, in
   memory: the last N transactions per card, so consecutive calls accumulate
   real history. Production would use Redis or DynamoDB with a TTL — an
   in-process dict does not survive a restart and does not share across
   replicas, which is stated rather than hidden.
3. **Score in batches.** Highest quality features, but not real-time.

The API supports 1 and 2 together: pass `history` if you have it, otherwise the
service uses what it has cached, and the response reports how many prior
transactions actually informed the score so the caller can judge it.

    uvicorn serve.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from collections import defaultdict, deque
from datetime import datetime
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT = os.environ.get("MODEL_PATH", "artifacts/pipeline.joblib")
HISTORY_LEN = int(os.environ.get("HISTORY_LEN", "50"))

app = FastAPI(
    title="Fraud scoring API",
    description="Transaction risk scoring. Features are causal, so scores "
                "improve as the service accumulates card history.",
    version="1.0.0",
)

_bundle: dict | None = None
# card_id -> recent transactions. In-process only: see the module docstring.
_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))


# ------------------------------------------------------------- schemas ----
class Transaction(BaseModel):
    user: int
    card: int
    timestamp: datetime
    amount: float = Field(..., description="Negative for refunds")
    use_chip: str = "Swipe Transaction"
    merchant: str
    merchant_city: str = "unknown"
    merchant_state: str = "unknown"
    zip: str = "unknown"
    mcc: int
    errors: str = ""


class ScoreRequest(BaseModel):
    transaction: Transaction
    history: list[Transaction] | None = Field(
        None,
        description="Recent transactions for the same card, oldest first. "
                    "Supply these when available; otherwise the service uses "
                    "its own cache, which may be empty after a restart.",
    )


class ScoreResponse(BaseModel):
    risk_score: float
    risk_band: Literal["low", "medium", "high"]
    decision: Literal["approve", "review"]
    threshold: float
    history_depth: int
    history_source: Literal["request", "cache", "none"]
    confidence_note: str | None = None
    model: str
    latency_ms: float


# --------------------------------------------------------------- model ----
def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not os.path.exists(ARTIFACT):
            raise RuntimeError(
                f"No model at {ARTIFACT}. Train one first: "
                f"python -m src.train --data <csv> --out artifacts"
            )
        _bundle = joblib.load(ARTIFACT)
    return _bundle


def _to_frame(rows: list[Transaction]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["card_id"] = df["user"].astype(str) + "-" + df["card"].astype(str)
    # The feature builder reads is_fraud for target encodings; at scoring time
    # the label is unknown, so zero it. Encodings for the scored row use only
    # prior rows, so this value never enters its own feature.
    df["is_fraud"] = 0
    return df.sort_values("timestamp").reset_index(drop=True)


@app.on_event("startup")
def startup() -> None:
    try:
        b = _load()
        print(f"Loaded {b['model_name']} | threshold {b['threshold']:.4f} "
              f"| {len(b['columns'])} features")
    except RuntimeError as exc:
        print(f"WARNING: {exc}")


# ------------------------------------------------------------ endpoints ---
@app.get("/health")
def health() -> dict:
    try:
        b = _load()
        return {"ok": True, "model": b["model_name"], "threshold": b["threshold"],
                "cards_cached": len(_history)}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/predict", response_model=ScoreResponse)
def predict(req: ScoreRequest) -> ScoreResponse:
    import time

    t0 = time.perf_counter()
    try:
        bundle = _load()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    txn = req.transaction
    card_id = f"{txn.user}-{txn.card}"

    if req.history:
        prior, source = list(req.history), "request"
    elif _history[card_id]:
        prior, source = list(_history[card_id]), "cache"
    else:
        prior, source = [], "none"

    frame = _to_frame([*prior, txn])
    features = bundle["feature_builder"].transform(frame)

    if "anomaly_score" in bundle["columns"]:
        features = features.copy()
        features["anomaly_score"] = bundle["anomaly"].score(features)

    X = features[bundle["columns"]].iloc[[-1]]      # only the incoming row
    score = float(bundle["model"].predict_proba(X)[0, 1])

    _history[card_id].append(txn)

    threshold = float(bundle["threshold"])
    band = "high" if score >= threshold else "medium" if score >= threshold * 0.5 else "low"

    # A score computed with no history rests on transaction-level features
    # alone. Saying so is more useful than returning a number that looks as
    # authoritative as a fully-informed one.
    note = None
    if len(prior) == 0:
        note = ("No prior transactions for this card — velocity and "
                "card-relative features are unavailable, so this score is "
                "less reliable than one computed with history.")
    elif len(prior) < 5:
        note = f"Only {len(prior)} prior transactions; card-relative features are thin."

    return ScoreResponse(
        risk_score=round(score, 6),
        risk_band=band,
        decision="review" if score >= threshold else "approve",
        threshold=round(threshold, 6),
        history_depth=len(prior),
        history_source=source,
        confidence_note=note,
        model=bundle["model_name"],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]) -> dict:
    """Score a chronological batch from one or more cards.

    Preferred over repeated `/predict` calls when scoring historical data:
    each row is scored with the full history preceding it in the batch, which
    is both faster and closer to how the model was trained.
    """
    import time

    t0 = time.perf_counter()
    try:
        bundle = _load()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not transactions:
        raise HTTPException(status_code=400, detail="empty batch")

    frame = _to_frame(transactions)
    features = bundle["feature_builder"].transform(frame)
    if "anomaly_score" in bundle["columns"]:
        features = features.copy()
        features["anomaly_score"] = bundle["anomaly"].score(features)

    scores = bundle["model"].predict_proba(features[bundle["columns"]])[:, 1]
    threshold = float(bundle["threshold"])
    return {
        "n": len(scores),
        "flagged": int((scores >= threshold).sum()),
        "threshold": round(threshold, 6),
        "scores": [
            {"index": i, "risk_score": round(float(s), 6),
             "decision": "review" if s >= threshold else "approve"}
            for i, s in enumerate(scores)
        ],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.delete("/cache")
def clear_cache() -> dict:
    n = len(_history)
    _history.clear()
    return {"cleared_cards": n}
