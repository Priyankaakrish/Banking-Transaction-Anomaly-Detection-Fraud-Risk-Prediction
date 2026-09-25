"""API tests.

The interesting behaviour is not "does it return a number" but whether the
service is honest about how much history informed that number. A score
computed with no card history rests on transaction-level features alone and
should not be presented as equivalent to a fully-informed one.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from serve.api import app  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "pipeline.joblib")

BASE = {"user": 0, "card": 0, "use_chip": "Swipe Transaction", "merchant": "111222333",
        "merchant_city": "City1", "merchant_state": "CA", "zip": "90210",
        "mcc": 5411, "errors": ""}


@pytest.fixture(scope="module")
def client():
    if not os.path.exists(ART):
        pytest.skip("train a model first: python -m src.train")
    c = TestClient(app)
    c.delete("/cache")
    return c


def test_health_reports_the_loaded_model(client):
    h = client.get("/health").json()
    assert h["ok"] is True
    assert "threshold" in h and h["model"]


def test_cold_start_is_flagged_as_low_confidence(client):
    """A score with no history must say so rather than look authoritative."""
    r = client.post("/predict", json={"transaction": {
        **BASE, "user": 777, "timestamp": "2019-03-01T10:00:00", "amount": 45.0}})
    d = r.json()
    assert d["history_depth"] == 0
    assert d["history_source"] == "none"
    assert d["confidence_note"] and "No prior transactions" in d["confidence_note"]


def test_anomalous_transaction_scores_above_a_normal_one(client):
    history = [{**BASE, "timestamp": f"2019-03-0{i}T09:00:00", "amount": 40.0 + i}
               for i in range(1, 8)]
    odd = client.post("/predict", json={
        "transaction": {**BASE, "timestamp": "2019-03-09T02:00:00", "amount": 1800.0,
                        "use_chip": "Online Transaction", "merchant_state": "NY"},
        "history": history}).json()
    normal = client.post("/predict", json={
        "transaction": {**BASE, "timestamp": "2019-03-09T13:00:00", "amount": 44.0},
        "history": history}).json()
    assert odd["risk_score"] > normal["risk_score"]
    assert odd["history_depth"] == 7 and odd["history_source"] == "request"


def test_cache_accumulates_history_across_calls(client):
    client.delete("/cache")
    for i in range(1, 6):
        client.post("/predict", json={"transaction": {
            **BASE, "user": 9, "timestamp": f"2019-04-0{i}T10:00:00", "amount": 50.0}})
    d = client.post("/predict", json={"transaction": {
        **BASE, "user": 9, "timestamp": "2019-04-06T10:00:00", "amount": 50.0}}).json()
    assert d["history_depth"] == 5
    assert d["history_source"] == "cache"


def test_supplied_history_takes_priority_over_cache(client):
    hist = [{**BASE, "user": 9, "timestamp": f"2019-04-1{i}T10:00:00", "amount": 60.0}
            for i in range(1, 4)]
    d = client.post("/predict", json={
        "transaction": {**BASE, "user": 9, "timestamp": "2019-04-20T10:00:00",
                        "amount": 60.0},
        "history": hist}).json()
    assert d["history_source"] == "request" and d["history_depth"] == 3


def test_decision_follows_the_threshold(client):
    d = client.post("/predict", json={"transaction": {
        **BASE, "user": 55, "timestamp": "2019-06-01T10:00:00", "amount": 30.0}}).json()
    expected = "review" if d["risk_score"] >= d["threshold"] else "approve"
    assert d["decision"] == expected
    assert d["risk_band"] in {"low", "medium", "high"}


def test_batch_scores_every_row(client):
    batch = [{**BASE, "user": 31, "timestamp": f"2019-05-{i:02d}T10:00:00",
              "amount": 50.0 if i < 10 else 2500.0} for i in range(1, 13)]
    b = client.post("/predict/batch", json=batch).json()
    assert b["n"] == 12 and len(b["scores"]) == 12
    # the large charges arrive last and must not score below the small ones
    small = max(s["risk_score"] for s in b["scores"][:9])
    large = max(s["risk_score"] for s in b["scores"][9:])
    assert large >= small


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json=[]).status_code == 400


def test_malformed_payload_is_rejected(client):
    r = client.post("/predict", json={"transaction": {"amount": 10}})
    assert r.status_code == 422        # pydantic validation, not a 500
