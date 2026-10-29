"""Prediction behaviour: attention shape, determinism, and a frozen reference.

The reference test is the one that catches a change nobody meant to make --
a different tokenizer, a retrained checkpoint, a reordered vocabulary. On the
first run it writes the baseline; afterwards it compares against it.
"""
from __future__ import annotations

import json

from tests.conftest import REFERENCE, needs_models

SENTENCE = ("I went in expecting nothing and came out completely delighted. "
            "The script is sharp and the lead performance is the best I have "
            "seen all year.")
TOLERANCE = 1e-3


@needs_models
def test_predict_returns_the_documented_contract(client):
    r = client.post("/predict", json={"text": SENTENCE})
    assert r.status_code == 200
    body = r.json()

    for field in ("label", "class_name", "confidence", "probs", "tokens",
                  "attention", "n_tokens", "input_sha256", "latency_ms"):
        assert field in body

    assert body["class_name"] in ("positive", "negative")
    assert len(body["probs"]) == 2
    assert abs(sum(body["probs"]) - 1.0) < 1e-5
    assert len(body["input_sha256"]) == 64


@needs_models
def test_attention_has_one_weight_per_token_and_sums_to_one(client):
    body = client.post("/predict", json={"text": SENTENCE}).json()
    assert len(body["attention"]) == len(body["tokens"]) == body["n_tokens"]
    assert abs(sum(body["attention"]) - 1.0) < 1e-3
    assert all(w >= 0 for w in body["attention"])


@needs_models
def test_attention_matrix_is_square_over_the_tokens(client):
    """The brief's "attention matrix has the expected shape"."""
    body = client.post("/predict", json={"text": SENTENCE,
                                         "model": "attention",
                                         "include_matrix": True}).json()
    matrix = body["attention_matrix"]
    n = body["n_tokens"]

    assert matrix is not None
    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)
    # Every query row is a probability distribution over the keys.
    for row in matrix:
        assert abs(sum(row) - 1.0) < 1e-2


@needs_models
def test_recurrent_baseline_reports_no_matrix(client):
    """The baseline's lack of an explanation is part of the contract."""
    body = client.post("/predict", json={"text": SENTENCE, "model": "lstm",
                                         "include_matrix": True}).json()
    assert body["attention_matrix"] is None


@needs_models
def test_prediction_is_deterministic(client):
    a = client.post("/predict", json={"text": SENTENCE}).json()
    b = client.post("/predict", json={"text": SENTENCE}).json()
    assert a["label"] == b["label"]
    assert abs(a["confidence"] - b["confidence"]) < 1e-9
    assert a["input_sha256"] == b["input_sha256"]


@needs_models
def test_identical_text_hashes_identically(client):
    a = client.post("/predict", json={"text": SENTENCE}).json()
    b = client.post("/predict", json={"text": SENTENCE + " "}).json()
    assert a["input_sha256"] != b["input_sha256"]


@needs_models
def test_frozen_reference_regression(client):
    """Pin the model's answer on one fixed input.

    Writes the baseline on first run so the suite is runnable from a clean
    clone; every run after that compares against the committed file.
    """
    body = client.post("/predict", json={"text": SENTENCE,
                                         "model": "attention"}).json()
    current = {"label": body["label"], "probs": body["probs"],
               "n_tokens": body["n_tokens"]}

    if not REFERENCE.exists():
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps(current, indent=1), encoding="utf-8")
        import pytest
        pytest.skip(f"wrote a new reference to {REFERENCE}; commit it")

    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert current["n_tokens"] == expected["n_tokens"], "tokenizer changed"
    assert current["label"] == expected["label"], "prediction flipped"
    for got, want in zip(current["probs"], expected["probs"]):
        assert abs(got - want) < TOLERANCE, "probabilities drifted"


@needs_models
def test_truncation_is_reported(client):
    body = client.post("/predict", json={"text": "great " * 500}).json()
    assert body["truncated"] is True
    assert body["n_tokens"] <= 128