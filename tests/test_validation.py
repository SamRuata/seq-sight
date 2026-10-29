"""The request-validation layer: bad input must fail at the edge, with a 422.

Without this a 200,000-character paste reaches the tokenizer and an empty
string reaches the embedding, and the failure surfaces as a 500 from somewhere
deep in the model.
"""
from __future__ import annotations

from shared.schemas import MAX_CHARS
from tests.conftest import needs_models


@needs_models
def test_empty_text_is_rejected(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


@needs_models
def test_whitespace_only_text_is_rejected(client):
    assert client.post("/predict", json={"text": "   \n\t "}).status_code == 422


@needs_models
def test_overlong_text_is_rejected(client):
    r = client.post("/predict", json={"text": "a " * MAX_CHARS})
    assert r.status_code == 422


@needs_models
def test_wrong_dtype_is_rejected(client):
    assert client.post("/predict", json={"text": 12345}).status_code == 422
    assert client.post("/predict", json={"text": ["a", "b"]}).status_code == 422


@needs_models
def test_missing_field_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


@needs_models
def test_unparseable_text_is_rejected_not_crashed(client):
    """Punctuation only: valid as a string, but tokenizes to nothing."""
    r = client.post("/predict", json={"text": "*** ... ***"})
    assert r.status_code == 422
    assert "token" in str(r.json()["detail"]).lower()


@needs_models
def test_unknown_model_name_is_a_404(client):
    r = client.post("/predict", json={"text": "good film", "model": "nope"})
    assert r.status_code == 404


@needs_models
def test_batch_rejects_an_empty_member(client):
    r = client.post("/predict_batch", json={"texts": ["fine", ""]})
    assert r.status_code == 422


@needs_models
def test_batch_size_is_capped(client):
    r = client.post("/predict_batch", json={"texts": ["ok"] * 100})
    assert r.status_code == 422