"""Ops endpoints and the service contract they advertise."""
from __future__ import annotations

from shared.text import CLASS_NAMES, MAX_LEN
from tests.conftest import needs_models


@needs_models
def test_healthz_reports_loaded_models(client):
    body = client.get("/healthz").json()
    assert body["status"] in ("ok", "degraded")
    assert body["vocab"] is True
    assert "attention" in body["models_loaded"]
    assert "lstm" in body["models_loaded"]


@needs_models
def test_classes_lists_the_real_labels(client):
    body = client.get("/classes").json()
    assert body["classes"] == CLASS_NAMES == ["negative", "positive"]
    assert body["n_classes"] == 2
    assert body["max_len"] == MAX_LEN
    assert body["vocab_size"] > 1000


@needs_models
def test_version_reports_the_torch_build(client):
    body = client.get("/version").json()
    assert body["torch_version"].startswith("2.4.1")
    assert body["served_model"] in ("attention", "lstm")


@needs_models
def test_models_endpoint_ranks_by_accuracy(client):
    body = client.get("/models").json()
    assert len(body) >= 2
    accuracies = [m["test_accuracy"] for m in body]
    assert accuracies == sorted(accuracies, reverse=True)
    # Only the attention model claims to be explainable.
    assert {m["model_type"]: m["explainable"] for m in body} == {
        "attention": True, "lstm": False}


@needs_models
def test_scaling_demo_matches_the_derivation(client):
    body = client.get("/scaling_demo", params={"d_k": 128}).json()
    assert 8.0 < body["unscaled_std"] < 15.0
    assert body["scaled_std"] < 1.5
    assert body["scaled_entropy"] > body["unscaled_entropy"]