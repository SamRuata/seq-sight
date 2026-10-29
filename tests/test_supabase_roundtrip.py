"""Supabase round trip, the privacy invariant, and the outage path."""
from __future__ import annotations

import hashlib

from tests.conftest import needs_models, needs_supabase

TEXT = "A quietly excellent film that earns every minute of its runtime."


@needs_models
@needs_supabase
def test_predict_logs_a_row_that_joins_to_a_run(client, supabase):
    """The brief's requirement: a prediction row, joined to a runs row."""
    client.post("/predict", json={"text": TEXT})

    digest = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    rows = (supabase.table("predictions").select("*")
            .eq("input_hash", digest).order("id", desc=True)
            .limit(1).execute().data)
    assert rows, "no predictions row was written"
    row = rows[0]

    run = (supabase.table("runs").select("id,model_type,accuracy")
           .eq("id", row["served_by_run_id"]).limit(1).execute().data)
    assert run, "served_by_run_id does not resolve to a runs row"

    assert row["predicted_class"] in ("positive", "negative")
    assert isinstance(row["attention_vector_json"], list)
    assert len(row["attention_vector_json"]) == row["n_tokens"]


@needs_models
@needs_supabase
def test_no_stored_column_is_large_enough_to_be_review_text(supabase):
    """The privacy invariant, enforced rather than documented.

    Nothing legitimate in a metadata row is 500 characters long. A review is.
    """
    rows = (supabase.table("predictions").select("*")
            .order("id", desc=True).limit(50).execute().data or [])
    for row in rows:
        for column, value in row.items():
            if isinstance(value, str):
                assert len(value) < 500, (
                    f"predictions.{column} holds {len(value)} characters -- "
                    f"this table must never contain review text")


@needs_models
@needs_supabase
def test_the_hash_is_a_sha256_of_the_input(client, supabase):
    body = client.post("/predict", json={"text": TEXT}).json()
    assert body["input_sha256"] == hashlib.sha256(
        TEXT.encode("utf-8")).hexdigest()


@needs_models
def test_prediction_survives_a_supabase_outage(client, monkeypatch):
    """Logging is not the product. A dead database must not fail a request."""
    import api.main as main

    def broken():
        raise RuntimeError("supabase is down")

    monkeypatch.setattr(main, "_supabase", broken)
    r = client.post("/predict", json={"text": TEXT})
    assert r.status_code == 200
    assert r.json()["class_name"] in ("positive", "negative")


@needs_models
def test_healthz_reports_degraded_during_an_outage(client, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_supabase", lambda: None)
    body = client.get("/healthz").json()
    assert body["status"] == "degraded"
    assert body["supabase"] is False
    assert body["models_loaded"]