"""Shared fixtures.

The API is exercised through TestClient as a CONTEXT MANAGER, which is what
makes FastAPI run its startup handler -- without that the model registry is
empty and every prediction 404s for reasons that look like a routing bug.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

REFERENCE = Path("tests/reference.json")


def have_models() -> bool:
    return Path("models/vocab.json").exists() and any(
        Path("models").glob("*.joblib"))


def have_supabase() -> bool:
    return bool(os.environ.get("SUPABASE_URL")
                and os.environ.get("SUPABASE_SERVICE_KEY"))


needs_models = pytest.mark.skipif(
    not have_models(),
    reason="no trained checkpoints; run `python -m api.train` first")

needs_supabase = pytest.mark.skipif(
    not have_supabase(),
    reason="SUPABASE_URL / SUPABASE_SERVICE_KEY not set")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def supabase():
    from supabase import create_client

    return create_client(os.environ["SUPABASE_URL"],
                         os.environ["SUPABASE_SERVICE_KEY"])