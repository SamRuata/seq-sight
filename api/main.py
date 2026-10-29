"""FastAPI model service -- Cloud #2.

Serves both trained models behind one contract so the UI can ask the same
question of each and compare the answers. The attention model returns its
weights; the recurrent baseline returns none, which is the comparison's point
rather than a gap in the implementation.

Privacy invariant, enforced here and tested in tests/test_supabase_roundtrip:
a review's text is hashed and discarded. What reaches Postgres is the hash,
the model's output, and the attention vector -- never the words. A stored
heatmap therefore cannot be labelled with the tokens it describes, which is a
real cost of the design and is stated in the README rather than worked around.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from shared.schemas import (ClassesResponse, Health, ModelInfo,
                            PredictBatchRequest, PredictRequest,
                            PredictResponse, PredictionRow, RunSummary,
                            ScalingDemo, Version)
from shared.text import CLASS_NAMES, MAX_LEN, encode, load_vocab, pad_batch

MODELS_DIR = Path("models")
DEFAULT_MODEL = "attention"
TAGS = [
    {"name": "prediction", "description": "Classify text, with attention."},
    {"name": "models", "description": "The trained runs and their metrics."},
    {"name": "meta", "description": "Classes, limits, and the scaling demo."},
    {"name": "ops", "description": "Liveness and build information."},
]

app = FastAPI(
    title="Seq-Sight API",
    version="1.0.0",
    description="LSTM vs. scaled dot-product attention for review sentiment, "
                "with per-token explanations.",
    openapi_tags=TAGS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- process state ---------------------------------------------------------
MODELS: Dict[str, dict] = {}       # name -> {"model", "meta"}
VOCAB: Dict[str, int] = {}
RUN_IDS: Dict[str, Optional[int]] = {}
STARTUP_ERROR: Optional[str] = None


def _supabase():
    from supabase import create_client

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _resolve_run_ids() -> None:
    """Match each loaded checkpoint to its `runs` row, newest first.

    Best-effort: a prediction must still be servable when Supabase is
    unreachable, so a failure here leaves run ids as None rather than raising.
    """
    try:
        client = _supabase()
        if client is None:
            return
        res = (client.table("runs").select("id,model_type,accuracy")
               .order("id", desc=True).limit(50).execute())
        for name, bundle in MODELS.items():
            want = bundle["meta"]["model_type"]
            acc = bundle["meta"]["metrics"]["test_accuracy"]
            for row in res.data or []:
                if row["model_type"] == want and abs(
                        float(row["accuracy"]) - acc) < 1e-9:
                    RUN_IDS[name] = int(row["id"])
                    break
    except Exception:                                         # noqa: BLE001
        pass


@app.on_event("startup")
def _startup() -> None:
    """Load the vocabulary and every checkpoint in models/.

    Failures are captured rather than raised: a service that cannot start
    cannot tell you why, and Render restarts it forever. /healthz reports.
    """
    global STARTUP_ERROR, VOCAB
    import joblib

    from api.models import build_model

    try:
        VOCAB = load_vocab()
        for path in sorted(MODELS_DIR.glob("*.joblib")):
            ckpt = joblib.load(path)
            model = build_model({**ckpt["config"],
                                 "vocab_size": ckpt["vocab_size"]})
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            MODELS[path.stem] = {"model": model, "meta": ckpt}
            RUN_IDS[path.stem] = None
        if not MODELS:
            STARTUP_ERROR = f"no checkpoints found in {MODELS_DIR}/"
        else:
            _resolve_run_ids()
    except Exception as exc:                                  # noqa: BLE001
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"


def _pick(name: Optional[str]) -> Tuple[str, dict]:
    chosen = name or (DEFAULT_MODEL if DEFAULT_MODEL in MODELS
                      else next(iter(MODELS), ""))
    if chosen not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model {chosen!r}; loaded: {sorted(MODELS)}")
    return chosen, MODELS[chosen]


def _log(name: str, payload: dict) -> None:
    """Write one predictions row. Never raises: logging is not the product.

    The client construction sits inside the try as well as the insert.
    Building a Supabase client can itself fail -- bad credentials, no DNS --
    and if that escapes, a database problem becomes a failed prediction.
    """
    try:
        client = _supabase()
        run_id = RUN_IDS.get(name)
        if client is None or run_id is None:
            return
        client.table("predictions").insert(
            {**payload, "served_by_run_id": run_id}).execute()
    except Exception:                                         # noqa: BLE001
        pass


def _infer(name: str, bundle: dict, text: str, include_matrix: bool,
           true_label: Optional[int], source: str) -> PredictResponse:
    started = time.perf_counter()

    ids, tokens = encode(text, VOCAB)
    if not tokens:
        raise HTTPException(
            status_code=422,
            detail="no recognisable tokens; the text is punctuation or symbols only")

    raw_len = len(text)
    batch_ids, lengths, mask = pad_batch([ids])

    with torch.no_grad():
        logits, weights, matrix = bundle["model"](batch_ids, lengths, mask)
        probs = torch.softmax(logits, dim=1)[0]

    label = int(probs.argmax())
    attention = ([round(float(w), 6) for w in weights[0][:len(tokens)].tolist()]
                 if weights is not None
                 else [round(1.0 / len(tokens), 6)] * len(tokens))
    latency = int((time.perf_counter() - started) * 1000)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    _log(name, {
        "input_hash": digest,
        "predicted_label": label,
        "predicted_class": CLASS_NAMES[label],
        "confidence": float(probs[label]),
        "probs": [float(p) for p in probs.tolist()],
        # Weights only. Storing the tokens beside them would reconstruct the
        # review, which is the one thing this table must never hold.
        "attention_vector_json": [round(w, 4) for w in attention],
        "n_tokens": len(tokens),
        "true_label": true_label,
        "correct": None if true_label is None else (label == true_label),
        "source": source,
        "latency_ms": latency,
    })

    return PredictResponse(
        model_name=name,
        run_id=RUN_IDS.get(name),
        label=label,
        class_name=CLASS_NAMES[label],
        confidence=float(probs[label]),
        probs=[float(p) for p in probs.tolist()],
        tokens=tokens,
        attention=attention,
        attention_matrix=(
            [[round(float(v), 5) for v in row[:len(tokens)]]
             for row in matrix[0][:len(tokens)].tolist()]
            if (include_matrix and matrix is not None) else None),
        n_tokens=len(tokens),
        truncated=raw_len > 0 and len(tokens) >= MAX_LEN,
        input_sha256=digest,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    name, bundle = _pick(req.model)
    return _infer(name, bundle, req.text, req.include_matrix,
                  req.true_label, "user")


@app.post("/predict_batch", response_model=List[PredictResponse],
          tags=["prediction"])
def predict_batch(req: PredictBatchRequest) -> List[PredictResponse]:
    name, bundle = _pick(req.model)
    return [_infer(name, bundle, t, False, None, "batch") for t in req.texts]


# ---------------------------------------------------------------------------
# Models and history
# ---------------------------------------------------------------------------
@app.get("/models", response_model=List[ModelInfo], tags=["models"])
def models() -> List[ModelInfo]:
    out = []
    for name, bundle in MODELS.items():
        meta = bundle["meta"]
        out.append(ModelInfo(
            name=name,
            model_type=meta["model_type"],
            n_params=meta["n_params"],
            test_accuracy=meta["metrics"]["test_accuracy"],
            macro_f1=meta["metrics"]["macro_f1"],
            train_seconds=meta["train_seconds"],
            best_epoch=meta["best_epoch"],
            explainable=meta["model_type"] == "attention",
        ))
    return sorted(out, key=lambda m: m.test_accuracy, reverse=True)


@app.get("/runs", response_model=List[RunSummary], tags=["models"])
def runs(limit: int = Query(20, ge=1, le=100)) -> List[RunSummary]:
    client = _supabase()
    if client is None:
        raise HTTPException(503, "Supabase is not configured")
    res = (client.table("runs")
           .select("id,model_type,accuracy,macro_f1,n_params,best_epoch,"
                   "train_seconds,epochs,batch_size,lr,created_at")
           .order("id", desc=True).limit(limit).execute())
    return [RunSummary(**row) for row in (res.data or [])]


@app.get("/predictions", response_model=List[PredictionRow], tags=["models"])
def predictions(limit: int = Query(25, ge=1, le=200)) -> List[PredictionRow]:
    client = _supabase()
    if client is None:
        raise HTTPException(503, "Supabase is not configured")
    res = (client.table("predictions").select("*")
           .order("id", desc=True).limit(limit).execute())
    return [PredictionRow(**row) for row in (res.data or [])]


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
def classes() -> ClassesResponse:
    return ClassesResponse(classes=CLASS_NAMES, n_classes=len(CLASS_NAMES),
                           max_len=MAX_LEN, vocab_size=len(VOCAB))


@app.get("/scaling_demo", response_model=ScalingDemo, tags=["meta"])
def scaling_demo(d_k: int = Query(128, ge=2, le=1024)) -> ScalingDemo:
    """Measured effect of the sqrt(d_k) divisor. Feeds the Concepts tab."""
    from api.models import saturation_demo

    return ScalingDemo(**saturation_demo(d_k=d_k))


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    db_ok = False
    try:
        client = _supabase()
        if client is not None:
            client.table("runs").select("id").limit(1).execute()
            db_ok = True
    except Exception:                                         # noqa: BLE001
        db_ok = False

    healthy = bool(MODELS) and bool(VOCAB)
    return Health(
        status="ok" if healthy and db_ok else "degraded",
        models_loaded=sorted(MODELS),
        vocab=bool(VOCAB),
        supabase=db_ok,
        detail=STARTUP_ERROR,
    )


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                         # noqa: BLE001
        sha = os.getenv("RENDER_GIT_COMMIT", "unknown")[:7] or "unknown"

    url = os.getenv("SUPABASE_URL", "")
    ref = url.split("//")[-1].split(".")[0] if url else None
    return Version(git_sha=sha, torch_version=torch.__version__,
                   supabase_project_ref=ref,
                   served_model=DEFAULT_MODEL if DEFAULT_MODEL in MODELS
                   else next(iter(MODELS), None))