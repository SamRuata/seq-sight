"""Pydantic request/response models: the contract between the three clouds.

The validation here is the brief's "request-validation layer". It is not
decoration -- a sequence model will happily accept an empty string, a 200,000
character paste, or a list of floats where ids belong, and fail somewhere deep
in the embedding lookup. Rejecting those at the edge with a 422 keeps the
failure legible.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

MAX_CHARS = 20_000

# Pydantic reserves the `model_` prefix for its own attributes. Several fields
# here legitimately begin with it (model_name, model_type), so the namespace
# guard is switched off rather than the fields renamed -- the API contract
# should read naturally to its consumers.
ALLOW_MODEL_PREFIX = {"protected_namespaces": ()}


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    model_config = ALLOW_MODEL_PREFIX

    text: str = Field(..., min_length=1, max_length=MAX_CHARS,
                      description="Raw review text. Never stored.")
    model: Optional[str] = Field(
        None, description="Which run to serve: 'attention' or 'lstm'. "
                          "Defaults to the deployed model.")
    include_matrix: bool = Field(
        False, description="Also return the full [T,T] self-attention matrix. "
                           "Off by default: at T=128 it is 16,384 numbers.")
    true_label: Optional[int] = Field(
        None, ge=0, le=1, description="Ground truth, when scoring a known row.")

    @field_validator("text")
    @classmethod
    def not_only_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty or whitespace only")
        return v


class PredictBatchRequest(BaseModel):
    """Request body for POST /predict_batch."""

    model_config = ALLOW_MODEL_PREFIX

    texts: List[str] = Field(..., min_length=1, max_length=32)
    model: Optional[str] = None

    @field_validator("texts")
    @classmethod
    def each_non_empty(cls, v: List[str]) -> List[str]:
        for i, t in enumerate(v):
            if not t or not t.strip():
                raise ValueError(f"texts[{i}] is empty")
            if len(t) > MAX_CHARS:
                raise ValueError(f"texts[{i}] exceeds {MAX_CHARS} characters")
        return v


class TokenWeight(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    model_name: str
    run_id: Optional[int]
    label: int
    class_name: str
    confidence: float
    probs: List[float]

    tokens: List[str] = Field(
        ..., description="The tokens the model actually scored, after "
                         "truncation. Returned to the caller, never stored.")
    attention: List[float] = Field(
        ..., description="One weight per token, summing to 1. This is the "
                         "explanation: how much each position contributed to "
                         "the decision.")
    attention_matrix: Optional[List[List[float]]] = Field(
        None, description="[T,T] self-attention. Row i is what position i "
                          "attended to. Null for the recurrent baseline, "
                          "which has no attention to report.")

    n_tokens: int
    truncated: bool = Field(
        ..., description="True when the review was longer than the window.")
    input_sha256: str
    latency_ms: int


class RunSummary(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    id: int
    model_type: Optional[str] = None
    accuracy: float
    macro_f1: float
    n_params: Optional[int] = None
    best_epoch: Optional[int] = None
    train_seconds: Optional[float] = None
    epochs: int
    batch_size: int
    lr: float
    created_at: datetime


class PredictionRow(BaseModel):
    id: int
    served_by_run_id: int
    input_hash: str
    predicted_label: int
    predicted_class: str
    confidence: float
    attention_vector_json: List[float]
    n_tokens: int
    true_label: Optional[int] = None
    correct: Optional[bool] = None
    source: str
    latency_ms: Optional[int] = None
    created_at: datetime


class ModelInfo(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    name: str
    model_type: str
    n_params: int
    test_accuracy: float
    macro_f1: float
    train_seconds: float
    best_epoch: int
    explainable: bool


class ClassesResponse(BaseModel):
    classes: List[str]
    n_classes: int
    max_len: int
    vocab_size: int


class ScalingDemo(BaseModel):
    """Measured evidence for the sqrt(d_k) divisor, for the Concepts tab."""

    d_k: int
    sqrt_d_k: float
    unscaled_std: float
    scaled_std: float
    unscaled_max_softmax: float
    scaled_max_softmax: float
    unscaled_entropy: float
    scaled_entropy: float
    uniform_entropy: float


class Health(BaseModel):
    status: str
    models_loaded: List[str]
    vocab: bool
    supabase: bool
    detail: Optional[str] = None


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]
    served_model: Optional[str]