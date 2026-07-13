"""Request/response models for the FastAPI service. Pure Pydantic — no
torch/transformers/peft involved, so this module (and its validation
behavior) is fully testable without the ML stack installed."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    report_text: str = Field(..., min_length=1, description="Raw, unlabeled report text.")


class ExtractionResponse(BaseModel):
    comparison: str
    indication: str
    findings: str
    impression: str
    diagnosis: list[str]


class SummarizationRequest(BaseModel):
    findings: str = Field(..., min_length=1)
    indication: str = ""
    comparison: str = ""


class SummarizationResponse(BaseModel):
    impression: str


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # Defaults to 1536 (see app.py) — full thinking-mode + long structured
    # answer, validated end-to-end on GPU in Colab. Overridable per-request
    # specifically so this can be smoke-tested quickly on CPU (e.g. Docker
    # Desktop with no GPU) without a 20-40+ minute wait for a full generation
    # every time — a short value here just proves the pipeline completes.
    max_new_tokens: int | None = Field(default=None, ge=16, le=2048)


class QAResponse(BaseModel):
    answer: str
    reasoning: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    adapters_loaded: list[str]
