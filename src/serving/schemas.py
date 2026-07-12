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


class QAResponse(BaseModel):
    answer: str
    reasoning: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    adapters_loaded: list[str]
