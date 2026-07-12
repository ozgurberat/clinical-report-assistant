"""Unit tests for src.serving.schemas — pure Pydantic validation, no ML
dependencies involved.

pydantic is deliberately NOT part of CI's dependency set (see
.github/workflows/ci.yml — it installs with `--no-deps` to keep the "offline"
test job fast and free of the full ML/serving stack, the same reason every
other test file in this repo avoids needing torch/transformers/peft/etc. at
import time). This module breaks that pattern by nature — it's testing
Pydantic schemas, so it needs pydantic — so pytest.importorskip() below tells
pytest to skip just this file when pydantic is absent, rather than erroring
out and aborting the entire collection (which is what happened the first
time this was pushed: one missing import marked all 43 other, unrelated
tests as failed too).

NOTE, unlike every other test file in this repo: these have NOT actually been
run anywhere yet. pydantic isn't installed in the sandbox this project was
built in, and it has no network access to install it (confirmed earlier when
even `pip install pytest` failed against a blocked proxy). Run this file for
real the first time you have pydantic available (Colab, the Docker build, or
your own machine) and report back if anything fails."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from src.serving.schemas import (  # noqa: E402
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    QARequest,
    QAResponse,
    SummarizationRequest,
)


def _expect_validation_error(build):
    try:
        build()
    except ValidationError:
        return
    raise AssertionError("expected a ValidationError, but none was raised")


def test_extraction_request_rejects_empty_report_text():
    _expect_validation_error(lambda: ExtractionRequest(report_text=""))


def test_extraction_request_accepts_valid_text():
    req = ExtractionRequest(report_text="Findings: clear lungs.")
    assert req.report_text == "Findings: clear lungs."


def test_extraction_response_rejects_missing_fields():
    _expect_validation_error(
        lambda: ExtractionResponse(comparison="", indication="", findings="")
        # impression and diagnosis omitted on purpose
    )


def test_extraction_response_accepts_complete_fields():
    resp = ExtractionResponse(
        comparison="",
        indication="Chest pain.",
        findings="Clear lungs.",
        impression="No acute disease.",
        diagnosis=["normal"],
    )
    assert resp.diagnosis == ["normal"]


def test_summarization_request_defaults_indication_and_comparison_empty():
    req = SummarizationRequest(findings="Clear lungs.")
    assert req.indication == ""
    assert req.comparison == ""


def test_summarization_request_rejects_empty_findings():
    _expect_validation_error(lambda: SummarizationRequest(findings=""))


def test_qa_request_top_k_defaults_to_none():
    req = QARequest(question="What's typical follow-up?")
    assert req.top_k is None


def test_qa_request_top_k_out_of_range_rejected():
    _expect_validation_error(lambda: QARequest(question="q", top_k=0))
    _expect_validation_error(lambda: QARequest(question="q", top_k=21))


def test_qa_request_top_k_boundary_values_accepted():
    assert QARequest(question="q", top_k=1).top_k == 1
    assert QARequest(question="q", top_k=20).top_k == 20


def test_qa_response_shape():
    resp = QAResponse(answer="Mild cardiomegaly is common.", reasoning="", sources=["1857", "2032"])
    assert resp.sources == ["1857", "2032"]


def test_health_response_shape():
    resp = HealthResponse(status="ok", cuda_available=True, adapters_loaded=["extraction", "summarization"])
    assert resp.adapters_loaded == ["extraction", "summarization"]
