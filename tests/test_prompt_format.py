"""Unit tests for src.finetuning.prompt_format — synthetic records mirroring
the real reports.jsonl schema, covering the data quirks found during EDA."""

import json

from src.finetuning.prompt_format import (
    build_extraction_example,
    build_summarization_example,
    clean_deidentified,
    clean_diagnosis,
)

FULL_RECORD = {
    "report_id": "CXR1",
    "comparison": "None.",
    "indication": "XXXX-year-old male, chest pain.",
    "findings": "The heart size is mildly enlarged. No pleural effusion.",
    "impression": "Mild cardiomegaly, no acute disease.",
    "mesh_major": ["cardiomegaly/mild", "no indexing"],
    "image_ids": ["CXR1_IM-0001-1001"],
}

NO_IMPRESSION_RECORD = {
    "report_id": "CXR2",
    "comparison": "",
    "indication": "",
    "findings": "Lungs are clear bilaterally.",
    "impression": "",
    "mesh_major": ["normal"],
    "image_ids": [],
}


def test_clean_deidentified_replaces_all_occurrences():
    assert clean_deidentified("XXXX-year-old, chest x-XXXX") == "[REDACTED]-year-old, chest x-[REDACTED]"


def test_clean_deidentified_leaves_normal_text_untouched():
    assert clean_deidentified("No acute cardiopulmonary process.") == "No acute cardiopulmonary process."


def test_clean_diagnosis_drops_administrative_tags():
    assert clean_diagnosis(["cardiomegaly", "no indexing", "normal"]) == ["cardiomegaly", "normal"]


def test_build_extraction_example_structure():
    example = build_extraction_example(FULL_RECORD)
    assert example is not None
    assert example["report_id"] == "CXR1"

    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]

    user_content = example["messages"][1]["content"]
    assert "XXXX" not in user_content
    assert "[REDACTED]" in user_content
    assert "mildly enlarged" in user_content

    target = json.loads(example["messages"][2]["content"])
    assert target["findings"].startswith("The heart")
    assert target["diagnosis"] == ["cardiomegaly/mild"]  # "no indexing" filtered out
    assert "[REDACTED]" in target["indication"]


def test_build_extraction_example_returns_none_for_fully_empty_record():
    empty = {**FULL_RECORD, "comparison": "", "indication": "", "findings": "", "impression": ""}
    assert build_extraction_example(empty) is None


def test_build_summarization_example_skips_empty_impression():
    assert build_summarization_example(NO_IMPRESSION_RECORD) is None


def test_build_summarization_example_omits_blank_optional_fields():
    example = build_summarization_example(FULL_RECORD)
    assert example is not None
    user_content = example["messages"][1]["content"]
    # comparison was "None." -> stripped but non-empty, so it IS included
    assert "Comparison:" in user_content
    assert "Indication:" in user_content
    assert "Findings:" in user_content
    assert example["messages"][2]["content"] == "Mild cardiomegaly, no acute disease."


def test_build_summarization_example_omits_truly_blank_fields():
    record = {**FULL_RECORD, "comparison": "", "indication": ""}
    example = build_summarization_example(record)
    assert example is not None
    user_content = example["messages"][1]["content"]
    assert "Comparison:" not in user_content
    assert "Indication:" not in user_content
    assert user_content.startswith("Findings:")
