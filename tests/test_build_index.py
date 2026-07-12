"""Unit tests for the pure-Python logic in src.rag.build_index —
build_embedding_text and build_payload. Neither needs sentence-transformers
or qdrant-client installed, same reasoning as tests/test_evaluate.py: the
heavy ML/DB imports are deferred inside main(), so this module can be
imported and tested in any plain Python environment."""

from src.rag.build_index import build_embedding_text, build_payload

FULL_RECORD = {
    "report_id": "CXR1",
    "comparison": "None.",
    "indication": "Chest pain.",
    "findings": "The heart is mildly enlarged. Lungs are clear.",
    "impression": "Mild cardiomegaly, no acute disease.",
    "mesh_major": ["cardiomegaly/mild", "no acute disease"],
    "image_ids": ["CXR1_IM-0001-1001"],
}

PARTIAL_RECORD = {
    "report_id": "CXR2",
    "comparison": "",
    "indication": "",
    "findings": "Lungs are clear bilaterally.",
    "impression": "",
    "mesh_major": ["normal"],
    "image_ids": [],
}

EMPTY_RECORD = {
    "report_id": "CXR3",
    "comparison": "",
    "indication": "",
    "findings": "",
    "impression": "",
    "mesh_major": [],
    "image_ids": [],
}


def test_build_embedding_text_includes_all_populated_sections():
    text = build_embedding_text(FULL_RECORD)
    assert "Comparison: None." in text
    assert "Indication: Chest pain." in text
    assert "Findings: The heart is mildly enlarged. Lungs are clear." in text
    assert "Impression: Mild cardiomegaly, no acute disease." in text


def test_build_embedding_text_skips_empty_sections_entirely():
    text = build_embedding_text(PARTIAL_RECORD)
    assert "Comparison:" not in text
    assert "Indication:" not in text
    assert "Impression:" not in text
    assert "Findings: Lungs are clear bilaterally." in text


def test_build_embedding_text_all_empty_returns_empty_string():
    assert build_embedding_text(EMPTY_RECORD) == ""


def test_build_embedding_text_preserves_section_order():
    text = build_embedding_text(FULL_RECORD)
    # Comparison should appear before Indication, before Findings, before Impression.
    assert text.index("Comparison") < text.index("Indication") < text.index("Findings") < text.index("Impression")


def test_build_payload_carries_expected_fields():
    payload = build_payload(FULL_RECORD)
    assert payload["report_id"] == "CXR1"
    assert payload["findings"] == "The heart is mildly enlarged. Lungs are clear."
    assert payload["diagnosis"] == ["cardiomegaly/mild", "no acute disease"]
    # image_ids shouldn't leak into the payload — not relevant to text retrieval.
    assert "image_ids" not in payload


def test_build_payload_handles_missing_keys_gracefully():
    # A record missing mesh_major entirely (rather than an empty list)
    # shouldn't crash — should just come back as an empty diagnosis list.
    minimal = {"report_id": "CXR4"}
    payload = build_payload(minimal)
    assert payload["report_id"] == "CXR4"
    assert payload["diagnosis"] == []
    assert payload["findings"] == ""
