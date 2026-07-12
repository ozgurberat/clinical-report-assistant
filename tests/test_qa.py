"""Unit tests for the pure-Python logic in src.rag.qa — build_context and
THINK_PATTERN. No GPU, no model, no vector DB needed: heavy imports are
deferred inside load_base_model()/generate_answer(), same pattern as
evaluate.py and build_index.py, so this module is importable and testable
in any plain Python environment."""

from src.rag.qa import THINK_PATTERN, build_context

RETRIEVED = [
    {
        "report_id": "1857",
        "comparison": "",
        "indication": "",
        "findings": "Mild cardiomegaly. Clear lungs.",
        "impression": "Mild cardiomegaly. Clear lungs.",
        "diagnosis": ["cardiomegaly/mild"],
        "score": 0.7066,
    },
    {
        "report_id": "2032",
        "comparison": "None.",
        "indication": "Cough.",
        "findings": "The lungs are clear.",
        "impression": "Clear lungs.",
        "diagnosis": ["normal"],
        "score": 0.6906,
    },
]


def test_build_context_includes_report_id_and_score():
    context = build_context(RETRIEVED)
    assert "[Report 1857]" in context
    assert "similarity=0.71" in context
    assert "[Report 2032]" in context
    assert "similarity=0.69" in context


def test_build_context_omits_empty_optional_fields():
    context = build_context(RETRIEVED)
    # Report 1857 has no comparison/indication — those lines shouldn't appear
    # attached to it (they should only show up under report 2032's block).
    first_block, second_block = context.split("\n\n")
    assert "Comparison:" not in first_block
    assert "Indication:" not in first_block
    assert "Comparison: None." in second_block
    assert "Indication: Cough." in second_block


def test_build_context_always_includes_findings_and_impression():
    context = build_context(RETRIEVED)
    assert "Findings: Mild cardiomegaly. Clear lungs." in context
    assert "Findings: The lungs are clear." in context


def test_build_context_joins_diagnosis_list_with_commas():
    multi_diagnosis = [
        {
            "report_id": "999",
            "comparison": "",
            "indication": "",
            "findings": "f",
            "impression": "i",
            "diagnosis": ["cardiomegaly/mild", "atelectasis/base/left"],
            "score": 0.5,
        }
    ]
    context = build_context(multi_diagnosis)
    assert "Diagnosis: cardiomegaly/mild, atelectasis/base/left" in context


def test_build_context_empty_list_returns_empty_string():
    assert build_context([]) == ""


def test_think_pattern_extracts_reasoning_and_strips_it():
    text = "<think>\nStep 1: consider X.\nStep 2: consider Y.\n</think>\n\nFinal answer here."
    match = THINK_PATTERN.search(text)
    assert match is not None
    assert "Step 1: consider X." in match.group(1)
    remaining = THINK_PATTERN.sub("", text).strip()
    assert remaining == "Final answer here."


def test_think_pattern_no_match_when_absent():
    text = "Just a direct answer, no thinking block."
    assert THINK_PATTERN.search(text) is None
    assert THINK_PATTERN.sub("", text).strip() == text
