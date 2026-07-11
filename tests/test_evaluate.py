"""Unit tests for src.finetuning.evaluate's pure-Python scoring functions
(token_f1, set_f1). These don't need a GPU or model, unlike the rest of
evaluate.py, so they're testable without Colab."""

from src.finetuning.evaluate import set_f1, token_f1


def test_token_f1_identical_strings_scores_one():
    assert token_f1("no acute cardiopulmonary findings", "no acute cardiopulmonary findings") == 1.0


def test_token_f1_completely_different_scores_zero():
    assert token_f1("cardiomegaly present", "lungs are clear") == 0.0


def test_token_f1_partial_overlap_scores_between_zero_and_one():
    score = token_f1("no acute findings", "no acute pulmonary findings")
    assert 0.0 < score < 1.0


def test_token_f1_both_empty_scores_one():
    # Two empty strings should count as a match, not a failure.
    assert token_f1("", "") == 1.0


def test_token_f1_one_empty_scores_zero():
    assert token_f1("", "some content") == 0.0
    assert token_f1("some content", "") == 0.0


def test_token_f1_is_case_insensitive():
    assert token_f1("Mild Cardiomegaly", "mild cardiomegaly") == 1.0


def test_set_f1_identical_lists_scores_one():
    assert set_f1(["cardiomegaly/mild", "normal"], ["cardiomegaly/mild", "normal"]) == 1.0


def test_set_f1_ignores_order():
    assert set_f1(["a", "b"], ["b", "a"]) == 1.0


def test_set_f1_partial_overlap():
    # 1 of 2 predicted correct, 1 of 2 gold recovered -> precision=recall=0.5 -> f1=0.5
    assert set_f1(["a", "c"], ["a", "b"]) == 0.5


def test_set_f1_no_overlap_scores_zero():
    assert set_f1(["x", "y"], ["a", "b"]) == 0.0


def test_set_f1_both_empty_scores_one():
    assert set_f1([], []) == 1.0


def test_set_f1_one_empty_scores_zero():
    assert set_f1([], ["a"]) == 0.0
    assert set_f1(["a"], []) == 0.0
