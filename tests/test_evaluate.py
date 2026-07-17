from __future__ import annotations

import pytest

from fi_lit.evaluate import EvaluationError, aggregate_scores, format_generation_prompt, rouge_l_f1, score_prediction


def test_prompt_matches_training_prefix_without_reference() -> None:
    prompt = format_generation_prompt({"definition": ["Classify sentiment."], "input": "Fine", "references": ["Positive"]})
    assert prompt == "Definition:\nClassify sentiment.\n\nInput:\nFine\n\nOutput:\n"
    assert "Positive" not in prompt


def test_scoring_uses_best_reference_and_normalizes_whitespace() -> None:
    scores = score_prediction("  MIXED ", ["negative", "mixed"])
    assert scores["exact_match"] == 1.0
    assert scores["rouge_l"] == 1.0
    assert rouge_l_f1("a b c", "a c") > 0.0


def test_aggregate_reports_micro_and_task_macro() -> None:
    summary = aggregate_scores([
        {"task_id": "a", "exact_match": 1.0, "rouge_l": 1.0},
        {"task_id": "a", "exact_match": 0.0, "rouge_l": 0.5},
        {"task_id": "b", "exact_match": 1.0, "rouge_l": 1.0},
    ])
    assert summary["micro"]["exact_match"] == pytest.approx(2 / 3)
    assert summary["macro_by_task"]["exact_match"] == pytest.approx(0.75)
    assert summary["tasks"] == 2


def test_scoring_rejects_missing_references() -> None:
    with pytest.raises(EvaluationError, match="reference"):
        score_prediction("answer", [])
