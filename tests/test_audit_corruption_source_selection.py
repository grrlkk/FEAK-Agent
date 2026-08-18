import pytest

from scripts.audit_corruption_source_selection import build_selection_bias_report


def test_selection_bias_report_separates_selection_and_generation_effects():
    source = [
        _source("r1", "짧다.", 4.0, "문항 하나"),
        _source("r2", "두 문장이다. 조금 더 길다.", 4.1, "문항 둘"),
        _source("r3", "세 문장이다. 충분히 길다. 예시도 있다.", 4.2, "문항 셋"),
        _source("r4", "마지막 글이다.", 4.3, "문항 넷"),
    ]
    attempts = [
        _attempt("r1", "failed", ["SHUFFLE_FLOW"], []),
        _attempt("r2", "ok", ["DELETE_SPECIFICS"], [{"operator": "DELETE_SPECIFICS"}]),
        _attempt("r3", "ok", ["INJECT_LEX_REPEAT"], [{"operator": "INJECT_LEX_REPEAT"}]),
        _attempt("r4", "partial", ["INSERT_OFFTOPIC", "SHUFFLE_FLOW"], [{"operator": "INSERT_OFFTOPIC"}]),
    ]
    selected = [attempts[1]]

    report = build_selection_bias_report(source, attempts, selected)

    assert report["counts"]["attempted"] == 4
    assert report["counts"]["selected"] == 1
    assert report["counts"]["generation_ok"] == 2
    assert report["counts"]["attempt_status"] == {
        "failed": 1,
        "ok": 2,
        "partial": 1,
    }
    assert report["cohorts"]["generation_incomplete"]["n"] == 2
    assert report["failure_profile"]["failed_operator"] == {
        "SHUFFLE_FLOW": 2,
    }
    assert report["question_distribution_note"]["all_attempted_questions_unique"] is True
    assert (
        report["comparisons"]["selected_vs_not_selected"]["numeric"]
        ["text_char_count"]["standardized_mean_difference"]
        is not None
    )
    assert (
        report["comparisons"]["selected_vs_not_selected"]
        ["exact_question_distribution"]["shared_questions"]
        == 0
    )


def test_selection_bias_report_rejects_non_complete_selected_chain():
    source = [_source("r1", "글이다.", 4.0, "문항")]
    attempt = _attempt("r1", "partial", ["SHUFFLE_FLOW"], [])

    with pytest.raises(ValueError, match="not complete"):
        build_selection_bias_report(source, [attempt], [attempt])


def _source(record_id: str, text: str, grade: float, question: str) -> dict:
    return {
        "record_id": record_id,
        "text": text,
        "grader_avg": grade,
        "question": question,
    }


def _attempt(record_id: str, status: str, planned: list[str], steps: list[dict]) -> dict:
    return {
        "record_id": record_id,
        "status": status,
        "planned_operators": planned,
        "steps": steps,
    }
