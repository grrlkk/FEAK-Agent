import json

from staged_feedback.events import save_json
from staged_feedback.report import render


def test_successful_gpu_preflight_resolves_old_failure_in_report(tmp_path):
    save_json(tmp_path / "preflight.json", {"cuda_available": True})
    save_json(tmp_path / "preflight.error.json", {"component": "native_kanana_gpu", "reason": "이전 실행 오류"})
    summary = render(tmp_path)
    assert not summary["anomalies"]
    assert len(summary["resolved_anomalies"]) == 1
    assert "resolved" not in json.loads((tmp_path / "preflight.error.json").read_text())


def test_shared_budget_report_distinguishes_attempt_and_session(tmp_path):
    save_json(tmp_path / "parent/budget.json", {"spent_upper_usd": 0.02, "budget_usd": 5})
    save_json(tmp_path / "retry/budget_reference.json", {"ledger": str(tmp_path / "parent/budget.json")})
    (tmp_path / "retry/calls.jsonl").write_text(json.dumps({"dispatched": True, "cost_usd": 0.01}) + "\n")
    summary = render(tmp_path / "retry")
    assert summary["confirmed_cost_usd"] == 0.01
    assert summary["budget"]["spent_upper_usd"] == 0.02


def test_comparison_reports_zero_turn_gate_and_old_unmeasured_tests(tmp_path, executor):
    from test_engine import execute
    from test_student_gates import Intake
    from staged_feedback.report import essay_metrics

    old, new = tmp_path / "old", tmp_path / "new"
    old_result, _, _, _ = execute(old / "essays/synthetic", executor)
    old_result.pop("baseline_tests")
    old_result.pop("excluded_initial_issues")
    save_json(old / "essays/synthetic/result.json", old_result)
    execute(new / "essays/synthetic", executor, llm=Intake())
    save_json(new / "selection.json", {"essays": [{"essay_id": "synthetic", "stratum": "low",
        "human_original_total": 30, "cohort": "repeated", "genre": "정서"}]})
    save_json(new / "comparison_reference.json", {"baseline_run": str(old), "repeated_ids": ["synthetic"]})
    summary = render(new)
    comparison = json.loads((new / "comparison.json").read_text())["essays"][0]
    assert comparison["delta"]["turns"] == -1
    assert comparison["before"]["original_tests_run"] is False
    assert comparison["before"]["excluded_initial_issues"] is None
    assert comparison["after"]["original_tests_run"] is True
    assert comparison["after"]["plan_change_reasons"]["student_task_transfer"] == 2
    assert comparison["after"]["plan_changes_initial"] == 2
    assert comparison["after"]["plan_changes_during_turns"] == 0
    assert len(summary["essay_metrics"]["synthetic"]["student_tasks"]) == 1
    assert "큰 틀을 고친 뒤" in (new / "report.html").read_text()


def test_continuation_costs_count_reused_calls_once(tmp_path):
    first, parent, current = (tmp_path / name for name in ("first", "parent", "current"))
    for directory, amounts in ((first, [.01]), (parent, [.002, .003]), (current, [.002, .004])):
        directory.mkdir()
        (directory / "calls.jsonl").write_text("".join(json.dumps({"cost_usd": value,
            "dispatched": True}) + "\n" for value in amounts))
    save_json(current / "continuation.json", {"parent_run": str(parent),
        "prior_attempts": [str(first), str(parent)], "reused_ids": ["done"], "rerun_ids": ["failed"],
        "reused_calls": 1, "reused_confirmed_cost_usd": .002, "note": "완료 결과 보존"})
    summary = render(current)
    assert summary["api_calls"] == 2 and summary["new_api_calls"] == 1
    assert summary["all_attempts_api_calls"] == 4
    assert abs(summary["all_attempts_confirmed_cost_usd"] - .019) < 1e-10
