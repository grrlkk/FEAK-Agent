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
