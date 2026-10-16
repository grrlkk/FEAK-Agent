"""Local Korean report; all raw records remain available beside the report."""
from collections import Counter
from html import escape
import json
from pathlib import Path

from .events import save_json


def json_block(value):
    return "<pre>" + escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def render(directory):
    directory = Path(directory)
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("essays/*/result.json"))]
    calls_path = directory / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()] if calls_path.exists() else []
    events = [json.loads(line) for p in directory.glob("essays/*/events.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
    rejected = [e["payload"] for e in events if e["kind"] == "operation_rejected"]
    summary = {"completed_essays": len(results), "api_calls": sum(c.get("dispatched", True) for c in calls),
        "mock_essays": sum(bool(r.get("mock")) for r in results),
        "request_attempts": len(calls),
        "confirmed_cost_usd": sum(c.get("cost_usd") or 0 for c in calls),
        "route_distribution": dict(Counter(r["route"] for r in results)),
        "initial_strategy_distribution": dict(Counter((r["initial_strategy"] or {}).get("name") for r in results)),
        "plan_changes": sum(e["kind"] == "plan_changed" and e["phase"] != "initial" and any(
            k in e["payload"] for k in ("strategy", "turn_plan", "turn_budget", "priorities", "dispositions", "new_issue_ids", "reopened_issue_ids")) for e in events),
        "operation_rejection_reasons": dict(Counter(r["reason_code"] for r in rejected)),
        "preserve_violation_attempts": sum(bool(r.get("preserve_violation_attempt")) for r in rejected),
        "test_trajectories": {r["essay_id"]: [t.get("test_summary") for t in r["turns"]] for r in results}}
    budget_path = directory / "budget.json"
    if budget_path.exists():
        summary["budget"] = json.loads(budget_path.read_text())
    errors = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("**/*.error.json")]
    summary["anomalies"] = [e for e in errors if not e.get("resolved")]
    summary["resolved_anomalies"] = [e for e in errors if e.get("resolved")]
    summary["essay_costs"] = {r["essay_id"]: {
        "confirmed_cost_usd": sum(c.get("cost_usd") or 0 for c in calls if c.get("essay_id") == r["essay_id"]),
        "api_calls": sum(c.get("dispatched", True) for c in calls if c.get("essay_id") == r["essay_id"])} for r in results}
    selection_path = directory / "selection.json"
    if selection_path.exists():
        summary["selected_essays"] = [{k: e[k] for k in ("essay_id", "stratum", "human_original_total")}
            for e in json.loads(selection_path.read_text())["essays"]]
    save_json(directory / "summary.json", summary)
    parts = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>단계별 피드백 검증 보고</title>",
        "<style>body{max-width:1100px;margin:40px auto;font:16px/1.7 sans-serif;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:15px}article{border-top:2px solid #aaa;margin-top:40px}details{margin:14px 0}li{margin:8px 0}</style>",
        "<h1>단계별 피드백 검증 보고</h1><p>수정본은 학생이 피드백을 반영한 글의 시뮬레이션입니다. 이 실행은 학습 효과나 품질 향상을 입증하지 않습니다.</p>",
        "<h2>실행 요약</h2>", json_block(summary)]
    for result in results:
        parts.extend([f"<article><h2>{escape(result['essay_id'])}</h2>",
            "<p>검증 모드: 합성 원문·mock LLM/진단</p>" if result.get("mock") else "<p>검증 모드: 실제 API·기존 진단 어댑터</p>",
            f"<p>경로: {escape(result['route'])} / 종료 근거: {escape(result['route_reason'])}</p>",
            "<h3>초기 난이도·전략·턴 계획</h3>", json_block({key: result[key] for key in
                ("difficulty", "initial_strategy", "initial_turn_plan", "strategy_history", "turn_plan_history")})])
        initial = directory / "essays" / result["essay_id"] / "initial_plan.json"
        if initial.exists():
            plan = json.loads(initial.read_text())
            parts.extend(["<details><summary>초기 이슈 목록과 테스트</summary>", json_block(plan["issues"]), "</details>"])
        for turn in result["turns"]:
            parts.extend([f"<h3>{turn['turn']}턴</h3>", "<h4>피드백과 체크리스트</h4>",
                json_block(turn.get("feedback")), json_block(turn.get("checklist", [])),
                "<h4>편집 연산</h4><ul>"])
            for operation in turn.get("operations", []):
                label = operation.get("description") or operation.get("reason", "")
                parts.append("<li>" + escape(label) + (" (적용)" if operation.get("applied") else " (거부)") + "</li>")
            parts.extend(["</ul><h4>수정본</h4><pre>", escape(turn["revised_text"]),
                "</pre><details><summary>테스트 결과·총괄 검토</summary>",
                json_block({"tests": turn.get("tests"), "review": turn.get("review")}), "</details>"])
        parts.extend(["<h3>최종 점수와 변화</h3>", json_block({k: result[k] for k in
            ("initial_scores", "final_scores", "score_delta", "score_n", "remaining_issue_ids")}),
            "<h3>학생 과제</h3>", json_block(result["student_tasks"]),
            "<details><summary>역할별 메모리 요약 예시</summary>", json_block(result["memory_summaries"][:8]),
            "</details><h3>최종본</h3><pre>", escape(result["final_text"]), "</pre></article>"])
    parts.append("</html>")
    (directory / "report.html").write_text("\n".join(parts), encoding="utf-8")
    return summary
