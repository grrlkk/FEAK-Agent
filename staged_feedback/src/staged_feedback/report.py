"""Local Korean report; all raw records remain available beside the report."""
from collections import Counter
from html import escape
import json
from pathlib import Path

from .events import save_json


def json_block(value):
    return "<pre>" + escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


PLAN_FIELDS = {"strategy", "turn_plan", "turn_budget", "priorities", "dispositions", "new_issue_ids",
               "reopened_issue_ids", "student_issue_ids", "excluded_original_pass_ids"}
REASONS = {"student_task_transfer": "학생 과제 전환", "original_test_pass": "원문에서 이미 통과",
           "new_issue": "새 문제", "resolved_issue": "해결된 문제", "reopened_issue": "다시 열린 문제",
           "plan_adjustment": "기타 계획 조정"}


def essay_metrics(directory):
    directory = Path(directory)
    path = directory / "calls.jsonl"
    calls = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    metrics = {}
    for path in sorted(directory.glob("essays/*/result.json")):
        result = json.loads(path.read_text())
        events = [json.loads(s) for s in (path.parent / "events.jsonl").read_text().splitlines()]
        changes = []
        for event in events:
            data = event["payload"]
            if event["kind"] != "plan_changed" or event["phase"] == "initial" or not PLAN_FIELDS & data.keys():
                continue
            categories = data.get("reason_categories")
            if categories is None:
                # Older runs predate explicit cause labels; retain their original reason too.
                categories = []
                if any(d["action"] == "student_task" for d in data.get("dispositions", [])):
                    categories.append("student_task_transfer")
                if data.get("new_issue_ids"):
                    categories.append("new_issue")
                if data.get("reopened_issue_ids"):
                    categories.append("reopened_issue")
                if any(e["kind"] == "issue_resolved" and e["turn"] == event["turn"] and
                       e["phase"] == event["phase"] for e in events):
                    categories.append("resolved_issue")
                categories = categories or ["plan_adjustment"]
            reason = data.get("reason", "")
            category_source = "recorded" if "reason_categories" in data else "inferred_from_events"
            correction = {}
            if "resolved_issue" in categories and not any(e["kind"] == "issue_resolved" and
                    e["seq"] < event["seq"] for e in events):
                # Earlier controller versions mislabeled an initially empty turn as a resolution.
                correction = {"original_categories": categories, "original_reason": reason}
                categories = [c for c in categories if c != "resolved_issue"] or ["plan_adjustment"]
                category_source = "corrected_from_events"
                reason = "해결 사건이 생기기 전, 초기 계획의 빈 턴에 남은 수정 이슈를 앞당겨 배정했습니다."
            changes.append({"phase": event["phase"], "turn": event["turn"], "reason": reason,
                "categories": categories, "category_source": category_source, **correction,
                "changed_fields": sorted(PLAN_FIELDS & data.keys())})
        rejected = [{"phase": e["phase"], "turn": e["turn"], **e["payload"]}
                    for e in events if e["kind"] == "operation_rejected"]
        selected_calls = [c for c in calls if c.get("essay_id") == result["essay_id"]]
        metrics[result["essay_id"]] = {
            "turns": len(result["turns"]), "api_calls": sum(c.get("dispatched", True) for c in selected_calls),
            "calls_by_role": dict(Counter(c["role"] for c in selected_calls if c.get("dispatched", True))),
            "initial_plan_rejections": [e["payload"] for e in events if e["kind"] == "initial_plan_rejected"],
            "plan_changes": len(changes), "plan_change_details": changes,
            "plan_changes_initial": sum(c["phase"] == "baseline" for c in changes),
            "plan_changes_during_turns": sum(c["phase"] != "baseline" for c in changes),
            "plan_change_reasons": dict(Counter(c for change in changes for c in set(change["categories"]))),
            "rejected_operations": rejected, "rejected_operation_events": len(rejected),
            "rejected_operation_attempts": sum("operation" in r for r in rejected),
            "operation_rejection_reasons": dict(Counter(r["reason_code"] for r in rejected)),
            "original_tests_run": "baseline_tests" in result,
            "baseline_tests": result.get("baseline_tests"),
            "excluded_initial_issues": result.get("excluded_initial_issues"),
            "student_tasks": result["student_tasks"], "student_task_count": len(result["student_tasks"]),
            "student_task_duplicates_removed": sum(e["kind"] == "student_task_deduplicated" for e in events),
            "deferred_micro_feedback": result.get("deferred_micro_feedback", []),
            "macro_gate_blockers": result.get("macro_gate_blockers", []),
            "route": result["route"], "stop_reason": result["route_reason"],
            "review_method": result.get("review_method", "legacy_tests"),
            "observational_flags": dict(Counter(f["kind"] for t in result["turns"] for f in t.get("observational_flags", []))),
        }
    return metrics


def compare_runs(directory, current):
    path = Path(directory) / "comparison_reference.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    previous = essay_metrics(reference["baseline_run"])
    selection = json.loads((Path(directory) / "selection.json").read_text())
    rows = []
    for essay in selection["essays"]:
        identifier = essay["essay_id"]
        before = previous.get(identifier) if identifier in reference["repeated_ids"] else None
        after = current.get(identifier)
        rows.append({"essay_id": identifier, "cohort": essay.get("cohort"), "stratum": essay["stratum"],
            "genre": essay.get("genre"), "before": before, "after": after,
            "delta": {key: after[key] - before[key] for key in ("turns", "api_calls", "plan_changes", "student_task_count")}
                if before is not None and after is not None else None})
    comparison = {"baseline_run": reference["baseline_run"], "essays": rows,
        "notes": ["원문 테스트 수행 여부는 실행별 original_tests_run을 따릅니다. 새 전체 글 수정 방식에는 코드 테스트·연산 거부가 없습니다.",
                  "이전 계획 변경의 원인 분류는 저장 사건에서 추론했으며 원래 이유를 함께 보존했습니다.",
                  "해결 사건 전의 빈 턴 배정을 해결로 잘못 분류한 기록은 사건 순서로 정정하며 원래 분류와 이유도 보존합니다.",
                  "계획 변경 한 건에 여러 원인이 있을 수 있습니다. 최초 계획 생성은 세지 않으며, 초기 변경은 원문 검사·과제 분리·관문 종료, 턴 중 변경은 수정 진행 중의 조정입니다."]}
    save_json(Path(directory) / "comparison.json", comparison)
    return comparison


def render(directory):
    directory = Path(directory)
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("essays/*/result.json"))]
    calls_path = directory / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()] if calls_path.exists() else []
    events = [json.loads(line) for p in directory.glob("essays/*/events.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
    rejected = [e["payload"] for e in events if e["kind"] == "operation_rejected"]
    metrics = essay_metrics(directory)
    comparison = compare_runs(directory, metrics)
    summary = {"completed_essays": len(results), "api_calls": sum(c.get("dispatched", True) for c in calls),
        "mock_essays": sum(bool(r.get("mock")) for r in results),
        "request_attempts": len(calls),
        "confirmed_cost_usd": sum(c.get("cost_usd") or 0 for c in calls),
        "route_distribution": dict(Counter(r["route"] for r in results)),
        "initial_strategy_distribution": dict(Counter((r["initial_strategy"] or {}).get("name") for r in results)),
        "plan_changes": sum(m["plan_changes"] for m in metrics.values()),
        "operation_rejection_reasons": dict(Counter(r["reason_code"] for r in rejected)),
        "preserve_violation_attempts": sum(bool(r.get("preserve_violation_attempt")) for r in rejected),
        "test_trajectories": {r["essay_id"]: [t.get("test_summary") for t in r["turns"]] for r in results if "baseline_tests" in r},
        "observational_flags": dict(Counter(f["kind"] for r in results for t in r["turns"] for f in t.get("observational_flags", [])))}
    summary["essay_metrics"] = metrics
    continuation = directory / "continuation.json"
    if continuation.exists():
        saved = json.loads(continuation.read_text())
        summary["continuation"] = {key: saved[key] for key in (
            "parent_run", "prior_attempts", "reused_ids", "rerun_ids", "reused_calls",
            "reused_confirmed_cost_usd", "note")}
        summary["new_api_calls"] = summary["api_calls"] - saved["reused_calls"]
        summary["new_confirmed_cost_usd"] = summary["confirmed_cost_usd"] - saved["reused_confirmed_cost_usd"]
        prior_calls = [json.loads(line) for run in saved["prior_attempts"]
                       for line in (Path(run) / "calls.jsonl").read_text().splitlines()]
        summary["all_attempts_api_calls"] = sum(c.get("dispatched", True) for c in prior_calls) + summary["new_api_calls"]
        summary["all_attempts_confirmed_cost_usd"] = sum(c.get("cost_usd") or 0 for c in prior_calls) + summary["new_confirmed_cost_usd"]
    budget_path = directory / "budget.json"
    reference = directory / "budget_reference.json"
    if reference.exists():
        summary["budget_reference"] = json.loads(reference.read_text())
        budget_path = Path(summary["budget_reference"]["ledger"])
    if budget_path.exists():
        summary["budget"] = json.loads(budget_path.read_text())
    errors = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("**/*.error.json")]
    preflight_path = directory / "preflight.json"
    if preflight_path.exists() and json.loads(preflight_path.read_text()).get("cuda_available"):
        errors = [{**e, "resolved": True} if e.get("component") == "native_kanana_gpu" else e for e in errors]
    summary["anomalies"] = [e for e in errors if not e.get("resolved")]
    summary["resolved_anomalies"] = [e for e in errors if e.get("resolved")]
    summary["essay_costs"] = {r["essay_id"]: {
        "confirmed_cost_usd": sum(c.get("cost_usd") or 0 for c in calls if c.get("essay_id") == r["essay_id"]),
        "api_calls": sum(c.get("dispatched", True) for c in calls if c.get("essay_id") == r["essay_id"])} for r in results}
    selection_path = directory / "selection.json"
    if selection_path.exists():
        summary["selected_essays"] = [{k: e[k] for k in ("essay_id", "stratum", "human_original_total", "genre", "cohort") if k in e}
            for e in json.loads(selection_path.read_text())["essays"]]
    save_json(directory / "summary.json", summary)
    parts = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>단계별 피드백 검증 보고</title>",
        "<style>body{max-width:1100px;margin:40px auto;font:16px/1.7 sans-serif;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:15px}article{border-top:2px solid #aaa;margin-top:40px}details{margin:14px 0}li{margin:8px 0}</style>",
        "<h1>단계별 피드백 검증 보고</h1><p>수정본은 학생이 피드백을 반영한 글의 시뮬레이션입니다. 이 실행은 학습 효과나 품질 향상을 입증하지 않습니다.</p>",
        "<h2>실행 요약</h2>",
        f"<p>최종 결과 {summary['completed_essays']}편 · 해당 결과의 API {summary['api_calls']}회 · 확인 비용 ${summary['confirmed_cost_usd']:.6f}</p>",
        json_block({key: summary[key] for key in ("budget", "anomalies") if key in summary})]
    if "continuation" in summary:
        parts.extend([f"<p>초기 검증·실패 시도 포함 전체: API {summary['all_attempts_api_calls']}회, 확인 비용 ${summary['all_attempts_confirmed_cost_usd']:.6f}. 완료된 {len(summary['continuation']['reused_ids'])}편은 재호출 없이 이어받았습니다.</p>",
            "<details><summary>결과 이어받기 출처</summary>", json_block(summary["continuation"]), "</details>"])
    if comparison:
        parts.extend(["<h2>이전 실행과 비교</h2><table border='1' cellpadding='7'><tr><th>표본</th><th>종류·사람 점수층</th><th>턴 수</th><th>API 호출</th><th>계획 변경: 합계 (초기 / 턴 중)</th></tr>"])
        for row in comparison["essays"]:
            before, after = row["before"], row["after"]
            cells = [row["essay_id"] + (" (기존)" if before else " (새 표본)"),
                     str(row["genre"]) + " / " + row["stratum"]]
            cells.extend((str(before[key]) + " → " if before else "") +
                         (str(after[key]) if after else "미완료") for key in ("turns", "api_calls"))
            def plan_label(metric):
                return (f"{metric['plan_changes']} ({metric['plan_changes_initial']} / {metric['plan_changes_during_turns']})"
                        if metric else "미완료")
            cells.append((plan_label(before) + " → " if before else "") + plan_label(after))
            parts.append("<tr>" + "".join("<td>" + escape(cell) + "</td>" for cell in cells) + "</tr>")
        parts.extend(["</table>", json_block(comparison["notes"])])
    parts.extend(["<details><summary>전체 실행 요약 원자료</summary>", json_block(summary), "</details>"])
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
        metric = metrics[result["essay_id"]]
        parts.extend(["<h3>원문 사전 테스트와 제외 이슈</h3>", json_block({
            "performed": metric["original_tests_run"], "tests": metric["baseline_tests"],
            "excluded": metric["excluded_initial_issues"]}), "<h3>계획 변경과 이유</h3><ul>"])
        for change in metric["plan_change_details"]:
            label = f"{change['phase']} / {change['turn']}턴: " + change["reason"]
            label += " (" + ", ".join(REASONS.get(c, c) for c in change["categories"]) + ")"
            parts.append("<li>" + escape(label) + "</li>")
        parts.extend(["</ul><h3>거부된 연산과 이유</h3>", json_block(metric["rejected_operations"])])
        if metric["initial_plan_rejections"]:
            parts.extend(["<h3>초기 진단 검증 오류와 재시도</h3>", json_block(metric["initial_plan_rejections"])])
        for turn in result["turns"]:
            if result.get("revision_mode") == "full_text":
                parts.extend([f"<h3>{turn['turn']}턴 · 전체 글 수정</h3>", "<h4>피드백</h4>", json_block(turn.get("feedback")),
                    "<h4>바뀐 문장</h4>", json_block(turn.get("changed_sentences", [])),
                    "<h4>관찰용 표시 (거부·되돌리기 없음)</h4>", json_block(turn.get("observational_flags", [])),
                    "<h4>총괄의 직접 검토</h4>", json_block(turn.get("review"))])
                continue
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
            "<h3>학생 과제</h3>"])
        if not result["student_tasks"]:
            parts.append("<p>없음</p>")
        for task in result["student_tasks"]:
            parts.extend(["<p><strong>" + escape(task.get("issue_id") or "학생 과제") + "</strong>: " +
                escape(task["instruction"]) + "<br>이유: " + escape(task["reason"]) + "</p>"])
        parts.append("<h3>큰 틀을 고친 뒤 다룰 미시 피드백</h3>")
        deferred = result.get("deferred_micro_feedback", [])
        if not deferred:
            parts.append("<p>별도로 미룬 미시 이슈 없음</p>")
        for item in deferred:
            parts.append("<p><strong>" + escape(item["issue_id"]) + "</strong>: " + escape(item["feedback"]) + "</p>")
        parts.extend([
            "<details><summary>역할별 메모리 요약 예시</summary>", json_block(result["memory_summaries"][:8]),
            "</details><h3>최종본</h3><pre>", escape(result["final_text"]), "</pre></article>"])
    parts.append("</html>")
    (directory / "report.html").write_text("\n".join(parts), encoding="utf-8")
    return summary
