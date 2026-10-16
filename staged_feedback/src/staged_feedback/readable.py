"""Offline Markdown narratives; historical inputs are read without modification."""
import hashlib
import json
from pathlib import Path
import re

from .events import save_json
from .observations import Observer
from .report import essay_metrics, REASONS

LABELS = {"easy": "낮음", "moderate": "중간", "hard": "높음", "small": "적음", "large": "많음",
    "macro": "거시", "micro": "미시", "front_first": "앞에서부터", "macro_first": "거시부터", "single_pass": "한 턴에 전체",
    "applied": "반영", "partial": "일부", "not_applied": "미반영", "pass": "통과", "fail": "미통과", "error": "검사 오류"}
FLAGS = {"student_sentence_deleted": "학생 문장 통째 삭제 관찰", "earlier_revision_changed": "앞 턴에서 고친 문장 변경",
    "anonymization_changed": "비식별 표시 변경", "outside_feedback_target": "피드백 대상 밖 변경"}
ROLES = {"coordinator_initial": "총괄 초기 판단", "feedback": "피드백", "reviser": "수정", "coordinator_review": "총괄 검토",
         "macro_tests": "기존 거시 테스트", "single_call_staged": "한 호출 비교"}


def sentence_spans(text):
    """Native punctuation/newline algorithm, usable offline without GPU imports."""
    boundaries = [m.end() for m in re.finditer(r'[.!?。！？][\"\'”’」』)\]]*(?=\s|$)|\r?\n', text)]
    spans, start = [], 0
    for end in [*boundaries, len(text)]:
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while a < b and text[b-1].isspace():
            b -= 1
        if a < b:
            spans.append({"id": f"s{len(spans)+1}", "text": text[a:b], "start": a, "end": b})
        start = end
    return spans


def observer():
    return Observer(sentence_spans=sentence_spans, marker_pattern=re.compile(r"#@[^#\r\n]+#"),
        placeholder_pattern=re.compile(r"__ANON_[A-Za-z0-9]+_[A-Za-z0-9]+__"))


def text(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def quoted(value):
    return "“" + text(value) + "”" if value else "(없음)"


def changes_text(changes):
    return [f"- 전: {quoted(c['before'])} → 후: {quoted(c['after'])}" for c in changes] or ["바뀐 문장 없음."]


def issue_text(key, issues):
    return f"{key}: {text(issues[key]['description'])}" if key in issues else str(key)


def essay_log(directory, essay, metric, load):
    path = directory / "essays" / essay["essay_id"]
    result, initial = load(path / "result.json"), load(path / "initial_plan.json", {})
    events = load(path / "events.jsonl", [])
    issues = {i["id"]: i for i in initial.get("issues", [])}
    issues.update({e["payload"]["issue"]["id"]: e["payload"]["issue"] for e in events if e["kind"] == "issue_added"})
    observe, lines = observer(), []

    def add(*paragraphs):
        for paragraph in paragraphs:
            lines.extend([paragraph, ""])

    def listing(items, empty="없음."):
        lines.extend(list(items) or [empty])
        lines.append("")

    add(f"# {essay['essay_id']}", "## [먼저 확인할 학생 과제]")
    listing(f"- {t.get('issue_id') or '별도 과제'}: {text(t['instruction'])} 이유: {text(t['reason'])}" for t in result.get("student_tasks", []))
    add(f"글 종류: {essay.get('genre', '미기록')} · 사람 원점수: {essay.get('human_original_total', '미기록')} · 점수층: {essay.get('stratum', '미기록')}", "## [총괄 초기 판단]")
    difficulty, strategy = initial.get("difficulty", {}), initial.get("strategy", {})
    add(f"난이도: {LABELS.get(difficulty.get('level'), '미기록')} · 수정량: {LABELS.get(difficulty.get('revision_amount'), '미기록')}.",
        f"판단 이유: {text(difficulty.get('reason', '미기록'))}",
        f"전략: {LABELS.get(strategy.get('name'), strategy.get('name', '미기록'))}. 이유: {text(strategy.get('reason', '미기록'))}",
        f"초기 계획: {initial.get('remaining_turns', initial.get('turn_budget', '미기록'))}턴. 실제 피드백 턴: {len(result['turns'])}턴. 계획 이유: {text(initial.get('reason', '미기록'))}")
    if difficulty.get("diagnostic_evidence"):
        add("판단에 사용한 진단 근거:")
        listing(f"- {e['key']} = {e.get('value')}: {text(e['explanation'])}" for e in difficulty["diagnostic_evidence"])
    add("초기 이슈 목록:")
    listing(f"- {i['id']} / {LABELS[i['level']]}" + (f" ({i['macro_type']})" if i.get('macro_type') else '') +
        f" / {'학생 과제' if i.get('needs_student_content') or i.get('macro_type') == 'student_input' else '수정 이슈'}: {text(i['description'])}" +
        (f" 재배열 대기: {', '.join(i['restructure_issue_ids'])}." if i.get('restructure_issue_ids') else '') for i in initial.get("issues", []))
    add("턴 계획:")
    listing(f"- {p['turn']}턴: {text(p['direction'])} 대상 {', '.join(p['issue_ids']) or '없음'}; 피드백 {str(p['feedback_count'])+'개' if 'feedback_count' in p else '계획에 개수 미기록'}." for p in initial.get("turn_plan", []))
    add("원문 테스트 결과:")
    if "baseline_tests" in result:
        listing(f"- {t['issue_id']}: {LABELS[t['status']]}. {text(t['reason'])} 확인 항목: {text(t['checklist'])}" for t in result["baseline_tests"])
        tested = {t["issue_id"] for t in result["baseline_tests"]}
        for i in initial.get("issues", []):
            if i["id"] not in tested:
                add(f"- {i['id']}: 테스트 없음. 학생 의도 확인 과제는 자동 판정하지 않았습니다.")
        add("원문에서 이미 통과하여 제외:")
        listing("- " + issue_text(x["issue"]["id"], issues) for x in result.get("excluded_initial_issues", []))
    else:
        add("실시하지 않음. 이 실행은 코드 테스트 도구를 제거하고 총괄이 직접 판단합니다." if result.get("review_method") == "coordinator" else "이 실행에는 원문 사전 테스트 기록이 없습니다.")
    if metric.get("initial_plan_rejections"):
        add(f"초기 응답 형식 검사 실패: {len(metric['initial_plan_rejections'])}회. 당시 형식 오류로 초기 판단을 다시 요청했습니다. JSON 원문과 긴 오류 덤프는 이 로그에 싣지 않았습니다.")
    add("초기 판단 뒤 계획 변경:")
    initial_changes = [e["payload"] for e in events if e["kind"] == "plan_changed" and e["phase"] == "baseline"]
    listing("- " + (f"턴 수 {d['turn_budget']['before']} → {d['turn_budget']['after']}. " if 'turn_budget' in d else "") + text(d.get("reason", "미기록")) for d in initial_changes)
    previous, budget = result["initial_text"], initial.get("turn_budget")
    for turn in result["turns"]:
        number = turn["turn"]
        feedback = turn.get("feedback", [])
        feedback = feedback.get("items", []) if isinstance(feedback, dict) else feedback
        items = [i for i in feedback if isinstance(i, dict)]
        add(f"## [{number}턴]", f"### 피드백: {len(feedback)}개")
        if "feedback_count_planned" in turn:
            add(f"총괄 계획: {turn['feedback_count_planned']}개. 실제 제시: {len(feedback)}개.")
        targets, _ = observe.targets(previous, items)
        for item in feedback:
            if isinstance(item, str):
                add("- " + text(item))
                continue
            key = item["issue_id"]
            add(f"**{key}**", "대상 문장: " + (" / ".join(quoted(s["text"]) for s in targets.get(key, [])) or "인용에 대응하는 문장을 찾지 못했습니다."),
                f"가리킨 부분: {quoted(item['target_quote'])}", f"무엇을: {text(item['what'])}", f"어떻게: {text(item['how'])}")
            if item.get("student_explanation"):
                add(f"학생에게 설명: {text(item['student_explanation'])}")
            if item.get("target_ids"):
                add("대상 범위 ID: " + ", ".join(item["target_ids"]))
            if item.get("needs_student_input"):
                add("학생 입력 필요 표시: 총괄에게 과제 여부 판단을 요청했습니다.")
        if "scope_revisions" in turn:
            add("### 범위별 수정")
            for scope in turn["scope_revisions"]:
                add(f"- {scope['scope_id']} ({', '.join(scope['issue_ids'])}): 전 {quoted(scope['before'])} → 후 {quoted(scope['after'])}")
                if scope.get("cannot_apply"):
                    add("수정 에이전트의 반영 불가 안내: " + text(scope["cannot_apply"]) + " 해당 범위 원문을 유지했습니다.")
            for missing in turn.get("scope_unavailable", []):
                add(f"- {missing['issue_id']}: 범위 지정 미완료. {text(missing['reason'])}")
        if "operations" in turn:
            add("### 수정: 적용 및 거부")
            if not turn["operations"]:
                add("제출된 수정 연산 없음.")
            for operation in turn["operations"]:
                op = operation.get("operation", {})
                label = op.get("id", operation.get("operation_id", "ID 미기록"))
                if operation.get("applied"):
                    add(f"적용 {label}: {text(operation.get('description', '수정 적용'))}")
                    if "text_before" in operation and "text_after" in operation:
                        listing(changes_text(observe.compare(operation["text_before"], operation["text_after"])["changed_sentences"]))
                    else:
                        add(f"전: {quoted(operation.get('before', op.get('quote')))} → 후: {quoted(op.get('replacement'))}")
                else:
                    add(f"거부 {label}{' (재시도)' if operation.get('retry') else ''}: {text(operation.get('reason', '이유 미기록'))} ({operation.get('reason_code', '코드 미기록')})")
                    if op:
                        proposed = op.get("replacement") or " / ".join(op.get("sentences") or []) or op.get("connector")
                        add(f"제안 전: {quoted(op.get('quote') or op.get('sentence_a'))} → 제안 후: {quoted(proposed)}. 실제 글에는 적용되지 않았습니다.")
        add("### 이번 턴 수정본에서 바뀐 문장")
        changes = turn.get("changed_sentences")
        if changes is None:
            changes = observe.compare(previous, turn["revised_text"])["changed_sentences"]
        listing(changes_text(changes))
        for gap in turn.get("layout_changes", []):
            add(f"줄바꿈·공백: 전 {quoted(gap['before'])} → 후 {quoted(gap['after'])}")
        add("### 관찰용 표시")
        if "observational_flags" not in turn:
            add("기존 실행에는 관찰 방식이 없었습니다. 당시 수정 거부 기록은 위에 표시했습니다.")
        elif not turn["observational_flags"]:
            add("없음. 수정본은 그대로 총괄 검토에 전달했습니다.")
        else:
            for f in turn["observational_flags"]:
                add(f"- {FLAGS.get(f['kind'], f['kind'])}: {text(f['reason'])}" + (f" 대상 {quoted(f['sentence'])}" if f.get("sentence") else ""))
                if f["kind"] == "anonymization_changed":
                    add("표시 순서: " + quoted(" / ".join(f["before"])) + " → " + quoted(" / ".join(f["after"])))
            add("이 표시는 수정본을 거부하거나 이전 글로 되돌리지 않았습니다.")
        add("### 총괄 검토")
        review = turn.get("review")
        if not review:
            add("호출하지 않음. " + turn.get("stop_reason", "해당 조건에는 별도 총괄 검토가 없습니다."))
        else:
            assessments = {a["issue_id"]: a for a in review.get("assessments", [])}
            for item in items:
                key = item["issue_id"]
                if key in assessments:
                    a = assessments[key]
                    add(f"- {key}: {LABELS[a['status']]}. 이유: {text(a['reason'])}")
                elif "assessments" in review:
                    add(f"- {key}: 총괄의 항목별 판단 누락. 코드가 반영으로 보충하지 않았습니다.")
                else:
                    label = "반영(해결로 지정)" if key in review.get("resolved_issue_ids", []) else "해결로 지정하지 않음"
                    disposition = next((d for d in review.get("dispositions", []) if d["issue_id"] == key), None)
                    add(f"- {key}: 총괄은 {label}. " + ("이유: " + text(disposition["reason"]) if disposition else "항목별 이유 필드는 당시 스키마에 없었습니다. 아래 전체 이유를 참고하세요."))
            if turn.get("tests"):
                add("당시 코드·거시 테스트의 별도 판정:")
                listing(f"- {t['issue_id']}: {LABELS[t['status']]}. {text(t['reason'])}" for t in turn["tests"])
            add("전체 검토 이유: " + text(review["reason"]), "새 피드백:")
            listing(f"- {i['id']} / {LABELS[i['level']]}: {text(i['description'])} 근거: {' / '.join(quoted(q) for q in i['evidence_quotes'])}" for i in review.get("new_issues", []))
            for d in review.get("dispositions", []):
                action = {"retry": "다음 턴 재수정", "defer": "보류", "student_task": "학생 과제"}.get(d["action"], d["action"])
                add(f"후속 처리 {d['issue_id']}: {action}. {text(d['reason'])}")
        turn_changes = [e["payload"] for e in events if e["kind"] == "plan_changed" and e["turn"] == number and e["phase"] != "initial"]
        old_budget = turn.get("budget_before", budget)
        for d in turn_changes:
            if "turn_budget" in d:
                budget = d["turn_budget"]["after"]
        budget = turn.get("budget_after", budget)
        if "remaining_after_review" in turn:
            add(f"남은 턴 수: 턴 시작 전 {turn['remaining_before']} → 이번 턴 사용 후 {turn['remaining_after_consumption']} → 총괄 검토 후 {turn['remaining_after_review']}. 이유: " + text(review["reason"]))
            if review["continue_work"] and review["remaining_turns"] == 0:
                add("총괄 출력 불일치: 계속 필요(continue_work=true)라고 했지만 남은 턴 수를 0으로 반환했습니다. 실행은 남은 턴 수 0에 따라 종료됐으며 미해결 이슈는 그대로 남습니다.")
        else:
            add(f"턴 조절: {old_budget} → {budget}. " + (text(review["reason"]) if review else "학생 과제 분리 또는 실행 조건에 따라 종료했습니다."))
        for d in turn_changes:
            corrected = next((c for c in metric["plan_change_details"] if c["turn"] == number and
                c.get("category_source") == "corrected_from_events" and c.get("original_reason") == d.get("reason")), None)
            categories = corrected["categories"] if corrected else d.get("reason_categories", [])
            reason = corrected["reason"] if corrected else d.get("reason", "미기록")
            causes = ", ".join(REASONS.get(c, c) for c in categories)
            add(f"- 계획 변경: {text(reason)}" + (f" 원인: {causes}." if causes else ""))
            if d.get("rejected_new_issues"):
                listing(f"- 당시 코드가 새 이슈 등록을 거부: {i['id']}. {text(i['reason'])}" for i in d["rejected_new_issues"])
        if "restore_results" in turn:
            add("### 총괄 지시에 따른 문장 복원")
            for restored in turn["restore_results"]:
                if restored["status"] == "applied":
                    add(f"- 이전 문장 {', '.join(restored['previous_sentence_ids'])}: 전 {quoted(restored['before'])} → 후 {quoted(restored['after'])}. 이유: {text(restored['reason'])}")
                else:
                    add(f"- 복원 {restored['status']}: {text(restored['detail'])}. 총괄 이유: {text(restored['reason'])}")
            if not turn["restore_results"]:
                add("복원 지시 없음.")
            add("복원까지 반영한 이번 턴의 최종 문장 변화:")
            listing(changes_text(turn["final_changed_sentences"]))
        continuing = turn.get("continue_work", number < len(result["turns"]))
        add("계속·종료: " + ("다음 턴 계속." if continuing else "종료. " + result["route_reason"]))
        if review and bool(review["continue_work"]) != continuing:
            add("총괄의 계속 요청과 실제 종료가 다릅니다. 실제 남은 턴 수·수정 가능한 이슈 상태에 따라 종료됐습니다." if "remaining_after_review" in turn else
                "총괄의 계속 요청과 실제 종료가 다릅니다. 실제 턴 예산·학생 과제·남은 이슈 상태에 따라 종료됐습니다.")
        previous = turn["revised_text"]
    add("## [결과]", "종료 이유: " + text(result["route_reason"]))
    resolved = set(result.get("resolved_issue_ids", []))
    if "resolved_issue_ids" not in result:
        for e in events:
            if e["kind"] == "issue_resolved":
                resolved.add(e["payload"]["issue_id"])
            elif e["kind"] == "plan_changed":
                resolved.difference_update(e["payload"].get("reopened_issue_ids", []))
    for label, ids in (("해결 이슈", sorted(resolved)), ("미해결 수정 이슈", result.get("remaining_issue_ids", []))):
        add(label + ":")
        listing("- " + issue_text(key, issues) for key in ids)
    add("학생 과제:")
    listing(f"- {t.get('issue_id') or '별도 과제'}: {text(t['instruction'])} 이유: {text(t['reason'])}" for t in result.get("student_tasks", []))
    add("큰 틀을 고친 뒤 다룰 미시 피드백:")
    listing(f"- {i['issue_id']}: {text(i['feedback'])}" for i in result.get("deferred_micro_feedback", []))
    add(f"호출 수: API {metric['api_calls']}회. " + ", ".join(f"{ROLES.get(k, k)} {v}회" for k, v in metric["calls_by_role"].items()) + ".",
        f"기존 N32 진단은 초기·최종 {result.get('diagnostic_calls', '미기록')}회이며 위 API 호출 수와 별도입니다.",
        f"계획 변경: {metric['plan_changes']}회(초기 분리·종료 {metric['plan_changes_initial']}회, 턴 중 {metric['plan_changes_during_turns']}회). 최초 계획 생성은 제외합니다.")
    for c in metric["plan_change_details"]:
        add(f"- {c['turn']}턴 / {', '.join(REASONS.get(x, x) for x in c['categories'])}: {text(c['reason'])}")
        if c.get("category_source") == "corrected_from_events":
            add("원자료의 ‘해결된 문제’ 분류는 앞선 해결 사건이 없어 보고 단계에서 정정했습니다. 원자료는 보존했습니다.")
    add(f"거부된 연산: {metric['rejected_operation_attempts']}개; 거부 관련 사건 {metric['rejected_operation_events']}건. " +
        ("이 실행에는 편집 연산·자동 거부 기능이 없습니다." if result.get("revision_mode") in {"full_text", "scoped"} else "재시도에서 빠뜨린 연산은 별도 사건으로 셉니다."))
    add("## [선택 제안]")
    listing("- " + text(s) for s in result.get("optional_suggestions", []))
    return "\n".join(lines).rstrip() + "\n"


def write_logs(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def load(path, default=None):
        path = Path(path)
        if not path.exists():
            return default
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(raw)

    selection, calls = load(directory / "selection.json", {"essays": []}), load(directory / "calls.jsonl", [])
    for path in directory.glob("essays/*/events.jsonl"):
        load(path)
    metrics = essay_metrics(directory)
    essays = selection["essays"] or [{"essay_id": key} for key in metrics]
    lines = [f"# {directory.name} 작업 로그", "", "저장된 초기 판단·턴 결과·사건·호출 기록을 문장으로 옮겼습니다. 로그 작성에는 추가 모델 호출이 없습니다. 문장 차이는 저장된 문자열에서 계산했습니다.", "",
        "| 글 | 종류·사람 점수 | 턴 | API 호출 | 계획 변경 | 학생 과제 |", "|---|---|---:|---:|---:|---:|"]
    documents = []
    for essay in essays:
        key = essay["essay_id"]
        if key not in metrics:
            lines.append(f"| {key} | 미완료 | — | — | — | — |")
            continue
        m = metrics[key]
        doc = essay_log(directory, essay, m, load)
        (destination / f"{key}.md").write_text(doc, encoding="utf-8")
        documents.append(doc)
        lines.append(f"| [{key}]({key}.md) | {essay.get('genre', '미기록')} · {essay.get('human_original_total', '미기록')} | {m['turns']} | {m['api_calls']} | {m['plan_changes']} | {m['student_task_count']} |")
    cost = sum(c.get("cost_usd") or 0 for c in calls)
    lines.extend(["", f"최종 결과에 연결된 API: {sum(c.get('dispatched', True) for c in calls)}회. 확인 비용: ${cost:.8f}.", ""])
    continuation = load(directory / "continuation.json")
    if continuation:
        summary = load(directory / "summary.json", {})
        lines.extend([f"이전 실패·재시도 포함 당시 실험 전체: API {summary.get('all_attempts_api_calls', '미기록')}회, 확인 비용 ${summary.get('all_attempts_confirmed_cost_usd', 0):.8f}. 완료된 {len(continuation['reused_ids'])}편은 기존 결과를 이어받았습니다.", ""])
    comparison = load(directory / "comparison.json")
    if comparison:
        lines.extend(["## 이전 실행과 비교", "", "| 글 | 턴 수 | API 호출 | 계획 변경 |", "|---|---:|---:|---:|"])
        for row in comparison["essays"]:
            b, a = row.get("before"), row.get("after")
            if a:
                cells = [f"{b[k]} → {a[k]}" if b else f"새 표본 → {a[k]}" for k in ("turns", "api_calls", "plan_changes")]
                lines.append(f"| {row['essay_id']} | " + " | ".join(cells) + " |")
        lines.extend(["", "계획 변경 이유와 학생 과제는 각 글의 결과에 있습니다. 거부·원문 통과 기록이 없는 새 방식은 미실시로 표시합니다.", ""])
    for document in documents:
        lines.extend(["---", "", re.sub(r"(?m)^(#+) ", r"#\1 ", document), ""])
    combined = destination / "WORK_LOG.md"
    combined.write_text("\n".join(lines), encoding="utf-8")
    unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    save_json(destination / "provenance.json", {"source_run": str(directory), "source_sha256": hashes,
        "source_unchanged": unchanged, "model_calls_for_log": 0,
        "note": "Markdown에는 원시 JSON을 넣지 않았습니다. 관찰은 의미 보존의 자동 판정이 아닙니다."})
    return str(combined)
