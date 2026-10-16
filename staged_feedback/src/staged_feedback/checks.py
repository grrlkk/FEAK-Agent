"""Small deterministic test library; all macro tests share one LLM call."""
from difflib import SequenceMatcher

from .edits import compact, quote_spans, unique_span
from .models import MacroVerdicts, TestResult


def _replacement_span(original, current, source_quote):
    a, b = unique_span(original, source_quote)
    ranges = []
    for kind, x1, x2, y1, y2 in SequenceMatcher(None, original, current, autojunk=False).get_opcodes():
        if kind == "equal" and x1 < b and a < x2:
            ranges.append((y1 + max(a, x1) - x1, y1 + min(b, x2) - x1))
        elif kind != "equal" and x1 < b and a < x2:
            if x1 < a or x2 > b:
                raise ValueError("대상 밖 변경과 섞여 표기 교체 범위를 확정하지 못했습니다.")
            ranges.append((y1, y2))
        elif kind == "insert" and a < x1 < b:
            ranges.append((y1, y2))
    if not ranges:
        return ""
    start, end = min(r[0] for r in ranges), max(r[1] for r in ranges)
    return current[start:end]


def run_micro(test, *, text, original, sentence_spans):
    if test.kind == "absent":
        occurrences = len(quote_spans(text, test.expression))
        return occurrences == 0, f"대상 표현 등장 횟수: {occurrences}"
    if test.kind == "count_at_most":
        occurrences = len(quote_spans(text, test.expression))
        return occurrences <= test.limit, f"등장 횟수 {occurrences}, 허용 상한 {test.limit}"
    if test.kind == "span_replaced":
        replacement = _replacement_span(original, text, test.source_quote)
        return compact(replacement) == compact(test.expected) and replacement == test.expected, f"대상 구간의 현재 표기: {replacement!r}"
    if test.kind == "sentence_contains":
        a, b = unique_span(text, test.sentence_quote)
        sentences = [s["text"] for s in sentence_spans(text) if s["start"] <= a and b <= s["end"]]
        if len(sentences) != 1:
            raise ValueError("테스트 대상 문장을 확정하지 못했습니다.")
        passed = compact(test.expression) in compact(sentences[0])
        return passed, "대상 문장의 지정 표현 포함 여부: " + ("예" if passed else "아니오")
    raise ValueError("미시 테스트 종류가 아닙니다.")


def run_tests(issues, *, text, original, sentence_spans, llm, context, origins=None):
    results, macro = [], []
    for issue in issues:
        if issue.test.kind == "macro":
            macro.append(issue)
            continue
        try:
            passed, reason = run_micro(issue.test, text=text, original=(origins or {}).get(issue.id, original), sentence_spans=sentence_spans)
            status = "pass" if passed else "fail"
        except (ValueError, TypeError) as exc:
            passed, reason, status = False, str(exc), "error"
        results.append(TestResult(issue_id=issue.id, kind=issue.test.kind, passed=passed,
                                  status=status, reason=reason, checklist=issue.test.checklist))
    if macro:
        try:
            verdicts = llm.request("macro_tests", {
                "original_text": original, "current_text": text,
                "tests": [{"issue_id": i.id, "target_quotes": i.test.target_quotes,
                           "question": i.test.question} for i in macro],
            }, MacroVerdicts, context=context)
            requested = {i.id for i in macro}
            observed = [v.issue_id for v in verdicts.results]
            if set(observed) != requested or len(observed) != len(requested):
                raise ValueError("거시 테스트 판정에 누락·중복·알 수 없는 이슈가 있습니다.")
            for verdict in verdicts.results:
                if verdict.passed and (not verdict.evidence_quotes or
                        any(not quote_spans(text, q) for q in verdict.evidence_quotes)):
                    raise ValueError("통과 판정의 근거가 현재 수정본에 없습니다.")
            mapping = {v.issue_id: v for v in verdicts.results}
            for issue in macro:
                verdict = mapping[issue.id]
                results.append(TestResult(issue_id=issue.id, kind="macro", passed=verdict.passed,
                    status="pass" if verdict.passed else "fail", reason=verdict.reason,
                    checklist=issue.test.checklist, evidence_quotes=verdict.evidence_quotes))
        except (ValueError, TypeError) as exc:
            results.extend(TestResult(issue_id=i.id, kind="macro", passed=False, status="error",
                reason=str(exc), checklist=i.test.checklist) for i in macro)
    return results


def test_summary(results):
    return {"all_passed": all(r.passed for r in results), "total": len(results),
            "passed": sum(r.passed for r in results),
            "macro_total": sum(r.kind == "macro" for r in results),
            "macro_passed": sum(r.kind == "macro" and r.passed for r in results),
            "micro_total": sum(r.kind != "macro" for r in results),
            "micro_passed": sum(r.kind != "macro" and r.passed for r in results),
            "errors": sum(r.status == "error" for r in results)}
