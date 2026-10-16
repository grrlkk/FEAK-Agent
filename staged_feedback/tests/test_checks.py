import pytest

from staged_feedback.checks import run_micro, run_tests
from staged_feedback.models import Issue, TestSpec as Spec
from conftest import sentences


@pytest.mark.parametrize("spec,text,original,passed", [
    (dict(kind="absent", expression="매우"), "조금 좋다.", "매우 좋다.", True),
    (dict(kind="absent", expression="매우"), "매우 좋다.", "매우 좋다.", False),
    (dict(kind="count_at_most", expression="매우", limit=1), "매우 매우 좋다.", "", False),
    (dict(kind="count_at_most", expression="매우", limit=1), "매우 좋다.", "", True),
    (dict(kind="span_replaced", source_quote="읽을수", expected="읽을 수"), "책을 읽을 수 있다.", "책을 읽을수 있다.", True),
    (dict(kind="span_replaced", source_quote="읽을수", expected="읽을 수"), "책을 읽을수 있다.", "책을 읽을수 있다.", False),
    (dict(kind="sentence_contains", sentence_quote="걷는다.", expression="그래서"), "그래서 걷는다.", "", True),
    (dict(kind="sentence_contains", sentence_quote="걷는다.", expression="그래서"), "걷는다. 그래서 웃는다.", "", False),
])
def test_micro_library(spec, text, original, passed):
    actual, _ = run_micro(Spec(checklist="고쳤나요?", **spec), text=text, original=original, sentence_spans=sentences)
    assert actual == passed


def test_replacement_checks_target_not_other_occurrence():
    spec = Spec(kind="span_replaced", source_quote="이것", expected="저것", checklist="해당 구간을 고쳤나요?")
    passed, _ = run_micro(spec, text="이것은 좋다. 저것은 크다.", original="이것은 좋다. 다른 것은 크다.", sentence_spans=sentences)
    assert not passed


def macro(identifier):
    return Issue(id=identifier, level="macro", description="연결을 확인합니다.", priority=1,
        evidence_quotes=["책을 읽는다."], test=dict(kind="macro", target_quotes=["책을 읽는다."],
        question="앞의 주장을 이어 가는가?", checklist="앞의 주장과 이어지나요?"))


class MacroLLM:
    def __init__(self, mode="ok"):
        self.calls, self.mode = [], mode

    def request(self, role, payload, schema, *, context):
        self.calls.append((role, payload))
        results = [dict(issue_id=i["issue_id"], passed=True, reason="주장과 이어집니다.",
                        evidence_quotes=["책을 읽는다."]) for i in payload["tests"]]
        if self.mode == "missing":
            results.pop()
        if self.mode == "duplicate":
            results[-1] = results[0]
        if self.mode == "foreign_quote":
            results[0]["evidence_quotes"] = ["원문에 없는 문장."]
        return schema.model_validate(dict(results=results))


@pytest.mark.parametrize("mode,status", [("ok", "pass"), ("missing", "error"), ("duplicate", "error"), ("foreign_quote", "error")])
def test_macro_batched_once_and_requires_complete_evidence(mode, status):
    llm = MacroLLM(mode)
    results = run_tests([macro("a"), macro("b")], text="책을 읽는다.", original="책을 읽는다.",
        sentence_spans=sentences, llm=llm, context={})
    assert len(llm.calls) == 1
    assert {r.status for r in results} == {status}


def test_missing_sentence_is_error():
    issue = Issue(id="i", level="micro", description="표현 확인", evidence_quotes=["없다"], priority=1,
        test=dict(kind="sentence_contains", sentence_quote="없다", expression="따라서", checklist="연결했나요?"))
    result = run_tests([issue], text="책을 읽는다.", original="책을 읽는다.", sentence_spans=sentences, llm=None, context={})
    assert result[0].status == "error"
    assert not result[0].passed
