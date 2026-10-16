import json

import pytest

from conftest import Diagnostic, ScriptedLLM, issue
from staged_feedback.config import Settings
from staged_feedback.edits import Rejected
from staged_feedback.engine import Engine
from staged_feedback.events import Journal
from staged_feedback.models import Issue, StudentTask
from test_engine import ESSAY, execute


class Intake(ScriptedLLM):
    def __init__(self, *, student=True, passed=False, intent=False):
        super().__init__()
        self.student, self.passed, self.intent = student, passed, intent

    def request(self, role, payload, schema, *, context):
        if role == "macro_tests":
            self.calls.append(dict(role=role, payload=payload, context=context))
            return schema.model_validate(dict(results=[dict(issue_id=t["issue_id"], passed=self.passed,
                reason="현재 글에 제시된 내용만 확인했습니다.", evidence_quotes=["나는 책을 읽을수 있다."])
                for t in payload["tests"]]))
        answer = super().request(role, payload, schema, context=context)
        if role == "coordinator_initial":
            problem = Issue(id="experience", level="macro", priority=1,
                description="실제로 책을 읽었던 상황을 알려 주세요." if not self.intent else "이 표현을 어떤 뜻으로 썼는지 알려 주세요.",
                evidence_quotes=["나는 책을 읽을수 있다."], needs_student_content=self.student,
                student_content_type=("intent" if self.intent else "new_experience") if self.student else None,
                test=None if self.intent else dict(kind="macro", target_quotes=["나는 책을 읽을수 있다."],
                    question="글에 책을 읽은 구체적인 상황이 있는가?", checklist="실제 상황이 글에 있는지 확인하세요."))
            answer.issues.insert(0, problem)
            # Even a mistaken LLM plan must never forward a student issue to revision.
            answer.turn_plan[0].issue_ids.insert(0, problem.id)
            if self.student:
                answer.student_tasks = [StudentTask(issue_id="experience", instruction=problem.description,
                    reason="학생의 경험이나 의도 확인이 필요합니다."),
                    StudentTask(issue_id="experience", instruction="해당 상황과 의도를 직접 설명해 주세요.", reason="다른 표현의 같은 과제입니다.")]
        return answer


@pytest.mark.parametrize("intent", [False, True])
def test_student_macro_gate_stops_before_revision_and_defers_micro(tmp_path, executor, intent):
    result, llm, diagnostic, _ = execute(tmp_path, executor, llm=Intake(intent=intent))
    assert result["route"] == "student_tasks"
    assert result["turns"] == []
    assert result["final_text"] == ESSAY["text"]
    assert result["macro_gate_blockers"] == ["experience"]
    assert result["deferred_micro_feedback"][0]["issue_id"] == "i1"
    assert result["deferred_micro_feedback"][0]["when"] == "큰 틀을 고친 뒤 다룰 것"
    assert len(result["student_tasks"]) == len(result["student_feedback"]) == 1
    assert not any(c["role"] in {"feedback", "reviser", "coordinator_review"} for c in llm.calls)
    assert len(diagnostic.scored) == 2 and not diagnostic.extracted
    journal = Journal(tmp_path / "events.jsonl")
    assert any(e["kind"] == "student_task_deduplicated" for e in journal.events)
    assert not journal.summary("feedback", phase="planned", remaining_turns=0)["candidates"] or all(
        i["id"] != "experience" for i in journal.summary("feedback", phase="planned", remaining_turns=0)["candidates"])
    if intent:
        assert not any(c["role"] == "macro_tests" for c in llm.calls)
        baseline = json.loads((tmp_path / "baseline_tests.json").read_text())
        assert baseline["untestable_student_issue_ids"] == ["experience"]


@pytest.mark.parametrize("student", [False, True])
def test_already_passing_macro_is_removed_before_feedback_even_if_flagged_student(tmp_path, executor, student):
    result, llm, _, _ = execute(tmp_path, executor, llm=Intake(student=student, passed=True))
    assert result["route"] == "single_turn"
    assert [i["issue"]["id"] for i in result["excluded_initial_issues"]] == ["experience"]
    assert not result["student_tasks"] and not result["macro_gate_blockers"]
    assert len([c for c in llm.calls if c["role"] == "macro_tests"]) == 1
    feedback = next(c for c in llm.calls if c["role"] == "feedback")
    assert [i["id"] for i in feedback["payload"]["candidates"]] == ["i1"]


def test_original_micro_pass_is_excluded_and_no_turn_is_spent(tmp_path, executor):
    class AlreadyCorrect(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                answer.issues[0].test.kind = "count_at_most"
                answer.issues[0].test.expression = "읽을수"
                answer.issues[0].test.limit = 1
            return answer
    result, llm, _, _ = execute(tmp_path, executor, llm=AlreadyCorrect())
    assert result["route"] == "no_revision" and not result["turns"]
    assert [c["role"] for c in llm.calls] == ["coordinator_initial"]
    assert result["excluded_initial_issues"][0]["test_result"]["passed"]


@pytest.mark.parametrize("references", [None, [], [""], ["foreign"], ["i1", "foreign"]])
def test_executor_rejects_missing_or_any_foreign_feedback_reference(executor, references):
    with pytest.raises(Rejected) as error:
        executor.apply(ESSAY["text"], dict(id="bad", issue_ids=references, op="replace", quote="읽을수",
            replacement="읽을 수"), allowed_issue_ids={"i1"})
    assert error.value.code == ("missing_feedback_issue" if references in (None, [], [""]) else "outside_feedback")


def test_missing_references_are_logged_per_operation_not_a_response_crash(tmp_path, executor):
    class Unscoped(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "reviser":
                return schema.model_validate({"operations": [dict(id="bad", op="replace", quote="할수",
                    replacement="할 수")], "student_tasks": []})
            return answer
    result, _, _, _ = execute(tmp_path, executor, llm=Unscoped())
    assert result["final_text"] == ESSAY["text"]
    assert [o["reason_code"] for o in result["turns"][0]["operations"]] == ["missing_feedback_issue"] * 2


def test_a_known_but_not_current_issue_is_rejected(tmp_path, executor):
    class Foreign(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                answer.issues.append(issue("later", "할수", "할 수"))
            if role == "reviser":
                answer.operations[0].issue_ids = ["later"]
                answer.operations[0].quote, answer.operations[0].replacement = "할수", "할 수"
            return answer
    result, _, _, _ = execute(tmp_path, executor, llm=Foreign())
    assert result["final_text"] == ESSAY["text"]
    assert all(o["reason_code"] == "outside_feedback" for o in result["turns"][0]["operations"])


def test_student_micro_task_does_not_block_other_editable_micro_issue(tmp_path, executor):
    class Mixed(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                answer.issues.append(Issue(id="word_intent", level="micro", priority=2,
                    description="이 낱말을 어떤 뜻으로 썼는지 알려 주세요.", evidence_quotes=["운동"],
                    needs_student_content=True, student_content_type="intent", test=None))
            return answer
    result, llm, _, _ = execute(tmp_path, executor, llm=Mixed())
    assert result["route"] == "single_turn" and "읽을 수" in result["final_text"]
    assert len(result["student_tasks"]) == 1
    assert all("word_intent" not in str(c["payload"]["feedback"]) for c in llm.calls if c["role"] == "reviser")


def test_new_student_macro_after_review_stops_before_next_turn(tmp_path, executor):
    class NewStudent(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_review":
                answer.new_issues = [Issue(id="intent", level="macro", priority=1,
                    description="글의 중심 의도를 알려 주세요.", evidence_quotes=["우리는 운동을 할수 있다."],
                    needs_student_content=True, student_content_type="intent", test=None)]
            return answer
    result, llm, _, _ = execute(tmp_path, executor, llm=NewStudent(extend=True))
    assert len(result["turns"]) == 1 and result["macro_gate_blockers"] == ["intent"]
    assert result["deferred_micro_feedback"][0]["issue_id"] == "i2"
    assert not any(c["context"]["turn"] == 2 for c in llm.calls)


def test_reviser_discovering_student_macro_cannot_apply_other_micro_operations(tmp_path, executor):
    class LateDiscovery(Intake):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "reviser":
                answer.student_tasks = [StudentTask(issue_id="experience",
                    instruction="책을 읽었던 실제 상황을 알려 주세요.", reason="새 경험이 필요합니다.")]
            return answer
    result, llm, _, _ = execute(tmp_path, executor, llm=LateDiscovery(student=False))
    assert result["final_text"] == ESSAY["text"]
    assert result["turns"][0]["operations"][0]["reason_code"] == "macro_student_gate"
    assert not any(c["role"] == "coordinator_review" for c in llm.calls)
    assert len([c for c in llm.calls if c["role"] == "reviser"]) == 1


@pytest.mark.parametrize("kind", [None, "new_content", "new_experience"])
def test_content_tasks_cannot_bypass_original_text_tests(kind):
    with pytest.raises(ValueError, match="Only a student intent question"):
        Issue(id="missing", level="macro", priority=1, description="경험의 구체적인 장면이 없습니다.",
            evidence_quotes=["나는 책을 읽었다."], needs_student_content=True,
            student_content_type=kind, test=None)


def test_invalid_initial_quote_is_logged_and_retried_before_issues_are_committed(tmp_path, executor):
    class BadQuote(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial" and not payload.get("validation_retry"):
                answer.issues[0].evidence_quotes = ["원문에 없는 인용"]
            return answer
    result, llm, _, _ = execute(tmp_path, executor, llm=BadQuote())
    assert result["route"] == "single_turn"
    assert len([c for c in llm.calls if c["role"] == "coordinator_initial"]) == 2
    events = Journal(tmp_path / "events.jsonl").events
    assert events[0]["kind"] == "initial_plan_rejected"
    assert len([e for e in events if e["kind"] == "issue_added"]) == 1


def test_initial_validation_retry_is_bounded(tmp_path, executor):
    class BadQuote(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                answer.issues[0].evidence_quotes = ["원문에 없는 인용"]
            return answer
    llm = BadQuote()
    with pytest.raises(ValueError):
        execute(tmp_path, executor, llm=llm)
    assert [c["role"] for c in llm.calls] == ["coordinator_initial"] * 2
    assert [e["kind"] for e in Journal(tmp_path / "events.jsonl").events] == ["initial_plan_rejected"] * 2


def test_empty_initial_turn_is_plan_adjustment_not_a_resolved_issue(tmp_path, executor):
    class EmptyTurn(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                answer.turn_plan[0].issue_ids = []
            return answer
    execute(tmp_path, executor, llm=EmptyTurn())
    event = next(e for e in Journal(tmp_path / "events.jsonl").events
                 if e["kind"] == "plan_changed" and e["phase"] == "planned")
    assert event["payload"]["reason_categories"] == ["plan_adjustment"]
