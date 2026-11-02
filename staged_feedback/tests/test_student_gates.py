from conftest import ScriptedLLM, issue
from test_engine import execute
from staged_feedback.models import Issue, StudentTask, TurnDirection, Strategy


class Intake(ScriptedLLM):
    def request(self, role, payload, schema, *, context):
        response = super().request(role, payload, schema, context=context)
        if role == "coordinator_initial":
            response.issues.append(Issue(id="student", level="macro", macro_type="student_input",
                description="주장의 의도를 확인해 주세요.", evidence_quotes=["책"], priority=1,
                needs_student_content=True, student_content_type="intent"))
            response.student_tasks = [StudentTask(issue_id="student", instruction="반어인지 설명해 주세요.", reason="학생의 의도 필요"),
                StudentTask(issue_id="student", instruction="반어인지 알려 주세요.", reason="의도 확인")]
        return response


def test_student_intent_does_not_stop_micro_revision(tmp_path, observer):
    result, engine, llm, _ = execute(tmp_path, observer, llm=Intake())
    assert len(llm.calls) == 4 and len(result["turns"]) == 1
    assert len(result["student_tasks"]) == 1
    assert not result["deferred_micro_feedback"]
    assert "읽을 수" in result["final_text"]
    assert not any(e["kind"] == "gate_blocked" for e in engine.journal.events)


def test_workers_only_flag_and_coordinator_creates_tasks(tmp_path, observer):
    class WorkerInput(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == "feedback":
                response.items[0].needs_student_input = True
            if role == "coordinator_review":
                assert payload["student_tasks"] == []
                assert payload["feedback"]["items"][0]["needs_student_input"]
                response.student_tasks = [StudentTask(issue_id="i1", instruction="의도를 알려 주세요.", reason="뜻을 확정할 수 없음")]
                response.assessments[0].status = "not_applied"
            return response
    result, _, llm, _ = execute(tmp_path, observer, llm=WorkerInput())
    assert [c["role"] for c in llm.calls] == ["coordinator_initial", "feedback", "coordinator_review"]
    assert result["final_text"] == result["initial_text"]
    assert len(result["student_tasks"]) == 1


def test_cannot_apply_preserves_scope_and_reaches_review(tmp_path, observer):
    class Cannot(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == "reviser":
                response.cannot_apply = "학생의 경험이 필요합니다."
            if role == "coordinator_review":
                assert payload["scope_revisions"][0]["cannot_apply"]
                assert payload["student_tasks"] == []
                response.assessments[0].status = "not_applied"
            return response
    result, _, _, _ = execute(tmp_path, observer, llm=Cannot())
    assert result["final_text"] == result["initial_text"]
    assert result["student_tasks"] == []


class Restructure(ScriptedLLM):
    def __init__(self, reopen=False):
        super().__init__(turns=3 if reopen else 2)
        self.reopen = reopen

    def request(self, role, payload, schema, *, context):
        response = super().request(role, payload, schema, context=context)
        if role == "coordinator_initial":
            response.issues.append(Issue(id="structure", level="macro", macro_type="restructure",
                description="기존 문장의 순서를 바꿉니다.", evidence_quotes=["읽을수"], priority=1))
            response.issues[0].restructure_issue_ids = ["structure"]
            response.issues.append(issue("independent"))
            response.turn_plan[0].issue_ids = ["structure", "i1", "independent"]
            response.turn_plan[1].issue_ids = ["i1"]
        if role == "feedback":
            for item in response.items:
                if item.issue_id == "structure":
                    item.target_ids = ["p1"]
        if role == "coordinator_review":
            for assessment in response.assessments:
                assessment.status = "applied"
            if self.reopen and context["turn"] == 2:
                response.reopened = ["structure", "i1"]
        return response


def test_only_related_micro_waits_and_runs_after_restructure(tmp_path, observer):
    result, _, llm, _ = execute(tmp_path, observer, llm=Restructure())
    turns = [c["payload"]["candidates"] for c in llm.calls if c["role"] == "feedback"]
    assert {i["id"] for i in turns[0]} == {"structure", "independent"}
    assert {i["id"] for i in turns[1]} == {"i1"}
    assert not result["deferred_micro_feedback"]


def test_reopened_structure_defers_related_micro_again(tmp_path, observer):
    result, _, llm, _ = execute(tmp_path, observer, llm=Restructure(reopen=True))
    turns = [c["payload"]["candidates"] for c in llm.calls if c["role"] == "feedback"]
    assert {i["id"] for i in turns[2]} == {"structure"}
    assert "i1" in result["remaining_issue_ids"]


def test_optional_only_does_not_create_task_or_revision(tmp_path, observer):
    class Optional(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                response.issues = []
                response.optional_suggestions = ["원한다면 예를 덧붙여도 좋습니다."]
                response.remaining_turns = 0
            return response
    result, _, llm, _ = execute(tmp_path, observer, llm=Optional())
    assert len(llm.calls) == 1
    assert result["student_tasks"] == result["remaining_issue_ids"] == []
    assert result["optional_suggestions"] == ["원한다면 예를 덧붙여도 좋습니다."]


def test_new_student_task_during_review_does_not_cancel_next_revision(tmp_path, observer):
    class LaterTask(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == 'coordinator_review' and context['turn'] == 1:
                response.new_issues.append(Issue(id='student', level='macro', macro_type='student_input',
                    description='학생의 의도를 설명해 주세요.', evidence_quotes=['책'], priority=1))
            return response
    result, _, llm, _ = execute(tmp_path, observer, llm=LaterTask(turns=2))
    assert len(result['turns']) == 2
    assert len(result['student_tasks']) == 1
    assert result['resolved_issue_ids'] == ['i1']
