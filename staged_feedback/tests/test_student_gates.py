from conftest import ScriptedLLM
from test_engine import execute


class Intake(ScriptedLLM):
    def request(self, role, payload, schema, *, context):
        response = super().request(role, payload, schema, context=context)
        if role == "coordinator_initial":
            from staged_feedback.models import Issue, StudentTask
            task = Issue(id="student", level="macro", description="주장의 의도를 확인해 주세요.",
                evidence_quotes=["책"], priority=1, needs_student_content=True, student_content_type="intent")
            response.issues.append(task)
            response.student_tasks = [StudentTask(issue_id="student", instruction="반어인지 설명해 주세요.", reason="학생의 의도 필요"),
                StudentTask(issue_id="student", instruction="반어인지 알려 주세요.", reason="의도 확인")]
        return response


def test_initial_student_intent_blocks_revisions_and_defers_micro(tmp_path, observer):
    result, _, llm, _ = execute(tmp_path, observer, llm=Intake())
    assert len(llm.calls) == 1 and not result["turns"]
    assert result["route"] == "student_tasks"
    assert result["macro_gate_blockers"] == ["student"]
    assert len(result["student_tasks"]) == 1
    assert result["deferred_micro_feedback"][0]["issue_id"] == "i1"
    assert result["final_text"] == result["initial_text"]


def test_tasks_from_reviser_do_not_reject_returned_text(tmp_path, observer):
    class StudentAtRevision(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == "reviser":
                from staged_feedback.models import StudentTask
                response.student_tasks.append(StudentTask(issue_id="i1", instruction="실제 경험을 써 주세요.", reason="학생 정보 필요"))
            return response
    result, _, llm, _ = execute(tmp_path, observer, llm=StudentAtRevision())
    assert "읽을 수" in result["final_text"]
    assert llm.calls[-1]["role"] == "coordinator_review"
    assert len(result["student_tasks"]) == 1
