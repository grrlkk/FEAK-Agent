import json
from pathlib import Path

import pytest

from staged_feedback.config import Settings
from staged_feedback.engine import Engine
from conftest import Anonymizer, Diagnostic, ScriptedLLM


def execute(path, observer, *, llm=None, text="책을 읽을수 있다. 함께 연습한다.", condition="planned"):
    llm, diagnostic = llm or ScriptedLLM(), Diagnostic()
    engine = Engine(Settings(condition=condition, mock=True), llm=llm, diagnostic=diagnostic,
        observer=observer, anonymizer_factory=Anonymizer, directory=path)
    result = engine.run(dict(essay_id="synthetic", prompt="일상을 쓰세요.", text=text))
    return result, engine, llm, diagnostic


def test_scoped_pipeline_and_direct_review(tmp_path, observer):
    result, engine, llm, diagnostic = execute(tmp_path, observer)
    assert result["final_text"] == "책을 읽을 수 있다. 함께 연습한다."
    assert [c["role"] for c in llm.calls] == ["coordinator_initial", "feedback", "reviser", "coordinator_review"]
    review = llm.calls[-1]["payload"]
    assert review["feedback"]["items"] and review["changed_sentences"]
    assert "observational_flags" in review and "test_results" not in review
    assert result["resolved_issue_ids"] == ["i1"]
    assert diagnostic.scored == [result["initial_text"], result["final_text"]]
    assert len(diagnostic.extracted) == 1
    assert not any(e["kind"] in {"operation_rejected", "test_result"} for e in engine.journal.events)


def test_single_pass_includes_all_six_issues_without_memory_truncation(tmp_path, observer):
    result, _, llm, _ = execute(tmp_path, observer, llm=ScriptedLLM(count=6))
    assert len(result["turns"]) == 1
    assert len(result["turns"][0]["feedback"]) == 6
    revision = next(c["payload"] for c in llm.calls if c["role"] == "reviser")
    assert len(revision["feedback"]) == 6
    assert "memory_summary" not in revision and "current_text" not in revision
    assert "allowed_operations" not in revision and "preserve" not in revision


def test_every_observation_is_passive_including_prior_changes(tmp_path, observer):
    original = "책을 읽을수 있다. 나는 #@이름#을 만났다. 바닷가에서 산책했다."
    llm = ScriptedLLM(turns=2, revisions=[lambda t: t.replace("읽을수", "읽을 수"), "독서를 즐긴다."])
    result, engine, llm, _ = execute(tmp_path, observer, text=original, llm=llm)
    assert result["final_text"] == "독서를 즐긴다. 나는 #@이름#을 만났다. 바닷가에서 산책했다."
    kinds = {f["kind"] for f in result["turns"][1]["observational_flags"]}
    assert "earlier_revision_changed" in kinds
    assert "anonymization_changed" not in kinds
    assert llm.calls[-1]["payload"]["observational_flags"]
    assert not any("rejected" in e["kind"] for e in engine.journal.events)


@pytest.mark.parametrize("revision", ["", "책을 읽을 수 있다. 즐겁다. 함께 연습한다.", "책을 읽을 수 있고 함께 연습한다."])
def test_empty_split_and_merge_outputs_are_preserved(tmp_path, observer, revision):
    result, _, _, _ = execute(tmp_path, observer, llm=ScriptedLLM(revisions=[revision]))
    assert result["final_text"] == revision + " 함께 연습한다."


def test_anonymization_masks_all_roles_and_restores_known_placeholders(tmp_path, observer):
    result, _, llm, _ = execute(tmp_path, observer, text="#@이름#은 책을 읽을수 있다.")
    assert result["final_text"] == "#@이름#은 책을 읽을 수 있다."
    for call in llm.calls:
        assert "#@이름#" not in json.dumps(call["payload"], ensure_ascii=False)
    assert "__ANON_" not in result["final_text"]
    assert not result["turns"][0]["observational_flags"]


def test_unreviewed_feedback_is_not_automatically_resolved(tmp_path, observer):
    result, _, _, _ = execute(tmp_path, observer, llm=ScriptedLLM(assessments=[]))
    assert result["remaining_issue_ids"] == ["i1"]
    assert "읽을 수" in result["final_text"]


def test_coordinator_can_find_partial_even_when_text_changed(tmp_path, observer):
    result, _, _, _ = execute(tmp_path, observer, llm=ScriptedLLM(assessments=[
        dict(issue_id="i1", status="partial", reason="추가 설명은 아직 어색합니다.")]))
    assert result["remaining_issue_ids"] == ["i1"]
    assert result["turns"][0]["review"]["assessments"][0]["status"] == "partial"


def test_no_memory_condition_and_comparison_full_text(tmp_path, observer):
    _, _, llm, _ = execute(tmp_path / "no_memory", observer, condition="planned_no_memory")
    assert all(call["payload"].get("memory_summary", "") == "" for call in llm.calls)
    result, _, llm, _ = execute(tmp_path / "baseline", observer, condition="single_call_staged")
    assert len(llm.calls) == 1 and len(result["turns"]) == 2


def test_new_run_cannot_overwrite_existing_events(tmp_path, observer):
    execute(tmp_path, observer)
    with pytest.raises(ValueError, match="fresh"):
        execute(tmp_path, observer)


def test_prompts_have_no_test_tool_or_operation_contracts():
    root = Path(__file__).resolve().parents[1]
    prompts = list((root / "prompts/staged_feedback").glob("*.txt"))
    assert len(prompts) == 5
    assert all(len(p.read_text()) <= 1500 for p in prompts)
    from staged_feedback.models import InitialPlan, Revision, Feedback, Review
    fields = set(Revision.model_fields)
    assert fields == {"revised_text", "cannot_apply"}
    for model in (InitialPlan, Feedback, Review):
        assert "TestSpec" not in json.dumps(model.model_json_schema())
    assert "operation_retries" not in Settings.model_fields
    assert "feedback_per_turn" not in Settings.model_fields


def test_switch_to_single_pass_schedules_next_turn_instead_of_ending_current(tmp_path, observer):
    from staged_feedback.models import Strategy
    class Switch(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == 'coordinator_review' and context['turn'] == 1:
                response.strategy = Strategy(name='single_pass', reason='다음 한 턴에 남은 이슈를 모두 다룹니다.')
                response.remaining_turns = 1
                response.continue_work = True
            return response
    result, _, _, _ = execute(tmp_path, observer, llm=Switch(turns=2))
    assert len(result['turns']) == 2
    assert result['turns'][0]['remaining_after_review'] == 1
    assert result['turns'][0]['continue_work'] is True
    assert result['turns'][1]['strategy']['name'] == 'single_pass'
    assert len(result['turns'][1]['feedback']) == 1
