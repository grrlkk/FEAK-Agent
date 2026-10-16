import json
from pathlib import Path

import pytest

from staged_feedback.config import Settings
from staged_feedback.engine import Engine
from staged_feedback.events import Journal
from conftest import Diagnostic, ScriptedLLM, issue

ESSAY = dict(essay_id="synthetic", prompt="일상에서 하는 활동을 소개하세요.", text="나는 책을 읽을수 있다.\n\n우리는 운동을 할수 있다.")


def execute(tmp_path, executor, *, extend=False, condition="planned", llm=None):
    llm, diagnostic = llm or ScriptedLLM(extend=extend), Diagnostic()
    engine = Engine(Settings(condition=condition), llm=llm, diagnostic=diagnostic, executor=executor, directory=tmp_path)
    return engine.run(ESSAY), llm, diagnostic, engine


def test_one_turn_finishes_with_all_tests(tmp_path, executor):
    result, llm, diagnostic, _ = execute(tmp_path, executor)
    assert result["route"] == "single_turn"
    assert result["final_text"].startswith("나는 책을 읽을 수 있다.")
    assert result["turns"][0]["test_summary"]["all_passed"]
    assert diagnostic.scored == [ESSAY["text"], result["final_text"]]
    assert diagnostic.extracted == [ESSAY["text"]]
    assert [c["role"] for c in llm.calls] == ["coordinator_initial", "feedback", "reviser", "coordinator_review"]


def test_extends_plan_without_reset_and_recalculates_features(tmp_path, executor):
    result, llm, diagnostic, _ = execute(tmp_path, executor, extend=True)
    assert result["route"] == "multi_turn"
    assert result["initial_strategy"]["name"] == "single_pass"
    assert result["strategy"]["name"] == "front_first"
    assert len(result["strategy_history"]) == len(result["turn_plan_history"]) == 2
    assert result["turns"][0]["strategy"]["name"] == "single_pass"
    assert result["turns"][0]["strategy_after_review"]["name"] == "front_first"
    assert result["turns"][0]["test_summary"]["passed"] == 1
    assert result["turns"][1]["test_summary"]["passed"] == 2
    assert "읽을 수" in diagnostic.extracted[1]
    assert "할수" in diagnostic.extracted[1]
    assert "할 수" in result["final_text"]
    assert len(diagnostic.scored) == 2
    assert len(diagnostic.extracted) == 2
    second = next(c for c in llm.calls if c["role"] == "feedback" and c["context"]["turn"] == 2)
    assert second["payload"]["current_features"]["features"]["length"] == len(diagnostic.extracted[1])
    assert {i["id"] for i in second["payload"]["candidates"]} == {"i2"}
    reviser = next(c for c in llm.calls if c["role"] == "reviser" and c["context"]["turn"] == 2)
    assert any(c["quote"] == "나는 책을 읽을 수 있다." for c in reviser["payload"]["preserve"])
    assert not result["remaining_issue_ids"]


@pytest.mark.parametrize("condition", ["planned_no_memory", "routed_no_memory"])
def test_no_memory_is_empty_summary(tmp_path, executor, condition):
    result, llm, _, _ = execute(tmp_path, executor, extend=True, condition=condition)
    assert all(c["payload"]["memory_summary"] == "" for c in llm.calls)
    assert all(v["summary"] == "" for v in result["memory_summaries"])
    # Mechanical preservation still applies in this ablation.
    assert result["turns"][1]["operations"][0]["applied"]


@pytest.mark.parametrize("omitted,reason", [(False, "quote_not_unique"), (True, "retry_omitted")])
def test_failed_operation_retries_once_then_drops(tmp_path, executor, omitted, reason):
    llm = ScriptedLLM(broken=True, omit_retry=omitted)
    result, llm, _, engine = execute(tmp_path, executor, llm=llm)
    assert len([c for c in llm.calls if c["role"] == "reviser"]) == 2
    assert result["final_text"] == ESSAY["text"]
    assert result["remaining_issue_ids"] == ["i1"]
    assert result["turns"][0]["operations"][-1]["reason_code"] == reason
    # Coordinator's unsupported resolution cannot override a failing test.
    changes = [e["payload"] for e in engine.journal.events if e["kind"] == "plan_changed"]
    assert any("i1" in c.get("rejected_resolutions", []) for c in changes)


def test_prior_fixed_sentence_cannot_be_changed_again(tmp_path, executor):
    class Violator(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "reviser" and context["turn"] == 2:
                return schema.model_validate(dict(operations=[dict(id="op1", op="replace", issue_ids=["i2"],
                    quote="읽을 수", replacement="읽을수")], student_tasks=[]))
            return answer
    result, _, _, _ = execute(tmp_path, executor, llm=Violator(extend=True))
    assert "읽을 수" in result["final_text"]
    assert all(op["reason_code"] == "preserve_violation" for op in result["turns"][1]["operations"])
    assert all(op["preserve_violation_attempt"] for op in result["turns"][1]["operations"])


def test_feedback_evidence_must_use_current_value(tmp_path, executor):
    class Stale(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "feedback":
                answer.items[0].feature_evidence[0].value = -1
            return answer
    with pytest.raises(ValueError, match="measured value"):
        execute(tmp_path, executor, llm=Stale())


@pytest.mark.parametrize("mode", ["outside_plan", "too_many"])
def test_feedback_is_limited_to_current_direction_and_count(tmp_path, executor, mode):
    class BadFeedback(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial" and mode == "outside_plan":
                answer.turn_plan[0].issue_ids = ["i1"]
            if role == "feedback":
                if mode == "outside_plan":
                    answer.items[0].issue_id = "i2"
                else:
                    answer.items *= 3
            return answer
    with pytest.raises(ValueError, match="planned issues"):
        execute(tmp_path, executor, llm=BadFeedback(extend=True))


def test_new_issue_after_passing_tests_requires_another_turn(tmp_path, executor):
    class NewIssue(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            turn = context["turn"]
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_review" and turn == 1:
                answer.new_issues = [issue("i2", "할수", "할 수")]
                answer.turn_budget, answer.continue_work = 2, True
                from staged_feedback.models import TurnDirection
                answer.turn_plan = [TurnDirection(turn=1, direction="첫 표기", issue_ids=["i1"]),
                                    TurnDirection(turn=2, direction="새로 찾은 표기", issue_ids=["i2"])]
            if turn == 2 and role == "feedback":
                answer.items[0].issue_id, answer.items[0].target_quote = "i2", "할수"
                answer.items[0].test = issue("i2", "할수", "할 수").test
            if turn == 2 and role == "reviser":
                answer.operations[0].issue_ids = ["i2"]
                answer.operations[0].quote, answer.operations[0].replacement = "할수", "할 수"
            return answer
    result, _, _, _ = execute(tmp_path, executor, llm=NewIssue())
    assert result["route"] == "multi_turn"
    assert result["turns"][0]["test_summary"]["all_passed"]
    assert result["turns"][0]["untested_issue_ids"] == ["i2"]
    assert not result["remaining_issue_ids"]


def test_macro_tests_are_batched_once_each_turn(tmp_path, executor):
    from test_checks import macro

    class WithMacro(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            if role == "macro_tests":
                self.calls.append(dict(role=role, payload=payload, context=context))
                return schema.model_validate(dict(results=[dict(issue_id=t["issue_id"], passed=context["turn"] > 1,
                    reason="문장 연결을 확인했습니다.", evidence_quotes=["나는 책을 읽을 수 있다."])
                    for t in payload["tests"]]))
            answer = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                for identifier in ["m1", "m2"]:
                    problem = macro(identifier)
                    problem.evidence_quotes = problem.test.target_quotes = ["나는 책을 읽을수 있다."]
                    answer.issues.append(problem)
                    answer.turn_plan[0].issue_ids.append(identifier)
            if role == "coordinator_review" and context["turn"] == 1:
                answer.turn_plan[1].issue_ids.extend(["m1", "m2"])
                from staged_feedback.models import Strategy
                answer.strategy = Strategy(name="macro_first", reason="문장 연결도 함께 확인합니다.")
            return answer
    result, llm, _, _ = execute(tmp_path, executor, llm=WithMacro(extend=True))
    calls = [c for c in llm.calls if c["role"] == "macro_tests"]
    assert len(calls) == 2
    assert all(len(c["payload"]["tests"]) == 2 for c in calls)
    assert result["strategy"]["name"] == "macro_first"
    assert not result["remaining_issue_ids"]


def test_direct_comparison_sends_every_initial_issue(tmp_path, executor):
    result, llm, _, _ = execute(tmp_path, executor, extend=True, condition="always_direct")
    reviser = next(c for c in llm.calls if c["role"] == "reviser")
    assert len(reviser["payload"]["feedback"]["items"]) == 2
    assert not any(c["role"] in {"feedback", "coordinator_review"} for c in llm.calls)
    assert len(result["turns"]) == 1


@pytest.mark.parametrize("condition", ["always_direct", "always_staged", "routed"])
def test_comparison_controls_mock_only(tmp_path, executor, condition):
    result, _, _, _ = execute(tmp_path, executor, condition=condition)
    assert result["condition"] == condition
    assert result["route"] == "single_turn"
    assert len(result["initial_turn_plan"]) == (4 if condition == "always_staged" else 1)


def test_single_call_comparison_calls_llm_once(tmp_path, executor):
    result, llm, diagnostic, _ = execute(tmp_path, executor, condition="single_call_staged")
    assert len(llm.calls) == 1
    assert len(result["turns"]) == 2
    assert len(diagnostic.scored) == 2
    assert not diagnostic.extracted


def test_journal_can_reload_and_project_full_history(tmp_path, executor):
    result, _, _, engine = execute(tmp_path, executor, extend=True)
    journal = Journal(tmp_path / "events.jsonl")
    assert journal.digest() == engine.journal.digest() == result["journal_sha256"]
    view = journal.summary("coordinator", phase="planned", remaining_turns=0)
    assert view["strategy"]["name"] == "front_first"
    assert len(view["turn_plan"]) == 2
    assert not view["remaining_issues"]
    assert len(view["flow"]) == 2
    assert view["flow"][-1]["tests"]["all_passed"]
    assert journal.summary("feedback", phase="planned", remaining_turns=0)["handled_issue_ids"] == ["i1", "i2"]
    assert journal.summary("reviser", phase="planned", remaining_turns=0)["earlier_edits"]
    original = (tmp_path / "events.jsonl").read_bytes()
    journal.append("turn_ended", phase="planned", turn=3, summary="추가 사건")
    assert (tmp_path / "events.jsonl").read_bytes().startswith(original)
    assert journal.summary("coordinator", phase="planned", remaining_turns=0, enabled=False) == ""


def test_prompts_are_short_and_begin_with_goal():
    paths = list((Path(__file__).parents[1] / "prompts/staged_feedback").glob("*.txt"))
    assert len(paths) == 5
    for path in paths:
        text = path.read_text()
        assert len(text) <= 1500
        assert text.startswith("목표:")
