import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from feak_tc.agent import run_pilot
from feak_tc.agent.roles import Roles
from feak_tc.agent.editing import text_units
from feak_tc.agent.schemas import (
    CRITERIA, ControllerConfig, Criterion, EditScope, OpenAIConfig, PatchResponse, PlanResponse,
    ReplacementPatch, Revision, RevisionPlan, RevisionVerdict, RubricAssessment, RubricCheck,
    Sample, SourceIssue, SourceReview,
)
from feak_tc.agent.scorer import ScoreState, StateScorer
from feak_tc.diagnose import Diagnosis, RUBRIC_KEYS
from feak_tc.runtime.openai import APIUnavailable, CallBudgetExceeded, OpenAIJSONClient, ResponseError
from scripts.run_pilot import main


ORIGINAL = "학생의 의견을 듣는다. 기존 조건은 유지한다."
FIRST = "학급 회의에서 학생의 의견을 듣는다. 기존 조건은 유지한다."
SECOND = "설문으로 학생의 의견을 듣는다. 기존 조건은 유지한다."
SAMPLE = Sample(sample_id="sample", writing_prompt="학생 의견을 수렴하는 방법", text=ORIGINAL)


def plan(text=ORIGINAL):
    return RevisionPlan(target_rubric="content_2", target_span=text,
                        problem="의견을 듣는 방법이 불분명하다", goal="의견 수렴 방법 하나를 제시한다",
                        must_preserve=["기존 조건은 유지한다"], evidence_spans=[text],
                        reader_impact="의견을 듣는 방법을 알 수 없다", context_check="다른 문장에 방법이 없다",
                        operation="replace", edit_scope=EditScope(start_unit="U0001", end_unit=text_units(text)[-1].unit_id))


def planning(text=ORIGINAL, outcome="edit"):
    checks = [RubricCheck(rubric=key, finding="no_actionable_issue", reason="구체적 결함 없음")
              for key in RUBRIC_KEYS]
    selected = plan(text) if outcome == "edit" else None
    if selected:
        checks[RUBRIC_KEYS.index(selected.target_rubric)].finding = "actionable"
    elif outcome == "needs_information":
        checks[0].finding = "needs_information"
    return PlanResponse(outcome=outcome, priority_checks=checks, plan=selected, reason="한 문제를 검토")


def verdict(**labels):
    values = {key: labels.get(key, "PASS") for key in CRITERIA}
    decision = "REJECT" if "FAIL" in values.values() else (
        "REVERIFY" if "UNCERTAIN" in values.values() else "ACCEPT")
    return RevisionVerdict(**{key: Criterion(label=value, reason="구체적인 판정 근거")
                              for key, value in values.items()}, decision=decision)


class Scorer:
    def __init__(self, values=None, trace=None):
        self.values = values or {ORIGINAL: 5, FIRST: 4, SECOND: 6}
        self.calls = []
        self.trace = trace if trace is not None else []

    def score(self, text):
        self.calls.append(text)
        self.trace.append("score:" + text)
        value = self.values[text]
        if isinstance(value, Exception):
            raise value
        return ScoreState({key: float(value) for key in RUBRIC_KEYS}, {"source": "test"})


class FakeRoles:
    def __init__(self, verdicts=None, candidates=None, trace=None):
        self.verdicts = iter(verdicts or [verdict()] * 10)
        self.candidates = iter(candidates or [FIRST, SECOND])
        self.plans = []
        self.revisions = []
        self.verifications = []
        self.trace = trace if trace is not None else []

    def plan(self, prompt, text, scores):
        self.plans.append((prompt, text, dict(scores)))
        return planning(text)

    def revise(self, prompt, text, request, feedback=None):
        self.revisions.append((prompt, text, request.model_dump(), feedback))
        return Revision(revised_text=next(self.candidates), summary_of_change="한 부분 수정")

    def verify(self, prompt, before, after, request):
        self.trace.append("verify:" + after)
        self.verifications.append((prompt, before, after, request.model_dump()))
        result = next(self.verdicts)
        if isinstance(result, Exception):
            raise result
        return result


def run(roles=None, scorer=None, **config):
    return run_pilot(SAMPLE, scorer=scorer or Scorer(), roles=roles or FakeRoles(),
                     config=ControllerConfig(max_iterations=1, **config))


def test_accepts_score_decrease_and_measures_only_after_rv_decision():
    trace = []
    result = run(FakeRoles(trace=trace), Scorer(trace=trace))
    assert result["accepted"] == 1
    assert result["final_text"] == FIRST
    row = result["attempts"][0]
    assert row["score_baseline"]["aggregate_delta"] < 0
    assert row["score_baseline"]["used_for_control"] is False
    assert trace == ["score:" + ORIGINAL, "verify:" + FIRST, "score:" + FIRST]
    names = [event["event"] for event in result["events"]]
    assert names.index("decision") < names.index("state", names.index("decision"))


def test_rejects_score_increase_and_retries_same_issue_from_unchanged_state():
    roles = FakeRoles(verdicts=[verdict(preservation="FAIL"), verdict()], candidates=[SECOND, FIRST])
    result = run(roles)
    assert len(roles.plans) == 1
    assert roles.revisions[0][1:3] == roles.revisions[1][1:3]
    assert roles.revisions[0][3] is None
    assert roles.revisions[1][3]["rejected_candidate"] == SECOND
    assert "preservation: FAIL" in roles.revisions[1][3]["reasons"][0]
    rejected, accepted = result["attempts"]
    assert rejected["controller_decision"] == "RETRY"
    assert rejected["score_baseline"]["aggregate_accept"] is True
    assert rejected["text_after"] == ORIGINAL
    assert rejected["scores_after"] == rejected["scores_before"]
    assert accepted["text_before"] == ORIGINAL
    assert result["final_text"] == FIRST


def test_uncertain_gets_one_fresh_reverification_without_revision_retry():
    roles = FakeRoles(verdicts=[verdict(necessity="UNCERTAIN"), verdict()])
    result = run(roles)
    assert roles.verifications[0] == roles.verifications[1]
    assert len(roles.revisions) == 1
    assert len(result["attempts"][0]["rv_checks"]) == 2
    assert result["accepted"] == 1


def test_repeated_uncertainty_exhausts_both_bounded_attempts_and_stops():
    roles = FakeRoles(verdicts=[verdict(global_benefit="UNCERTAIN")] * 4)
    result = run(roles)
    assert len(roles.verifications) == 4
    assert len(result["attempts"]) == 2
    assert result["attempts"][-1]["rv"]["decision"] == "REVERIFY"
    assert result["attempts"][-1]["acceptance_decision"] == "REJECT"
    assert result["attempts"][-1]["controller_decision"] == "STOP"
    assert result["stop_reason"] == "repeated_rejection"
    assert result["final_text"] == ORIGINAL


def test_fail_takes_precedence_over_uncertain_and_does_not_reverify():
    roles = FakeRoles(verdicts=[verdict(preservation="FAIL", necessity="UNCERTAIN")])
    result = run(roles, max_retry_per_issue=0)
    assert len(roles.verifications) == 1
    assert result["accepted"] == 0


def test_three_iterations_use_only_last_accepted_state_and_then_stop():
    third = SECOND + " 追加 설명이다."
    roles = FakeRoles(candidates=[FIRST, SECOND, third])
    result = run_pilot(SAMPLE, scorer=Scorer({ORIGINAL: 5, FIRST: 4, SECOND: 6, third: 5}),
                       roles=roles, config=ControllerConfig())
    assert [row[1] for row in roles.plans] == [ORIGINAL, FIRST, SECOND]
    assert result["accepted"] == 3
    assert result["stop_reason"] == "max_iterations"


def test_planner_stop_never_generates_or_verifies():
    roles = FakeRoles()
    roles.plan = lambda *args: planning(outcome="no_actionable_issue")
    result = run(roles)
    assert result["stop_reason"] == "no_actionable_issue"
    assert result["attempts"] == []
    assert roles.revisions == []


def test_missing_information_is_logged_as_a_distinct_planning_stop():
    roles = FakeRoles()
    roles.plan = lambda *args: planning(outcome="needs_information")
    result = run(roles)
    assert result["stop_reason"] == "needs_information"
    assert result["final_text"] == ORIGINAL and not roles.revisions


def test_reviser_abstention_preserves_original_without_retry_verification_or_scoring():
    roles, scorer = FakeRoles(), Scorer()
    roles.revise = lambda *args: Revision(revised_text=ORIGINAL, outcome="cannot_revise",
                                        summary_of_change="새로운 사실이 필요하여 수정 불가")
    result = run(roles, scorer)
    assert result["stop_reason"] == "revision_not_feasible"
    assert result["status"] == "completed" and result["accepted"] == 0
    assert result["final_text"] == ORIGINAL and not roles.verifications
    assert scorer.calls == [ORIGINAL]
    assert result["attempts"][0]["candidate"] is None
    assert result["attempts"][0]["acceptance_decision"] == "NOT_GENERATED"
    assert result["attempts"][0]["controller_decision"] == "STOP"


@pytest.mark.parametrize("error", [ResponseError("invalid RV"), CallBudgetExceeded("budget")])
def test_verifier_failure_preserves_and_logs_candidate_without_adoption(error):
    result = run(FakeRoles(verdicts=[error]))
    assert result["status"] == "error"
    assert result["final_text"] == ORIGINAL
    assert result["attempts"][0]["candidate"] == FIRST
    assert result["attempts"][0]["acceptance_decision"] == "ERROR"
    assert any(row["event"] == "candidate" for row in result["events"])


def test_measurement_failure_does_not_reverse_an_rv_acceptance():
    result = run(scorer=Scorer({ORIGINAL: 5, FIRST: RuntimeError("scorer unavailable")}))
    assert result["status"] == "error"
    assert result["accepted"] == 1
    assert result["final_text"] == FIRST
    assert result["final_scores"] is None
    assert result["attempts"][0]["acceptance_decision"] == "ACCEPT"
    assert result["attempts"][0]["scores_after"] is None


def test_initial_scoring_failure_returns_original_with_error():
    result = run(scorer=Scorer({ORIGINAL: RuntimeError("initial failure")}))
    assert result["status"] == "error"
    assert result["final_text"] == ORIGINAL
    assert result["attempts"] == []


def test_noop_is_logged_and_rejected_without_rv():
    roles = FakeRoles(candidates=[ORIGINAL])
    result = run(roles, max_retry_per_issue=0)
    assert result["accepted"] == 0
    assert roles.verifications == []
    assert result["attempts"][0]["candidate"] == ORIGINAL


def test_disabling_shadow_baseline_does_not_score_rejected_candidate():
    scorer = Scorer()
    result = run(FakeRoles(verdicts=[verdict(preservation="FAIL")]), scorer,
                 log_score_baseline=False, max_retry_per_issue=0)
    assert scorer.calls == [ORIGINAL]
    assert result["attempts"][0]["scores_candidate"] is None


def test_rv_contract_rejects_two_axis_format_and_inconsistent_decision():
    data = verdict(preservation="FAIL").model_dump()
    data["decision"] = "ACCEPT"
    with pytest.raises(ValidationError):
        RevisionVerdict.model_validate(data)
    with pytest.raises(ValidationError):
        RevisionVerdict.model_validate({"target_fulfillment": {"label": "pass", "reason": "legacy"},
                                        "preservation": {"label": "pass", "reason": "legacy"}})


def test_planner_api_schema_restricts_rubric_to_exact_ids():
    schema = RevisionPlan.model_json_schema()
    assert set(schema["properties"]["target_rubric"]["enum"]) == set(RUBRIC_KEYS)
    data = plan().model_dump()
    data["target_rubric"] = "content_2 (설명구체성)"
    with pytest.raises(ValidationError):
        RevisionPlan.model_validate(data)


def test_rv_payload_is_allowlisted_and_repeated_calls_have_no_history():
    calls = []
    def client(**kwargs):
        calls.append(kwargs)
        return verdict().model_dump()
    roles = Roles(client)
    for _ in range(2):
        roles.verify(SAMPLE.writing_prompt, ORIGINAL, FIRST, plan())
    assert calls[0] == calls[1]
    payload = json.loads(calls[0]["user"])
    assert set(payload) == {"writing_prompt", "current_draft", "candidate_revised_draft",
                           "planner_goal", "must_preserve", "text_diff"}
    assert "problem" not in payload and "target_rubric" not in payload
    assert payload["text_diff"]


def test_schema_retry_repeats_fresh_payload_and_is_bounded():
    calls = []
    def client(**kwargs):
        calls.append(kwargs)
        return {"previous_rationale": "MUST NOT LEAK"}
    with pytest.raises(ResponseError):
        Roles(client, schema_retries=1).verify("문항", ORIGINAL, FIRST, plan())
    assert len(calls) == 2 and calls[0] == calls[1]
    assert "MUST NOT LEAK" not in calls[-1]["user"]


def test_feedback_is_only_in_reviser_payload_and_planner_gets_eight_definitions():
    calls = []
    def client(**kwargs):
        calls.append(kwargs)
        if kwargs["schema"] is SourceReview:
            issue = SourceIssue(**{key: getattr(plan(), key) for key in SourceIssue.model_fields})
            return SourceReview(assessments=[RubricAssessment(**check.model_dump(),
                issue=issue if check.finding == "actionable" else None)
                for check in planning().priority_checks]).model_dump()
        if kwargs["schema"] is RevisionPlan:
            return plan().model_dump()
        return PatchResponse(outcome="revised", patch=ReplacementPatch(
            **plan().edit_scope.model_dump(), expected_text=ORIGINAL, replacement=FIRST),
            summary_of_change="변경").model_dump()
    roles = Roles(client)
    roles.plan("문항", ORIGINAL, {key: 5.0 for key in RUBRIC_KEYS})
    roles.revise("문항", ORIGINAL, plan(), {"reasons": ["보존 실패"]})
    payload = json.loads(calls[0]["user"])
    assert set(payload["rubric_definitions"]) == set(RUBRIC_KEYS)
    assert set(payload) == {"writing_prompt", "current_draft", "rubric_definitions", "text_units"}
    planning_payload = json.loads(calls[1]["user"])
    assert set(planning_payload["rubric_scores"]) == set(RUBRIC_KEYS)
    assert "rejection_feedback" not in planning_payload
    assert json.loads(calls[2]["user"])["rejection_feedback"]["reasons"] == ["보존 실패"]


def test_scorer_preserves_native_continuous_values_final_grades_and_caches():
    calls = []
    def diagnose(text):
        calls.append(text)
        return Diagnosis(text, {key: 9 for key in RUBRIC_KEYS}, {}, [],
                         {"rf_corrected_score": [9.125] * 8, "soft_mean": [8.9] * 8,
                          "soft_std": [0.1] * 8})
    scorer = StateScorer(SimpleNamespace(diagnose=diagnose))
    state = scorer.score(ORIGINAL)
    assert set(state.scores.values()) == {9.125}  # No invented 0-10 scale/clipping/rounding.
    assert state.metadata["final_grades"] == {key: 9 for key in RUBRIC_KEYS}
    assert scorer.score(ORIGINAL) is state and len(calls) == 1


def test_cli_five_sample_smoke_and_output_collision(tmp_path):
    output = tmp_path / "run"
    args = ["--input", "examples/pilot_samples.jsonl", "--offline-smoke", "--output-dir", str(output)]
    assert main(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["samples"] == 5 and report["api_calls"] == 0
    assert len((output / "trajectories.jsonl").read_text().splitlines()) == 5
    with pytest.raises(SystemExit):
        main(args)


def test_duplicate_sample_ids_rejected_before_output_creation(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text((SAMPLE.model_dump_json() + "\n") * 2)
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--offline-smoke", "--output-dir", str(output)])
    assert not output.exists()


def api_client(cfg=None, create=None, on_record=None):
    cfg = cfg or OpenAIConfig(max_calls_per_sample=1, max_calls_total=2)
    client = OpenAIJSONClient(cfg, on_record)
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return client


def response():
    return SimpleNamespace(id="resp_test", model="gpt-5-mini", status="completed",
                           output_text=verdict().model_dump_json(), output=[],
                           usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 50, "output_tokens": 80}))


def test_api_requests_are_independent_and_respect_sample_and_total_budgets():
    calls, records = [], []
    def create(**kwargs):
        calls.append(kwargs)
        return response()
    client = api_client(create=create, on_record=records.append)
    kwargs = {"system": "test", "user": "{}", "schema": RevisionVerdict}
    client.start_sample("one")
    client(**kwargs)
    with pytest.raises(CallBudgetExceeded):
        client(**kwargs)
    client.start_sample("two")
    client(**kwargs)
    client.start_sample("three")
    with pytest.raises(CallBudgetExceeded):
        client(**kwargs)
    assert len(calls) == len(records) == 2
    assert calls[0] == calls[1]
    assert calls[0]["store"] is False and calls[0]["truncation"] == "disabled"
    assert "previous_response_id" not in calls[0] and "conversation" not in calls[0]
    assert calls[0]["text"]["format"]["strict"] is True


def test_api_errors_are_sanitized_logged_and_counted():
    import openai
    records = []
    def create(**kwargs):
        raise openai.APIConnectionError(message="secret must not appear", request=None)
    client = api_client(create=create, on_record=records.append)
    with pytest.raises(APIUnavailable) as error:
        client(system="s", user="{}", schema=RevisionVerdict)
    assert "secret" not in str(error.value)
    assert "secret" not in json.dumps(records)
    assert client.calls == 1 and records[0]["status"] == "error"


def test_api_refusal_stops_without_schema_retry():
    refused = response()
    refused.output = [SimpleNamespace(content=[SimpleNamespace(type="refusal")])]
    client = api_client(create=lambda **kwargs: refused)
    with pytest.raises(APIUnavailable):
        Roles(client, schema_retries=1).verify("문항", ORIGINAL, FIRST, plan())
    assert client.calls == 1
