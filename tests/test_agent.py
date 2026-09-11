import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from feak_tc.agent import run_agent
from feak_tc.agent.local_llm import LocalCallBudgetExceeded, LocalJSONClient, parse_json_object
from feak_tc.agent.retrieval import ExemplarStore
from feak_tc.agent.roles import LocalRoles
from feak_tc.agent.schemas import (
    AxisVerdict, GuardVerdict, LocalModelConfig, PlanResponse, RevisionRequest, RevisionVerdict,
)
from feak_tc.diagnose import Diagnosis, RUBRIC_KEYS
from feak_tc.mvp.llm import LLMResponseError
from feak_tc.mvp.schemas import Candidate, Patch
from scripts.run_agent import main


ORIGINAL = "학생의 의견을 듣는다. 기존 조건은 유지한다."


def request(text=ORIGINAL):
    return RevisionRequest(action_type="ADD_DETAIL", target_rubric="content_2",
                           target_span=text.split(".")[0] + ".", problem="설명이 부족하다.",
                           instruction="의견을 듣는 방법을 한 문장으로 보충한다.",
                           preserve=["기존 조건은 유지한다."])


def verdict(target="pass", preservation="pass"):
    return RevisionVerdict(target_fulfillment=AxisVerdict(label=target, reason="목표 판정 근거"),
                           preservation=AxisVerdict(label=preservation, reason="보존 판정 근거"))


def guard(preservation="pass"):
    return GuardVerdict(preservation=AxisVerdict(label=preservation, reason="전역 보존 근거"),
                        coherence=AxisVerdict(label="pass", reason="전체 흐름 유지"))


@pytest.fixture
def cfg(monkeypatch):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/agent_local.yaml").read_text())
    config["controller"].update(max_steps=3, candidates_per_step=1)
    config["validity"]["enabled"] = False
    config["hard_constraints"].update(edit_ratio_max=1.0)
    monkeypatch.setattr("feak_tc.mvp.transition.semantic_text_similarity", lambda a, b: (1.0, {"method": "test"}))
    return config


class Scorer:
    def __init__(self, flat=False):
        self.calls = []
        self.flat = flat

    def diagnose(self, text):
        self.calls.append(text)
        score = 4.0 if self.flat else 4.0 + text.count("보충")
        return Diagnosis(text, {k: score for k in RUBRIC_KEYS}, {}, ["content_2"],
                         metadata={"diagnoser": "test"})


class Roles:
    def __init__(self, verdicts=None, guards=None, fixed_text=None):
        self.verdicts = iter(verdicts or [verdict()] * 10)
        self.guards = iter(guards or [guard()] * 10)
        self.histories = []
        self.fixed_text = fixed_text
        self.guard_inputs = []

    def plan(self, question, before, history, exemplars):
        self.histories.append(copy.deepcopy(history))
        return PlanResponse(plan=request(before.text), reason="구체적 요구")

    def patch(self, text, plan, candidate_index):
        after = self.fixed_text or text.replace(plan.target_span, plan.target_span + " 보충 설명이다.", 1)
        return Candidate(plan.action_type, plan.target_rubric, plan.target_span, plan.instruction,
                         new_text=after, patch=Patch("insert_after", plan.target_span,
                                                    plan.target_span, "보충 설명이다.", plan.instruction))

    def verify(self, *args):
        return next(self.verdicts)

    def guard(self, *args):
        self.guard_inputs.append(args)
        item = next(self.guards)
        if isinstance(item, Exception):
            raise item
        return item


def run(cfg, roles=None, scorer=None):
    return run_agent(ORIGINAL, question="학생 의견 수렴 방법", diagnoser=scorer or Scorer(),
                     roles=roles or Roles(), cfg=cfg)


def test_accept_repeat_and_stop_preserve_full_snapshots(cfg):
    scorer = Scorer()
    output = run(cfg, scorer=scorer)
    assert output["status"] == "completed"
    assert output["accepted"] == 3
    assert output["stop_reason"] == "max_steps"
    assert output["original_text"] == ORIGINAL
    assert output["final_text"].count("보충") == 3
    assert len(scorer.calls) == 4  # accepted diagnosis reused in the next step
    assert [row["parent_id"] for row in output["checkpoints"]] == [None, 0, 1, 2]
    assert all(row["safe"] for row in output["checkpoints"])
    assert all(row["after"]["rubrics"] for row in output["events"] if row["event"] == "candidate")
    phases = [row["stage"] for row in output["events"] if row["event"] == "phase"]
    assert phases == ["diagnose"] + ["plan", "generate", "rediagnose", "verify", "guard"] * 3
    first_patch = next(i for i, row in enumerate(output["events"]) if row["event"] == "patch")
    first_verdict = next(i for i, row in enumerate(output["events"]) if row["event"] == "candidate")
    assert first_patch < first_verdict


@pytest.mark.parametrize("target,preservation", [("fail", "pass"), ("pass", "fail"),
                                                   (None, "pass"), ("pass", None), ("partial", "pass")])
def test_rv_failure_or_unassessable_cannot_be_accepted(cfg, target, preservation):
    roles = Roles(verdicts=[verdict(target, preservation)] * 3)
    output = run(cfg, roles)
    assert output["final_text"] == ORIGINAL
    assert output["accepted"] == 0
    assert output["stop_reason"] == "no_progress"
    assert any(row["event"] == "replan" for row in output["events"])
    assert roles.histories[1][0]["decision"] == "reject"


def test_noop_wins_without_measured_gain(cfg):
    output = run(cfg, scorer=Scorer(flat=True))
    assert output["accepted"] == 0
    candidates = [x for x in output["events"] if x["event"] == "candidate"]
    assert "no_improvement_over_noop" in candidates[0]["reject_reasons"]


def test_rollback_restores_best_checkpoint_not_original(cfg):
    cfg["controller"]["max_rollbacks"] = 1
    roles = Roles(guards=[guard(), guard("fail")])
    output = run(cfg, roles)
    assert output["rollbacks"] == 1
    assert output["accepted"] == 2
    assert output["final_checkpoint_id"] == 1
    assert output["final_text"] == output["checkpoints"][1]["text"]
    assert output["checkpoints"][2]["safe"] is False
    assert output["stop_reason"] == "rollback_limit"
    assert roles.guard_inputs[1][1] == ORIGINAL  # original anchor never changes
    assert roles.guard_inputs[1][2] == output["checkpoints"][1]["text"]


@pytest.mark.parametrize("error,reason", [(LLMResponseError("bad JSON"), "runtime_error"),
                                         (OSError("weights missing"), "runtime_error"),
                                         (TypeError("invalid verifier result"), "runtime_error"),
                                         (LocalCallBudgetExceeded(), "llm_call_budget")])
def test_guard_error_or_budget_restores_safe_text(cfg, error, reason):
    output = run(cfg, Roles(guards=[error]))
    assert output["final_text"] == ORIGINAL
    assert output["rollbacks"] == 1
    assert output["stop_reason"] == reason
    assert output["events"][-2]["event"] in {"error", "rollback"}


def test_rejected_trajectory_cannot_cycle(cfg):
    fixed = ORIGINAL.replace("학생의 의견을 듣는다.", "학생의 의견을 듣는다. 보충 설명이다.")
    roles = Roles(fixed_text=fixed, guards=[guard("fail")])
    output = run(cfg, roles)
    assert output["accepted"] == 1
    assert output["final_text"] == ORIGINAL
    assert any(row.get("reason") == "repeated_failed_plan" for row in output["events"])
    assert any(x["decision"] == "rollback" for x in roles.histories[1])


def test_same_state_rejects_repeated_failed_plan_before_generation(cfg):
    roles = Roles(verdicts=[verdict("fail")])
    output = run(cfg, roles)
    assert sum(row["event"] == "candidate" for row in output["events"]) == 1
    assert any(row.get("reason") == "repeated_failed_plan" for row in output["events"])


def test_invalid_plan_replans_without_patch_or_scoring(cfg):
    class InvalidPlan(Roles):
        def plan(self, *args):
            bad = request().model_copy(update={"target_span": "존재하지 않는 문장"})
            return PlanResponse(plan=bad, reason="bad span")

        def patch(self, *args):
            pytest.fail("Invalid target must not be patched")

    scorer = Scorer()
    result = run(cfg, InvalidPlan(), scorer)
    assert result["accepted"] == 0
    assert len(scorer.calls) == 1


def test_duplicate_candidates_are_not_rescored_or_verified(cfg):
    cfg["controller"].update(max_steps=1, candidates_per_step=2)
    scorer = Scorer()
    output = run(cfg, Roles(verdicts=[verdict()]), scorer)
    assert len(scorer.calls) == 2
    assert output["accepted"] == 1
    assert any("duplicate_candidate" in row.get("reject_reasons", []) for row in output["events"])


def test_llm_rv_prompt_has_only_operational_inputs_and_validates_two_axes(cfg):
    captured = []

    def client(**kwargs):
        captured.append(kwargs)
        return verdict().model_dump()

    response = LocalRoles(client, cfg).verify("과제", ORIGINAL, request(), "수정문")
    assert response.preservation.label == "pass"
    user = captured[0]["user"]
    assert "preserve" in user
    for forbidden in ("reference_repair", "candidate_type", "target_gain", "heuristic_score",
                      "action_consistency", "edit_appropriateness"):
        assert forbidden not in user


def test_invalid_structured_response_retries_then_fails(cfg):
    calls = []

    def client(**kwargs):
        calls.append(kwargs)
        return {"target_fulfillment": "pass"}

    with pytest.raises(LLMResponseError):
        LocalRoles(client, cfg).verify("과제", ORIGINAL, request(), "수정문")
    assert len(calls) == cfg["controller"]["schema_retries"] + 1


def test_patch_json_failure_retries_without_coercing_plain_text(cfg):
    calls = []

    def client(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise LLMResponseError("Local model did not return a complete JSON object")
        return {"after": "예를 들어, 학생회의에서 의견을 나눌 수 있다."}

    patched = LocalRoles(client, cfg).patch(ORIGINAL, request(), 0)
    assert len(calls) == 2
    assert "Previous response failed validation" in calls[1]["user"]
    assert "JSON schema" in calls[0]["user"]
    assert patched.patch.after.startswith("예를 들어")


def test_patcher_uses_local_callback_and_preserves_constraints(cfg, monkeypatch):
    captured = []
    monkeypatch.setattr("feak_tc.mvp.patch.request_json", lambda **kw: pytest.fail("Remote API called"))

    def client(**kwargs):
        captured.append(kwargs)
        return {"after": "학생회의에서 다양한 의견을 모을 수 있다."}

    patched = LocalRoles(client, cfg).patch(ORIGINAL, request(), 0)
    assert "기존 조건은 유지한다." in captured[0]["user"]
    assert "학생회의에서" in patched.new_text
    assert patched.patch.before == request().target_span


@pytest.mark.parametrize("payload", [{"after": ["bad"]}, {"after": 12}, {}])
def test_patch_output_requires_a_string(cfg, payload):
    with pytest.raises(LLMResponseError):
        LocalRoles(lambda **kw: payload, cfg).patch(ORIGINAL, request(), 0)


def test_two_axis_schema_preserves_unassessable_and_rejects_extra_axis():
    assert verdict(None).target_fulfillment.label is None
    with pytest.raises(ValidationError):
        RevisionVerdict.model_validate({**verdict().model_dump(), "action_consistency": "pass"})


@pytest.mark.parametrize("text", ['[]', '{"x": NaN} trailing', 'preface {"x": 1}', '{"x":'])
def test_local_json_rejects_incomplete_or_nonobject_outputs(text):
    with pytest.raises(LLMResponseError):
        parse_json_object(text)


def test_local_json_accepts_one_fenced_object():
    assert parse_json_object('```json\n{"after":"수정문"}\n```') == {"after": "수정문"}


def test_local_budget_checked_before_model_load(cfg, monkeypatch):
    client = LocalJSONClient(LocalModelConfig.model_validate(cfg["local_llm"]))
    client.calls = client.cfg.max_calls
    monkeypatch.setattr(client, "_load", lambda: pytest.fail("Model should not load"))
    with pytest.raises(LocalCallBudgetExceeded):
        client(system="x", user="y")


def test_retrieval_rejects_evaluation_rows_and_excludes_related_essays(tmp_path):
    path = tmp_path / "examples.jsonl"
    row = {"essay_id": "train_1", "source_group": "family_1", "text": "토론으로 의견을 모은다.",
           "split": "test", "rubrics": {"content_2": 8}}
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="split=train"):
        ExemplarStore(path)
    row["split"] = "train"
    path.write_text(json.dumps(row))
    store = ExemplarStore(path)
    assert store.retrieve(ORIGINAL, ["content_2"], source_group="family_1") == []
    assert store.retrieve(row["text"], ["content_2"]) == []
    assert store.retrieve(ORIGINAL, ["content_2"], source_group="other")[0]["essay_id"] == "train_1"


def test_cli_offline_smoke_writes_log_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalJSONClient, "__call__", lambda *a, **kw: pytest.fail("Local model loaded"))
    output = tmp_path / "run.json"
    args = ["--offline-smoke", "--text", ORIGINAL, "--question", "과제", "--output", str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["runtime"]["offline_smoke"] is True
    assert output.with_suffix(".events.jsonl").exists()
    with pytest.raises(SystemExit):
        main(args)


def test_empty_input_rejected_before_model_calls(cfg):
    with pytest.raises(ValueError, match="empty"):
        run_agent(" ", question="과제", diagnoser=Scorer(), roles=Roles(), cfg=cfg)


def test_planner_can_stop_without_a_patch(cfg):
    class Finished(Roles):
        def plan(self, *args):
            return PlanResponse(plan=None, reason="추가로 고칠 근거가 없다.")

    result = run(cfg, Finished())
    assert result["stop_reason"] == "planner_no_request"
    assert result["final_text"] == ORIGINAL


def test_invalid_patch_skips_rediagnosis_and_rv(cfg):
    cfg["validity"]["enabled"] = True
    scorer = Scorer()
    result = run(cfg, Roles(fixed_text="."), scorer)
    assert len(scorer.calls) == 1
    assert result["accepted"] == 0
    assert all("rv" not in x for x in result["events"] if x["event"] == "candidate")


def test_failed_initial_diagnosis_returns_original_with_error(cfg):
    class BrokenScorer:
        def diagnose(self, text):
            raise RuntimeError("Scorer unavailable")

    result = run(cfg, scorer=BrokenScorer())
    assert result["status"] == "error"
    assert result["final_text"] == ORIGINAL
    assert result["checkpoints"] == []


def test_missing_local_weights_are_logged_without_api_fallback(cfg):
    class MissingModel(Roles):
        def plan(self, *args):
            raise OSError("Local model weights not found")

    result = run(cfg, MissingModel())
    assert result["status"] == "error"
    assert result["final_text"] == ORIGINAL
    assert "weights not found" in result["events"][-2]["message"]


def test_kanana_worker_reuses_one_diagnoser_and_reports_errors(monkeypatch):
    from feak_tc.agent.observer import _worker

    inputs = iter([ORIGINAL, "broken", ORIGINAL, None])
    replies = []
    factory_calls = []

    class Connection:
        def recv(self):
            return next(inputs)

        def send(self, value):
            replies.append(value)

        def close(self):
            pass

    class Diagnoser(Scorer):
        def diagnose(self, text):
            if text == "broken":
                raise RuntimeError("sample failure")
            return super().diagnose(text)

    def factory(kind, **kwargs):
        factory_calls.append((kind, kwargs))
        return Diagnoser()

    monkeypatch.setattr("feak_tc.diagnose.get_diagnoser", factory)
    _worker(Connection(), {"question": "과제"})
    assert len(factory_calls) == 1
    assert [row[0] for row in replies] == [True, False, True]
    assert "sample failure" in replies[1][1]
