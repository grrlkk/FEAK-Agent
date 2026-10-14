from types import SimpleNamespace

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.api import PhaseBudget
from verak.v3.common import load_config
from verak.v3.ec_verification import Phase2API, select_sol
from verak.v3.ec_verification import both_runs_ok, judge_payload, validate_judgment
from verak.v3.ec_metrics import agreement, pilot_projection, token_usage
from verak.v3.common import write_json
from verak.v3.phase2 import sample_ec


def test_requested_sol_only_and_newest_family_first():
    assert select_sol(["gpt-6-sol", "gpt-6.1-sol"]) == "gpt-6.1-sol"
    assert select_sol(["gpt-6-sol", "gpt-6-astra"]) == "gpt-6-sol"
    assert select_sol(["gpt-5.6-sol", "gpt-6-astra"]) is None
    assert select_sol(["gpt-6.1-sol-2026-10-01", "gpt-6-sol"]) == "gpt-6.1-sol-2026-10-01"


def test_discovery_cached_and_shares_phase2_budget(tmp_path):
    config = load_config()
    config["paths"]["phase2_output"] = tmp_path
    calls = []

    def list_models():
        calls.append("GET")
        return SimpleNamespace(data=[SimpleNamespace(model_dump=lambda: {"id": "gpt-6-sol"})])

    factory = lambda _: SimpleNamespace(models=SimpleNamespace(list=list_models), close=lambda: None)
    api = Phase2API(config, 2, client_factory=factory)
    assert api.discover()["selected_model"] == "gpt-6-sol"
    restarted = Phase2API(config, 2, client_factory=factory)
    assert restarted.discover() == api.discover()
    assert calls == ["GET"] and restarted.budget.used == 1
    assert restarted.budget.reserve() == 2  # Later judgments use the same ledger.
    with pytest.raises(CallBudgetExceeded):
        api.budget.reserve()
    with pytest.raises(ValueError, match="another phase"):
        PhaseBudget(tmp_path / "api_budget.json", 2, authorized_ceiling=220).reserve()
    with pytest.raises(ValueError):
        Phase2API(config, 221)


def test_discovery_never_calls_api_when_budget_zero(tmp_path):
    config = load_config()
    config["paths"]["phase2_output"] = tmp_path
    def forbidden():
        pytest.fail("GET made with zero budget")
    factory = lambda _: SimpleNamespace(models=SimpleNamespace(list=forbidden), close=lambda: None)
    with pytest.raises(CallBudgetExceeded):
        Phase2API(config, 0, client_factory=factory).discover()


def test_discovery_failures_consume_budget(tmp_path):
    config = load_config()
    config["paths"]["phase2_output"] = tmp_path
    def failure():
        raise RuntimeError("simulated request failure")
    factory = lambda _: SimpleNamespace(models=SimpleNamespace(list=failure), close=lambda: None)
    api = Phase2API(config, 1, client_factory=factory)
    with pytest.raises(RuntimeError, match="simulated"):
        api.discover()
    assert api.budget.used == 1
    with pytest.raises(CallBudgetExceeded):
        api.discover()


def test_phase1b_decisions_and_teacher_unchanged():
    config = load_config()
    assert config["scorer"]["average_k"] == 1
    assert config["reward"]["quality"]["scorer_method"] == "score"
    assert config["check"]["quality_source"] == "score"
    assert config["reward"]["quality"]["noise_floor_by_genre"] == {
        "설명": 0.71466138, "논증": 0.21266690, "정서": 0.52354468}
    assert config["reward"]["quality"]["noise_floor"] == pytest.approx(0.52943076)
    assert config["teacher"]["model"] == "gpt-5-mini"
    assert config["teacher"]["reasoning_effort"] == "low"
    assert config["ec_judge"]["reasoning_effort"] == "high"


def judgment_row(sid="valid:1:S1"):
    result = {"token_id": "M2", "segmentation_ok": True, "tag_ok": True,
              "relation_candidates_ok": True, "note": "authored fixture"}
    return {"sentence_id": sid, "sentence": "가면 온다.", "morphemes": [],
            "ec_tokens": [{"token_id": "M2"}], "human_ok": None, "llm_ok": None,
            "llm_judgment_1": {"tokens": [dict(result)]}, "llm_judgment_2": {"tokens": [dict(result)]}}


def test_per_token_coverage_strict_booleans_and_both_runs_required():
    row = judgment_row()
    assert both_runs_ok(row) is True
    row["llm_judgment_2"]["tokens"][0]["tag_ok"] = False
    assert both_runs_ok(row) is False
    row["llm_judgment_2"] = None
    assert both_runs_ok(row) is None
    for tokens in ([], [row["llm_judgment_1"]["tokens"][0]] * 2):
        with pytest.raises(ValueError):
            validate_judgment({"tokens": tokens}, row)
    value = row["llm_judgment_1"]["tokens"][0]
    with pytest.raises(ValueError):
        validate_judgment({"tokens": [{**value, "tag_ok": "true"}]}, row)


def test_judgments_are_separate_requests_with_same_prompt_and_shared_ledger(tmp_path):
    config = load_config()
    config["paths"]["phase2_output"] = tmp_path
    config["ec_judge"]["model"] = "gpt-6.1-sol"
    write_json(tmp_path / "models.json", {"selected_model": "gpt-6.1-sol"})
    row = judgment_row()
    calls = []
    class Client:
        def __init__(self, cfg, on_record):
            assert cfg.reasoning_effort == "high" and cfg.model == "gpt-6.1-sol"
            self.record = on_record
        def _load(self):
            pass
        def start_sample(self, sample_id):
            pass
        def __call__(self, **kwargs):
            calls.append(kwargs)
            self.record({"response_id": str(len(calls)), "status": "completed", "usage": {}})
            return {"tokens": row["llm_judgment_1"]["tokens"]}
        def close(self):
            pass
    api = Phase2API(config, 2)
    first = api.judge(row, 1, client_factory=Client)
    row["llm_judgment_1"] = first
    second = api.judge(row, 2, client_factory=Client)
    assert first["phase_call"] == 1 and second["phase_call"] == 2
    assert first["response_id"] != second["response_id"]
    assert calls[0] == calls[1]
    assert set(judge_payload(row)) == {"sentence", "morphemes", "ec_tokens"}
    assert row["human_ok"] is None
    with pytest.raises(CallBudgetExceeded):
        api.judge(row, 2, client_factory=Client)
    assert len(calls) == 2


def test_cost_gate_counts_reasoning_once_and_requires_under_twenty():
    records = [{"stage": "ec_judgment", "sentence_id": f"valid:{i}:S1",
                "usage": {"input_tokens": 1000, "output_tokens": 2000,
                          "output_tokens_details": {"reasoning_tokens": 1500}}}
               for i in range(10) for _ in range(2)]
    rows = [judgment_row(f"valid:{i}:S1") for i in range(10)]
    usage = token_usage(records)
    assert usage["cost_usd"] == pytest.approx(.44)
    assert usage["reasoning_tokens"] == 30000
    gate = pilot_projection(rows, records)
    assert gate["projected_total_usd"] == pytest.approx(4.4)
    assert gate["continue_allowed"]
    assert not pilot_projection(rows, records, cost_limit=4.4)["continue_allowed"]
    with pytest.raises(ValueError, match="usage is incomplete"):
        pilot_projection(rows, records[:-1])


def test_agreement_uses_all_token_booleans_but_not_free_text_notes():
    a, b = judgment_row(), judgment_row("valid:2:S1")
    a["llm_judgment_2"]["tokens"][0]["note"] = "different explanation"
    b["llm_judgment_2"]["tokens"][0]["relation_candidates_ok"] = False
    result = agreement([a, b])
    assert result["sentence_agreement_rate"] == .5
    assert result["llm_ok_rate"] == .5
    assert result["fields"]["tag_ok"]["llm_ok_token_rate"] == 1
    assert result["disagreement_sentence_ids"] == ["valid:2:S1"]


def test_ec_sampling_is_seeded_over_sentences_independent_of_input_order():
    pool = [{"source_line": i // 10, "sid": f"S{i % 10 + 1}"} for i in range(200)]
    assert sample_ec(pool) == sample_ec(reversed(pool))
    assert len(sample_ec(pool)) == 100
    with pytest.raises(ValueError):
        sample_ec(pool[:99])
