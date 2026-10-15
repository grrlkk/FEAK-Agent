"""Budget, fresh samples, independent prompts and diagnostic-only categories."""

from copy import deepcopy

import pytest
from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.common import load_config, read_json
from verak.v3.phase2 import read_jsonl
from verak.v3.phase2b import input_hash
from verak.v3.phase2c import COUNTS
from verak.v3.phase2c_metrics import pilot_projection, rates, summarize
from verak.v3.phase2c_verification import FIELDS, Phase2cAPI, both_ok, payload, prompt_hash


def row(check="omission"):
    return {"item_id": "test:S1", "sentence_id": "test:S1", "sid": "S1", "paragraph": "P2",
        "sentence": "공부한다.", "previous_sentences": [{"sid": "S0", "paragraph": "P1", "sentence": "학생들은 모였다."}],
        "subject_omitted": True, "check": check, "level": "SENTENCE", "pilot": True,
        "llm_judgment_1": None, "llm_judgment_2": None, "llm_ok": None, "human_ok": None}


def test_identical_prompt_no_identity_or_other_judgment():
    r = row()
    first, hashed, frozen = payload(r), prompt_hash(r), input_hash([r])
    r.update(reference={"label": "비공개 선행사"}, llm_judgment_1={"omission_ok": False, "note": "숨길 판단"})
    assert payload(r) == first and prompt_hash(r) == hashed
    r.pop("reference")
    assert input_hash([r]) == frozen
    assert first["previous_sentences"][0]["paragraph"] == "P1"


def test_shared_budget_two_independent_requests_and_failure_count(tmp_path):
    cfg = load_config()
    cfg["paths"]["phase2c_output"] = tmp_path
    api = Phase2cAPI(cfg, 3)
    class Fake:
        def __init__(self, cfg, cb): self.cb = cb
        def _load(self): pass
        def start_sample(self, sid): pass
        def close(self): pass
        def __call__(self, **kwargs):
            self.cb({"usage": {"input_tokens": 100, "output_tokens": 80,
                "output_tokens_details": {"reasoning_tokens": 50}}, "response_id": str(api.budget.used)})
            return {"omission_ok": True, "referent_type": "previous_sentence", "note": "확인"}
    a, b = [api.judge(row(), run, client_factory=Fake) for run in (1, 2)]
    assert a["response_id"] != b["response_id"] and a["prompt_sha256"] == b["prompt_sha256"]
    class Failure(Fake):
        def __call__(self, **kwargs): raise RuntimeError("offline fake error")
    with pytest.raises(RuntimeError): api.judge(row(), 1, client_factory=Failure)
    assert Phase2cAPI(cfg, 3).budget.used == 3
    with pytest.raises(CallBudgetExceeded): api.judge(row(), 1, client_factory=Fake)
    with pytest.raises(ValueError): Phase2cAPI(cfg, 471)


def test_pilot_accounting_reserve_and_missing_call_fail_closed():
    rows, records = [], []
    for check in COUNTS:
        for i in range(5):
            r = row(check)
            r["item_id"] = f"{check}:{i}"
            r["llm_judgment_1"] = r["llm_judgment_2"] = {FIELDS[check]: True}
            rows.append(r)
            for run in (1, 2):
                records.append({"stage": "structure_judgment", "item_id": r["item_id"], "check": check,
                    "usage": {"input_tokens": 1000, "output_tokens": 100,
                              "output_tokens_details": {"reasoning_tokens": 80}}})
    gate = pilot_projection(rows, records)
    assert gate["projected_total_usd"] == pytest.approx(460 * .003)
    assert gate["projection_with_10_reserve_calls_usd"] == pytest.approx(470 * .003)
    assert gate["usage"]["reasoning_tokens"] == 3200 and gate["usage"]["output_tokens"] == 4000
    with pytest.raises(ValueError): pilot_projection(rows, records[:-1])
    missing = deepcopy(rows)
    missing[0]["llm_judgment_2"] = None
    with pytest.raises(ValueError): pilot_projection(missing, records)


def test_referent_distribution_not_a_gate_and_two_run_false_counts_failure():
    r = row()
    r["llm_judgment_1"] = {"omission_ok": True, "referent_type": "previous_sentence"}
    r["llm_judgment_2"] = {"omission_ok": True, "referent_type": "writer_or_generic"}
    assert both_ok(r) and rates([r])["agreement"] == 1
    result = summarize([r], [])
    assert not result["phase3_gates"]["SENTENCE"]["passed"]  # Incomplete sample.
    assert result["referent_type_diagnostic_only"]["predicted_true"]["agreement"] == 0
    r["llm_judgment_2"]["omission_ok"] = False
    assert rates([r])["pass_rate"] == 0 and rates([r])["disagreement_ids"] == ["test:S1"]


def test_private_frozen_sample_quotas_and_freshness():
    cfg = load_config()
    path = cfg["paths"]["metadata"] / "structure_check_phase2c.jsonl"
    if not path.exists():
        pytest.skip("Private verification data unavailable")
    from collections import Counter
    rows = read_jsonl(path)
    assert Counter(r["check"] for r in rows) == COUNTS
    assert Counter(r["subject_omitted"] for r in rows if r["check"] == "omission") == {True: 40, False: 40}
    assert sum(r["cross_paragraph"] for r in rows if r["check"] == "conjunction") >= 10
    styles = [r for r in rows if r["check"] == "style"]
    assert len(Counter(r["essay_id"] for r in styles)) == 15
    assert set(Counter(r["essay_id"] for r in styles).values()) == {4}
    assert sum(r["embedded_or_quoted_ending"] for r in styles) >= 15
    old = {r["sentence_id"] for name in ("ec_check.jsonl", "structure_check_phase2b.jsonl")
           for r in read_jsonl(cfg["paths"]["metadata"] / name)}
    assert len({r["sentence_id"] for r in rows}) == 230
    assert not old & {r["sentence_id"] for r in rows}
    assert all(r["previous_sentences"][-1]["sid"] == r["predecessor_id"] for r in rows if r["predecessor_id"])
    prepared = read_json(cfg["paths"]["phase2c_output"] / "preparation.json")
    assert input_hash(rows) == prepared["input_sha256"]
    assert all(r["human_ok"] is None for r in rows)
