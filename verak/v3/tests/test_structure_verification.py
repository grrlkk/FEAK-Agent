"""Phase 2b call accounting, independent prompts and gate denominators."""

from copy import deepcopy

import pytest
from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.common import load_config
from verak.v3.phase2b import input_hash
from verak.v3.structure_metrics import pilot_projection, rates, summarize
from verak.v3.structure_verification import StructureAPI, both_ok, payload, prompt_hash, validate


def row(check="WORD", subgroup="relation"):
    return {"item_id": "test:S1", "sentence_id": "test:S1", "sid": "S1", "sentence": "가면 좋다.",
            "previous_sentences": [], "morphemes": [], "check": check, "subgroup": subgroup,
            "unambiguous_ec": [{"token_id": "M2", "form": "면", "relation": "CONDITION"}],
            "polarity_modality": None, "pilot": True,
            "reference": {"target": "S0", "label": "학생", "confidence": "HIGH"},
            "initial_conjunction": {"form": "그러나", "relation": "CONTRAST"},
            "style": "한다", "dominant_style": "한다",
            "llm_judgment_1": None, "llm_judgment_2": None, "llm_ok": None, "human_ok": None}


def test_same_prompt_no_previous_verdict_or_confidence_leak():
    item = row("SENTENCE", "HIGH")
    first, hashed = payload(item), prompt_hash(item)
    original = input_hash([item])
    item["llm_judgment_1"] = {"antecedent_ok": False, "note": "do not show this"}
    assert payload(item) == first and prompt_hash(item) == hashed
    assert "confidence" not in payload(item)["reference"]
    assert input_hash([item]) == original


def test_na_is_not_a_success_and_cannot_hide_an_applicable_field():
    item = row()
    value = {"relation_ok": True, "polarity_modality_ok": None, "note": "관계 적합"}
    validate(value, item)
    with pytest.raises(ValueError): validate({**value, "relation_ok": None}, item)
    with pytest.raises(ValueError): validate({**value, "polarity_modality_ok": True}, item)
    item["llm_judgment_1"] = item["llm_judgment_2"] = value
    assert both_ok(item)
    result = rates([item])
    assert set(result["fields"]) == {"relation_ok"}
    assert result["fields"]["relation_ok"]["n"] == 1
    assert not summarize([item], [])["phase3_gates"]["WORD"]["passed"]


def test_two_requests_reserve_shared_budget_and_record_usage(tmp_path):
    cfg = load_config()
    cfg["paths"]["phase2b_output"] = tmp_path
    api = StructureAPI(cfg, 2)
    class Fake:
        def __init__(self, cfg, cb): self.cb = cb
        def _load(self): pass
        def start_sample(self, sid): pass
        def close(self): pass
        def __call__(self, **kwargs):
            self.cb({"usage": {"input_tokens": 100, "output_tokens": 80,
                              "output_tokens_details": {"reasoning_tokens": 50}}, "response_id": str(api.budget.used)})
            return {"relation_ok": True, "polarity_modality_ok": None, "note": "확인"}
    item = row()
    a = api.judge(item, 1, client_factory=Fake)
    b = api.judge(item, 2, client_factory=Fake)
    assert a["response_id"] != b["response_id"] and a["prompt_sha256"] == b["prompt_sha256"]
    assert StructureAPI(cfg, 2).budget.used == 2
    with pytest.raises(CallBudgetExceeded): api.judge(item, 1, client_factory=Fake)


def test_pilot_weighted_projection_and_missing_usage_fail_closed():
    rows, records = [], []
    for check, subgroup, field, cost_scale in (("WORD", "relation", "relation_ok", 1),
            ("SENTENCE", "HIGH", "antecedent_ok", 2), ("TEXT", "style", "style_ok", 3)):
        for i in range(5):
            item = row(check, subgroup)
            item["item_id"] = f"{check}:{i}"
            item["llm_judgment_1"] = item["llm_judgment_2"] = {field: True}
            rows.append(item)
            for run in (1, 2):
                records.append({"stage": "structure_judgment", "item_id": item["item_id"], "check": check,
                    "usage": {"input_tokens": 1000 * cost_scale, "output_tokens": 100,
                              "output_tokens_details": {"reasoning_tokens": 80}}})
    gate = pilot_projection(rows, records)
    assert gate["continue_allowed"]
    assert gate["projected_total_usd"] == pytest.approx(.003 * 160 + .005 * 240 + .007 * 160)
    assert gate["usage"]["reasoning_tokens"] == 2400 and gate["usage"]["output_tokens"] == 3000
    with pytest.raises(ValueError): pilot_projection(rows, records[:-1])
    missing = deepcopy(rows)
    missing[0]["llm_judgment_2"] = None
    with pytest.raises(ValueError): pilot_projection(missing, records)
