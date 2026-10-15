"""Record/restore contracts using real Bareun observations of new synthetic text."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random

import pytest

from verak.src.schemas import Token
from verak.v3.api import PhaseBudget
from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.common import load_config, read_json, sha_text
from verak.v3.corrupt.document import BareunBank, Document, MARKER, positional_changes, protected
from verak.v3.corrupt.operators import (LEVELS, Proposal, apply, candidates,
    exact_restoration_satisfies, recovery_target, restore_record, local_verified)
from verak.v3.corrupt.builder import composition, curriculum_attempts, source_coupled_changes
from verak.v3.corrupt.qc import qc_payload, qc_results, select_qc, validate_judgments
from verak.v3.phase2 import restore_profile

FIXTURE = Path(__file__).with_name("fixtures") / "corrupt_bareun.json"


@pytest.fixture
def sample(tmp_path):
    saved = read_json(FIXTURE)
    config = load_config()
    config["paths"]["phase3_output"] = tmp_path
    bank = BareunBank(config)
    bank.memory = {text: [Token(**t) for t in tokens] for text, tokens in saved["tokens"].items()}
    def tokens(text):
        if text not in bank.memory:
            raise ValueError("Offline fixture has no such Bareun observation")
        return deepcopy(bank.memory[text])
    bank.tokens = tokens
    doc = Document.from_profile(saved["text"], restore_profile(saved["profile"]))
    return doc, bank, saved


@pytest.mark.parametrize("op", list(LEVELS))
def test_every_operator_apply_record_restore_target(sample, op):
    doc, bank, saved = sample
    proposal = Proposal(**saved["proposals"][op])
    changed, record = apply(doc, proposal, bank)
    assert changed.text != doc.text
    assert record["op"] == op and record["level"] == LEVELS[op]
    assert record["sids"] and record["recovery_target"]
    assert not exact_restoration_satisfies(changed, record)
    restored = restore_record(changed, record, bank)
    assert restored.text == doc.text
    assert restored.snapshot() == doc.snapshot()
    assert exact_restoration_satisfies(restored, record)
    assert "antecedent" not in str(record["coupled_changes"])
    if op.startswith("L_"):
        assert record["verification"] == "bareun_reanalysis"


def test_inverses_require_correct_order_and_hash(sample):
    doc, bank, saved = sample
    changed, record = apply(doc, Proposal(**saved["proposals"]["L_REGISTER"]), bank)
    with pytest.raises(ValueError, match="reverse"):
        restore_record(doc, record, bank)
    record["inverse"]["tail"] += " "
    with pytest.raises(ValueError, match="corrupt"):
        restore_record(changed, record, bank)


@pytest.mark.parametrize("span,expected", [((0,1),False), ((2,3),True), ((1,9),True),
    ((3,3),True), ((1,1),False), ((8,8),False)])
def test_marker_boundary_protection(span, expected):
    assert protected("x#@개체명#z", *span) is expected


@pytest.mark.parametrize("op", [op for op in LEVELS if op.startswith("L_")] + ["G_VAGUE"])
def test_multi_unit_has_no_internal_candidates(sample, op):
    doc, _, _ = sample
    for unit in doc.units:
        unit.tokens.extend([Token("다", "EF", len(unit.text)-1, len(unit.text))]*2)
    assert not candidates(doc, op)


def test_closed_conjunctions_and_coarse_classes_differ(sample):
    doc, _, _ = sample
    for op in ("L_CONN", "L_CONJ"):
        proposals = candidates(doc, op)
        assert proposals
        assert all(p.params["coarse_before"] != p.params["coarse_after"] for p in proposals)


def test_delete_is_never_paragraph_first(sample):
    doc, _, _ = sample
    assert all(doc.locate(p.sids[0])[1] > 0 for p in candidates(doc, "G_DELETE_SUPPORT"))


def test_combined_recovery_targets_anchor_to_source_not_intermediate_position(sample):
    doc, bank, _ = sample
    # Removing S5 shifts S6 within its paragraph. S6's recovery site must still
    # be its source position, otherwise restoring both records becomes impossible.
    deleted, _ = apply(doc, Proposal("G_DELETE_SUPPORT", ["S5"]), bank)
    proposal = Proposal("G_VAGUE", ["S6"], {}, (0, len(doc.locate("S6")[2].text)), "unused")
    originals = {"S6": doc.locate("S6")[2].text}
    source_target = recovery_target(doc, proposal, doc.structure(), originals)
    intermediate_target = recovery_target(deleted, proposal, deleted.structure(), originals)
    assert source_target["position"] != intermediate_target["position"]
    assert exact_restoration_satisfies(doc, {"sids": ["S6"], "recovery_target": source_target})
    assert not exact_restoration_satisfies(doc, {"sids": ["S6"], "recovery_target": intermediate_target})


def test_paragraph_ops_require_three_paragraphs(sample):
    doc, _, _ = sample
    doc.paragraphs, doc.gaps = doc.paragraphs[:2], doc.gaps[:2]
    assert not candidates(doc, "G_PARA_SWAP")
    assert not candidates(doc, "G_SENT_MOVE")


def test_offtopic_never_uses_same_question(sample):
    doc, _, saved = sample
    donor = saved["proposals"]["G_OFFTOPIC"]["params"]["donor"]
    assert not candidates(doc, "G_OFFTOPIC", donors=[donor], question_hash=donor["question_hash"])


def test_coupled_edges_are_positions_not_antecedent_claims(sample):
    doc, bank, saved = sample
    changed, record = apply(doc, Proposal(**saved["proposals"]["G_PARA_SWAP"]), bank)
    assert record["coupled_changes"] == positional_changes(doc.structure(), changed.structure())
    for value in record["coupled_changes"]:
        assert value["previous_before"] != value["previous_after"]
        assert value["kind"] in {"CONJ", "DEP"}
        if value["kind"] == "DEP":
            assert value["interpretation"] == "positional_hint"
            assert value["relative_weight"] < 1


def test_inserted_sentence_has_no_source_dependency_target(sample):
    doc, bank, _ = sample
    donor = {"source_id": "synthetic:2", "question_hash": "other", "sid": "S1",
             "text": doc.locate("S8")[2].text, "style": "한다"}
    inserted, _ = apply(doc, Proposal("G_OFFTOPIC", ["I1"],
        {"donor": donor, "paragraph": 0, "position": 3}), bank)
    moved, record = apply(inserted, Proposal("G_SENT_MOVE", ["S3"],
        {"from_paragraph": 0, "from_position": 2, "to_paragraph": 1, "to_position": 3}), bank)
    assert any(c["sid"] == "I1" for c in record["coupled_changes"])
    anchored = source_coupled_changes(record["coupled_changes"], doc.structure())
    assert all(c["sid"] != "I1" for c in anchored)


def test_curriculum_compositions():
    rng = random.Random(13)
    for _ in range(100):
        assert composition("L1", rng) == (0,1)
        assert composition("L2", rng) in {(0,2), (0,3)}
        assert composition("L3", rng) in {(1,0), (1,1)}
        assert composition("L4", rng) in {(2,1), (2,2)}


def test_incompatible_curriculum_uses_only_explicit_allowed_levels():
    values = curriculum_attempts("L1", ("L1", "L3"))
    assert len(values) == 80
    assert all(v[0] == "L1" and v[2] == 0 for v in values[:40])
    assert all(v[0] == "L3" and v[2] == 1 for v in values[40:])
    assert {v[0] for v in curriculum_attempts("L1", ("L1",))} == {"L1"}


def test_phase_budget_shared_and_fail_closed(tmp_path):
    path = tmp_path / "budget.json"
    first = PhaseBudget(path, 2, authorized_ceiling=100, phase="v3_phase3_corruption")
    second = PhaseBudget(path, 2, authorized_ceiling=100, phase="v3_phase3_corruption")
    assert first.reserve() == 1
    assert second.reserve() == 2
    with pytest.raises(CallBudgetExceeded):
        first.reserve()
    with pytest.raises(ValueError):
        PhaseBudget(path, 101, authorized_ceiling=100)


def test_qc_each_record_has_exact_transition_context(sample):
    doc, bank, saved = sample
    changed, record = apply(doc, Proposal(**saved["proposals"]["L_REGISTER"]), bank)
    record["record_id"] = "r1"
    payload = qc_payload({"question": "q", "genre": "설명", "records": [record],
                          "corrupted_layout": changed.snapshot()})
    assert payload["records"][0]["before_document"] == doc.text
    assert payload["records"][0]["after_document"] == changed.text
    assert "op" not in payload["records"][0]  # No expected damage label or score given to judge.


def test_qc_missing_or_duplicate_records_rejected():
    row = {"records": [{"record_id": "a"}, {"record_id": "b"}]}
    for ids in (("a",), ("a","a"), ("a","b","c")):
        with pytest.raises(ValueError):
            validate_judgments(row, {"judgments": [{"record_id": i} for i in ids]})


def test_qc_either_field_below_threshold_disables_operator():
    rows = [{"records": [{"op": "L_CONN", "record_id": str(i)}], "llm_judgment": {
        "judgments": [{"record_id": str(i), "damage_real": i < 4, "original_is_fix": i < 3, "note": ""}]}}
        for i in range(5)]
    result = qc_results(rows, [])
    assert result["operators"]["L_CONN"]["damage_real"] == .8
    assert "L_CONN" in result["disabled_operators"]
    assert "G_VAGUE" in result["disabled_operators"]  # Unobserved != passed.


def test_qc_sampling_is_deterministic_and_source_distinct():
    ops = list(LEVELS)
    pool = [{"episode_id": str(i), "source_id": str(i//2), "split": "agent_dev",
             "level": "L"+str(i%4+1), "records": [{"op": ops[(i+j)%len(ops)]} for j in range(3)]}
            for i in range(240)]
    first = select_qc(pool)
    assert first == select_qc(list(reversed(pool)))
    assert len(first) == len({r["source_id"] for r in first}) == 60


def test_final_builder_stops_if_qc_removes_an_entire_level(tmp_path, monkeypatch):
    import sys
    from verak.v3.cli import build_corruptions as cli
    config = load_config()
    config["paths"]["phase3_output"] = tmp_path
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "read_json", lambda _: {"essays": 60, "complete": True,
        "disabled_operators": [op for op, level in LEVELS.items() if level == "SENTENCE"]})
    def forbidden(*args, **kwargs):
        pytest.fail("A final dataset was built despite losing all SENTENCE operators")
    monkeypatch.setattr(cli, "build_dataset", forbidden)
    monkeypatch.setattr(sys, "argv", ["build", "--split", "agent_dev", "--max-api-calls", "0",
                                     "--out", str(tmp_path / "final.jsonl")])
    with pytest.raises(ValueError, match="SENTENCE"):
        cli.main()
