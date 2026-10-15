from collections import Counter
from copy import deepcopy

import pytest

from verak.v3.averaged_calibration import choose_average_k, summarize_averaged
from verak.v3.common import load_config, read_json
from verak.v3.data_policy import assert_split_integrity, split_questions
from verak.v3.score.averaging import average_prefix, whitespace_probes
from verak.v3.tests.test_score import make_scorer


def test_split_has_twenty_percent_dev_questions_in_every_genre():
    labels = {f"{genre}-{i:03}": genre for genre, n in (("설명", 93), ("논증", 35), ("정서", 37)) for i in range(n)}
    split = split_questions(labels, genres=labels, seed=13)
    assert_split_integrity(split, labels)
    assert Counter(labels[q] for q in split["agent_dev"]) == {"설명": 19, "논증": 7, "정서": 7}
    assert split == split_questions(reversed(labels), genres=dict(reversed(list(labels.items()))), seed=13)
    with pytest.raises(ValueError, match="cover"):
        split_questions(labels, genres={})


def test_real_resplit_keeps_old_manifest_and_all_essays():
    data = load_config()["paths"]["metadata"]
    if not (data / "splits_v1.json").exists():
        pytest.skip("Phase 1b private split not prepared")
    old, new = read_json(data / "splits_v1.json"), read_json(data / "splits.json")
    assert_split_integrity(new, old["agent_train"] + old["agent_dev"])
    assert new["counts"]["agent_dev"]["genres_questions"] == {"설명": 19, "논증": 7, "정서": 7}
    assert sum(new["counts"][x]["essays"] for x in ("agent_train", "agent_dev")) == 8000
    assert old["counts"]["agent_dev"]["genres_questions"]["논증"] == 3


def test_whitespace_sets_are_nested_repeatable_and_never_edit_markers():
    text = "#@사람 이름# 학생은 물의  상태를 관찰하고 결과를 자세히 기록했다.\n다음 문단이다."
    five = whitespace_probes(text, 5, seed=13)
    assert five[:3] == whitespace_probes(text, 3, seed=13)
    assert five[:1] == whitespace_probes(text, 1, seed=13)
    assert five == whitespace_probes(text, 5, seed=13)
    assert len({p.text for p in five}) == 5
    for probe in five[1:]:
        p = probe.position
        assert p >= len("#@사람 이름#")
        assert len(probe.text) == len(text) + 1
        assert probe.text[:p] + probe.text[p+1:] == text
        assert "#@사람 이름#" in probe.text and probe.text.count("\n") == 1
    for invalid in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            whitespace_probes(text, invalid)
    with pytest.raises(ValueError, match="Not enough"):
        whitespace_probes("공백없음", 3)


def test_averaged_score_uses_true_calls_and_default_k_without_changing_base_cache(tmp_path):
    scorer, model = make_scorer(tmp_path)
    fingerprint = scorer.fingerprint
    scorer.config["scorer"].update(average_k=5, average_seed=13)
    text = "one two three four five six"
    first = scorer.score_averaged("question", text, use_cache=False)
    second = scorer.score_averaged("question", text, 5, use_cache=False)
    assert model.generations == model.forwards == 10
    assert first.mean == second.mean == average_prefix(first.members, 5)
    assert first.positions == second.positions
    assert all(not r["cache_hit"] for r in first.members + second.members)
    one = scorer.score_averaged("question", text, 1)
    assert one.mean == first.members[0]["mean"] and model.generations == 10
    config = deepcopy(scorer.config)
    tokenizer = scorer.tokenizer
    scorer.close()
    from verak.v3.score.kanana import KananaScorer
    reopened = KananaScorer(config, model=model, tokenizer=tokenizer)
    assert reopened.fingerprint == fingerprint
    reopened.close()


def test_gate_requires_25_percent_reduction_and_determinism():
    assert choose_average_k({1: 1., 3: .9, 5: .8}, True)["chosen_k"] == 1
    assert not choose_average_k({1: 1., 3: .9, 5: .8}, True)["proceed_phase2"]
    selected = choose_average_k({1: 1., 3: .8, 5: .75}, True)
    assert selected["chosen_k"] == 3 and selected["proceed_phase2"]
    assert choose_average_k({1: 1., 3: .84, 5: .75}, True)["chosen_k"] == 5
    assert choose_average_k({1: 1., 3: .5, 5: .5}, False)["reason"] == "determinism_failed"


def test_averaged_noise_is_signed_and_timing_excludes_cached_calls():
    records = []
    for i, sign in enumerate((-1, 1)):
        for variant in ("original", "whitespace", "paraphrase"):
            values = [5.] * 5 if variant != "paraphrase" else [5. + sign, 5., 5., 5., 5.]
            records.append({"id": str(i), "variant": variant, "genre": "논증", "result": {
                "k": 5, "members": [{"mean": x, "cache_hit": False} for x in values],
                "member_seconds": [1.] * 5}})
    result = summarize_averaged(records, repeat_records=[{"identical": True}], expected_ids=["0", "1"])
    assert result["ks"]["1"]["noise_floor"] == 2
    assert result["ks"]["5"]["noise_floor"] == pytest.approx(.4)
    assert result["ks"]["3"]["scoring_time_per_essay"]["overall"]["mean_seconds"] == 3
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_averaged(records[:-1], repeat_records=[{"identical": True}], expected_ids=["0", "1"])
    records[0]["result"]["members"][0]["cache_hit"] = True
    with pytest.raises(ValueError, match="uncached"):
        summarize_averaged(records, repeat_records=[{"identical": True}], expected_ids=["0", "1"])
