import json
from pathlib import Path

import pytest

from verak.v3.common import Example, extract_question_essay, load_config, read_json, sha_text, write_json
from verak.v3.data_policy import (assert_data_tree_clean, assert_not_training_essay,
    assert_split_integrity, feedback_headings, load_examples, split_questions, stratified_sample)


def test_question_split_is_disjoint_complete_and_repeatable():
    questions = [f"q{i}" for i in range(25)] + ["q0"] * 10
    split = split_questions(questions, seed=13)
    assert_split_integrity(split, questions)
    assert len(split["agent_train"]) == 20 and len(split["agent_dev"]) == 5
    assert split == split_questions(reversed(questions), seed=13)
    with pytest.raises(ValueError, match="leakage"):
        assert_split_integrity({"agent_train": ["same"], "agent_dev": ["same"]})
    with pytest.raises(ValueError, match="coverage"):
        assert_split_integrity(split, questions + ["missing"])


def test_training_essay_guard_checks_exact_text_in_nested_data_files(tmp_path):
    forbidden = "학습에 이미 사용한 문장입니다.\n두 번째 문단입니다."
    deny = {sha_text(forbidden)}
    write_json(tmp_path / "metadata.json", {"hash": sha_text(forbidden)})
    assert_data_tree_clean(tmp_path, deny)
    write_json(tmp_path / "bad.json", {"deep": [{"arbitrary_name": forbidden}]})
    with pytest.raises(ValueError, match="training essay"):
        assert_data_tree_clean(tmp_path, deny)
    with pytest.raises(ValueError):
        assert_not_training_essay(forbidden, deny)


def test_source_parser_does_not_leak_feedback_keywords_or_normalize_essay():
    row = {"user": "질문: 문항\n에세이: 글  내용\n다음 문단\n핵심 키워드: 비밀",
           "assistant": "gold must not be visible", "grader_1_scores": [9] * 8}
    assert extract_question_essay(row) == ("문항", "글  내용\n다음 문단")


def test_stratified_selection_represents_every_present_genre():
    examples = [Example(i, "q", f"text{i}", "qh", f"eh{i}", genre, False)
                for i, genre in enumerate(["설명"] * 20 + ["논증"] * 7 + ["정서"] * 5)]
    selected = stratified_sample(examples, 12, 13)
    assert len({e.id for e in selected}) == 12
    assert {e.genre for e in selected} == {"설명", "논증", "정서"}
    assert selected == stratified_sample(reversed(examples), 12, 13)


def test_real_manifests_have_no_leakage_and_no_training_essays():
    config = load_config()
    directory = config["paths"]["metadata"]
    if not (directory / "splits.json").exists():
        pytest.skip("Run prepare_phase1 to build the private corpus metadata")
    splits = read_json(directory / "splits.json")
    assert_split_integrity(splits)
    train, dev = load_examples(config, "agent_train"), load_examples(config, "agent_dev")
    assert len(train) + len(dev) == 8000
    assert {e.question_hash for e in train}.isdisjoint(e.question_hash for e in dev)
    deny = set(read_json(directory / "audit_index.json")["train"]["essay_hashes"])
    for example in train + dev:
        assert_not_training_essay(example.text, deny)
    assert_data_tree_clean(directory, deny)
    excluded = read_json(directory / "eval_exclusions.json")
    assert [r["row"] for r in excluded["rows"]] == [4064, 4406, 4416, 4599, 4605, 4625, 4926, 4945, 5334, 5353, 5688]
    assert excluded["valid_exclusions"] == []
    provenance = read_json(directory / "question_provenance.json")
    assert sum(x["seen_by_scorer"] for x in provenance["valid"].values()) == 164
    assert sum(x["seen_by_scorer"] for x in provenance["test"].values()) == 107
    for bad in ("test", "train", "scorer_test"):
        with pytest.raises(ValueError, match="before Phase 9"):
            load_examples(config, bad)
