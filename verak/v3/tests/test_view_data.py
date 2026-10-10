import pytest

from verak.v3.common import Example, load_config, read_json
from verak.v3.view_data import filter_episode_examples, load_episode_examples, percentiles


def test_hard_view_limit_excludes_without_truncating_and_requires_current_hash():
    essays = [Example(i, "문항", "원문" * i, "qh", f"hash{i}", "논증", False) for i in range(1, 4)]
    records = {essay.id: {"essay_hash": essay.essay_hash, "compact_tokens": length}
               for essay, length in zip(essays, [2999, 3000, 3001])}
    assert filter_episode_examples(essays, records) == essays[:2]
    assert essays[2].text == "원문원문원문"
    records[essays[0].id]["essay_hash"] = "stale"
    with pytest.raises(ValueError, match="Missing or stale"):
        filter_episode_examples(essays, records)
    with pytest.raises(ValueError):
        filter_episode_examples(essays, {})


def test_distribution_linear_percentiles():
    result = percentiles([1, 2, 3, 4, 100])
    assert result["median"] == 3
    assert result["p90"] == pytest.approx(61.6)
    assert result["max"] == 100


def test_episode_data_requires_eligibility_index(tmp_path):
    config = load_config()
    config["paths"]["metadata"] = tmp_path
    with pytest.raises(ValueError, match="eligibility must be prepared"):
        load_episode_examples(config)


def test_prepared_episode_splits_honor_budget_and_keep_questions_disjoint():
    config = load_config()
    path = config["paths"]["metadata"] / "view_eligibility.json"
    if not path.exists():
        pytest.skip("Prepare private compact-view eligibility first")
    index = read_json(path)
    train, dev = load_episode_examples(config, "agent_train"), load_episode_examples(config, "agent_dev")
    assert {example.question_hash for example in train}.isdisjoint(example.question_hash for example in dev)
    assert all(index["essays"][example.id]["compact_tokens"] <= 3000 for example in train + dev)
    assert len(train) + len(dev) == sum(row["compact_tokens"] <= 3000 for row in index["essays"].values())
    assert config["view"]["format"] == "compact" and config["view_token_budget"] == 3000
