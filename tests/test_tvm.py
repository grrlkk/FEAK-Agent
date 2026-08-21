import numpy as np
import pytest

from feak_tc.tvm.data import (
    SCORER_DERIVED_FEATURES,
    TVM_FEATURE_VARIANTS,
    encode_prompt_sections,
    load_pair_similarities,
    make_tvm_split,
    pair_id,
    split_sha256,
    transition_prompt,
)
from feak_tc.tvm.baselines import run_fixed_baselines
from feak_tc.tvm.comparison import paired_accuracy_comparison
from feak_tc.tvm.selection import select_best_runs
from feak_tc.tvm.training import pairwise_margin_loss


def test_tvm_split_is_deterministic_complete_and_essay_disjoint():
    rows = [_row(index, essay=f"essay-{index // 2}") for index in range(80)]
    first = make_tvm_split(rows, folds=5, test_fold=1, validation_fold=2, seed=17)
    second = make_tvm_split(rows, folds=5, test_fold=1, validation_fold=2, seed=17)
    assert split_sha256(first) == split_sha256(second)
    assert sum(first["summary"][name]["pairs"] for name in ("train", "validation", "test")) == len(rows)
    groups = {
        name: {rows[index]["essay_id"] for index in first[name]}
        for name in ("train", "validation", "test")
    }
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_scorer_free_prompt_omits_only_scorer_outputs():
    pair = _pair()
    full = transition_prompt(pair, "chosen", feature_variant="full")["features"]
    scorer_free = transition_prompt(
        pair, "chosen", feature_variant="scorer_free"
    )["features"]
    for feature in SCORER_DERIVED_FEATURES:
        assert f"{feature}=" in full
        assert f"{feature}=" not in scorer_free
    assert "target_gap_reduction=" in scorer_free
    assert set(TVM_FEATURE_VARIANTS["full"]) - set(
        TVM_FEATURE_VARIANTS["scorer_free"]
    ) == set(SCORER_DERIVED_FEATURES)


def test_balanced_encoding_preserves_both_ends_of_before_and_after():
    tokenizer = _CharacterTokenizer()
    encoded = encode_prompt_sections(
        tokenizer,
        {
            "instruction": "평가",
            "question": "질문" * 100,
            "plan": "action=TEST",
            "before": "A" * 600 + "B" * 600,
            "after": "C" * 600 + "D" * 600,
            "features": "edit_ratio=+0.1",
            "suffix": "[가치 표현]",
        },
        max_length=512,
    )
    assert len(encoded["input_ids"]) <= 512
    for character in "ABCD":
        assert tokenizer.encode(character)[0] in encoded["input_ids"]
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])


def test_verified_similarity_cache_requires_exact_pair_order(tmp_path):
    rows = [_row(0), _row(1)]
    path = tmp_path / "states.npz"
    embeddings = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        pair_ids=np.asarray([pair_id(row) for row in rows]),
        embeddings=embeddings,
        model=np.asarray("BAAI/bge-m3"),
        snapshot=np.asarray("snapshot"),
        prompt_digest=np.asarray("prompt"),
    )
    similarities, info = load_pair_similarities(path, rows)
    assert similarities[pair_id(rows[0])] == pytest.approx(1.0)
    assert similarities[pair_id(rows[1])] == pytest.approx(0.0)
    assert info["pairs"] == 2
    with pytest.raises(ValueError, match="pair IDs"):
        load_pair_similarities(path, list(reversed(rows)))


def test_pairwise_margin_loss_rewards_a_larger_chosen_gap():
    torch = pytest.importorskip("torch")
    margins = torch.tensor([0.1])
    good = pairwise_margin_loss(
        torch.tensor([2.0]), torch.tensor([0.0]), margins
    )
    bad = pairwise_margin_loss(
        torch.tensor([0.0]), torch.tensor([2.0]), margins
    )
    assert good < bad


def test_validation_selection_never_uses_test_metrics_and_checks_lr_grid():
    reports = [
        _validation_report(1e-6, accuracy=0.8, loss=0.5),
        _validation_report(2e-6, accuracy=0.9, loss=0.7),
        _validation_report(3e-6, accuracy=0.9, loss=0.4),
    ]
    reports[0]["test"] = {"pairwise_accuracy": 1.0}
    selected = select_best_runs(
        reports, expected_learning_rates=[1e-6, 2e-6, 3e-6]
    )
    assert selected[0]["learning_rate"] == 3e-6
    with pytest.raises(ValueError, match="incomplete LR sweep"):
        select_best_runs(reports[:2], expected_learning_rates=[1e-6, 2e-6, 3e-6])
    reports[0]["test_split_evaluated"] = True
    with pytest.raises(ValueError, match="test-evaluated"):
        select_best_runs(reports)


def test_fixed_baselines_fit_train_and_cover_validation_and_test():
    pytest.importorskip("lightgbm")
    rows = []
    pairs = []
    embeddings = np.zeros((40, 2, 4), dtype=np.float32)
    for index in range(40):
        row = {
            **_row(index),
            "reverse_action": f"ACTION_{index % 4}",
        }
        rows.append(row)
        pair = _pair()
        pair.update(
            {
                "pair_id": pair_id(row),
                "essay_id": row["essay_id"],
                "corruption_op": row["corruption_op"],
            }
        )
        pair["chosen"]["features"].update(
            {"action_type": row["reverse_action"], "target_rubric": "content_1"}
        )
        pair["rejected"]["features"].update(
            {"action_type": row["reverse_action"], "target_rubric": "content_1"}
        )
        pairs.append(pair)
        embeddings[index, 0, index % 4] = 1.0
        embeddings[index, 1, index % 4] = -1.0
    split = make_tvm_split(rows, folds=5, test_fold=1, validation_fold=2, seed=5)
    report, predictions = run_fixed_baselines(
        rows,
        pairs,
        embeddings,
        split,
        seed=5,
        heuristic_config={
            "rubric_score_range": 8,
            "weights": {
                "target_gain": 2.0,
                "target_gap_reduction": 1.0,
                "evidence_match": 0.5,
                "goal_preservation": 0.5,
                "non_target_drop": -2.0,
                "edit_ratio": -0.5,
            },
        },
    )
    assert report["fit_split"] == "train only"
    assert set(report["evaluation"]) == {
        "feature_lightgbm_full",
        "feature_lightgbm_scorer_free",
        "bge_m3_action_linear",
        "heuristic_full",
        "heuristic_scorer_free",
        "immediate_target_gain",
    }
    assert {row["split"] for row in predictions} == {"validation", "test"}
    assert all(row["correct"] for row in predictions)


def test_paired_accuracy_comparison_uses_shared_pair_ids():
    left = [
        {"pair_id": "a", "correct": True},
        {"pair_id": "b", "correct": True},
        {"pair_id": "c", "correct": False},
    ]
    right = [
        {"pair_id": "c", "correct": False},
        {"pair_id": "a", "correct": False},
        {"pair_id": "b", "correct": True},
    ]
    result = paired_accuracy_comparison(
        left,
        right,
        left_name="left",
        right_name="right",
        bootstrap_samples=100,
    )
    assert result["accuracy_difference"] == pytest.approx(1 / 3)
    assert result["discordant"] == {
        "left_only_correct": 1,
        "right_only_correct": 0,
    }
    assert result["mcnemar_exact_two_sided_p"] == 1.0
    with pytest.raises(ValueError, match="identical pair IDs"):
        paired_accuracy_comparison(left, right[:2], left_name="left", right_name="right")


def _row(index: int, *, essay: str | None = None) -> dict:
    operators = [
        "DELETE_SPECIFICS",
        "INJECT_LEX_REPEAT",
        "INSERT_OFFTOPIC",
        "SHUFFLE_FLOW",
    ]
    return {
        "essay_id": essay or f"essay-{index}",
        "stage_k": 1,
        "corruption_op": operators[index % len(operators)],
        "target_rubric": "content_1",
    }


def _pair() -> dict:
    features = {
        "target_gain": 1.0,
        "target_gap_reduction": 0.5,
        "non_target_drop": 0.1,
        "evidence_match": 0.9,
        "edit_ratio": 0.1,
        "goal_preservation": 0.95,
        "emb_sim": 0.95,
    }
    return {
        "question": "질문",
        "action_type": "ADD_DETAIL",
        "intent": "근거 보강",
        "target_rubric": "content_1",
        "chosen": {
            "before_text": "전",
            "after_text": "후",
            "features": features,
        },
        "rejected": {
            "before_text": "후",
            "after_text": "전",
            "features": {
                **features,
                "target_gain": -1.0,
                "target_gap_reduction": -0.5,
                "evidence_match": 0.0,
            },
        },
    }


def _validation_report(rate: float, *, accuracy: float, loss: float) -> dict:
    return {
        "gate": "tvm_stage1_validation",
        "test_split_evaluated": False,
        "run_dir": f"/tmp/lr-{rate}",
        "model_key": "qwen",
        "model_role": "primary",
        "feature_variant": "full",
        "learning_rate": rate,
        "seed": 7,
        "data": {"sha256": "data", "prompt_sha256": "prompt"},
        "split": {"digest": "split"},
        "validation": {"pairwise_accuracy": accuracy, "mean_loss": loss},
    }


class _CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) + 10 for character in text]
