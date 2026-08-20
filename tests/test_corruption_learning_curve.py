import numpy as np
import pytest

from feak_tc.corruption.g2 import TRANSITION_FEATURES
from feak_tc.corruption.learning_curve import (
    assess_generation_need,
    make_grouped_folds,
    run_feature_learning_curve,
    run_text_learning_curve,
    select_nested_train_indices,
)


def test_grouped_folds_have_no_essay_leakage_and_nested_training_sets():
    rows = [_row(index, essay_id=f"essay-{index // 2}") for index in range(40)]
    folds = make_grouped_folds(rows, folds=4, seed=7)
    coverage = []
    for fold in folds:
        train = set(fold["train_indices"])
        test = set(fold["test_indices"])
        train_essays = {rows[index]["essay_id"] for index in train}
        test_essays = {rows[index]["essay_id"] for index in test}
        assert train.isdisjoint(test)
        assert train_essays.isdisjoint(test_essays)
        small = set(select_nested_train_indices(fold, rows, 8))
        large = set(select_nested_train_indices(fold, rows, 16))
        assert small < large <= train
        coverage.extend(test)
    assert sorted(coverage) == list(range(len(rows)))


def test_text_learning_curve_recovers_a_clear_pairwise_direction():
    rows = [_row(index) for index in range(36)]
    folds = make_grouped_folds(rows, folds=3, seed=11)
    embeddings = np.zeros((len(rows), 2, 8), dtype=np.float32)
    embeddings[:, 0, 0] = 1.0
    embeddings[:, 1, 0] = -1.0
    report, predictions = run_text_learning_curve(
        rows,
        embeddings,
        folds,
        train_sizes=[6, 12],
        seed=11,
    )
    assert [point["point"] for point in report["points"]] == ["6", "12", "full"]
    assert all(point["accuracy"] == 1.0 for point in report["points"])
    assert all(row["correct"] for row in predictions)
    decision = assess_generation_need(report)
    assert decision["decision"] == "stop_generation"


def test_text_learning_curve_can_learn_opposite_action_directions():
    rows = [_row(index) for index in range(48)]
    folds = make_grouped_folds(rows, folds=3, seed=12)
    embeddings = np.zeros((len(rows), 2, 4), dtype=np.float32)
    for index, row in enumerate(rows):
        sign = 1.0 if row["corruption_op"] in {
            "DELETE_SPECIFICS",
            "INSERT_OFFTOPIC",
        } else -1.0
        embeddings[index, 0, 0] = sign
        embeddings[index, 1, 0] = -sign
    report, predictions = run_text_learning_curve(
        rows,
        embeddings,
        folds,
        train_sizes=[16],
        seed=12,
        condition_key="corruption_op",
        model_name="conditioned",
    )
    assert report["condition_key"] == "corruption_op"
    assert report["pair_dimensions"] == 16
    assert report["points"][-1]["accuracy"] == 1.0
    assert all(row["correct"] for row in predictions)


def test_feature_learning_curve_recovers_clear_target_gain():
    pytest.importorskip("lightgbm")
    rows = [_row(index) for index in range(36)]
    pairs = [_pair(row) for row in rows]
    folds = make_grouped_folds(rows, folds=3, seed=13)
    report, predictions = run_feature_learning_curve(
        rows,
        pairs,
        folds,
        train_sizes=[8, 16],
        transition_features=TRANSITION_FEATURES,
        seed=13,
    )
    assert report["backend"] == "lightgbm.LGBMRanker"
    assert report["points"][-1]["accuracy"] == 1.0
    assert all(row["correct"] for row in predictions)


@pytest.mark.parametrize(
    ("previous", "final", "expected"),
    [
        (0.80, 0.83, "generate_more"),
        (0.70, 0.71, "investigate_model_or_construct_before_generation"),
    ],
)
def test_generation_decision_distinguishes_growth_from_low_plateau(
    previous: float,
    final: float,
    expected: str,
):
    report = {
        "points": [
            {"point": "600", "accuracy": previous},
            {"point": "full", "accuracy": final},
        ]
    }
    assert assess_generation_need(report)["decision"] == expected


def _row(index: int, *, essay_id: str | None = None) -> dict:
    operators = [
        ("DELETE_SPECIFICS", "content_2"),
        ("INJECT_LEX_REPEAT", "expression_1"),
        ("INSERT_OFFTOPIC", "organization_2"),
        ("SHUFFLE_FLOW", "organization_1"),
    ]
    operator, rubric = operators[index % len(operators)]
    return {
        "essay_id": essay_id or f"essay-{index}",
        "stage_k": 1,
        "corruption_op": operator,
        "target_rubric": rubric,
    }


def _pair(row: dict) -> dict:
    chosen = _features(1.0, 0.0)
    rejected = _features(-1.0, 0.5)
    return {
        "pair_id": f"{row['essay_id']}:stage1",
        "essay_id": row["essay_id"],
        "stage_k": 1,
        "corruption_op": row["corruption_op"],
        "target_rubric": row["target_rubric"],
        "chosen": chosen,
        "rejected": rejected,
    }


def _features(target_gain: float, non_target_drop: float) -> dict:
    return {
        "action_type": "STYLE_REFINE",
        "target_rubric": "expression_1",
        "target_gain": target_gain,
        "target_gap_reduction": target_gain,
        "non_target_drop": non_target_drop,
        "evidence_match": 0.9,
        "edit_ratio": 0.1,
        "goal_preservation": 0.9,
        "emb_sim": 0.9,
    }
