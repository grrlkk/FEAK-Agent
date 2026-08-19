import pytest

from feak_tc.corruption.g2 import (
    TRANSITION_FEATURES,
    build_gbm_pairs,
    build_human_review_pairs,
    evaluate_human_review,
    evaluate_two_human_reviews,
    run_grouped_gbm,
    run_grouped_lightgbm_ranker,
)


def test_g2_builds_nine_measured_transition_features():
    rows = [_audit_row("essay-1", accepted=True)]
    pairs = build_gbm_pairs(
        rows,
        elite_stats={},
        similarity_fn=lambda before, after: (0.9, {"method": "test"}),
    )
    assert len(pairs) == 1
    assert set(pairs[0]["chosen"]) == set(TRANSITION_FEATURES)
    assert pairs[0]["chosen"]["target_gain"] == 1.0
    assert pairs[0]["rejected"]["target_gain"] == -1.0
    assert pairs[0]["similarity"]["method"] == "test"


def test_g2_grouped_gbm_separates_clear_bidirectional_pairs():
    pairs = []
    for index in range(12):
        chosen = _feature_row(target_gain=1.0, non_target_drop=0.0)
        rejected = _feature_row(target_gain=-1.0, non_target_drop=0.5)
        pairs.append(
            {
                "pair_id": f"essay-{index}:stage1",
                "essay_id": f"essay-{index}",
                "corruption_op": "INJECT_GRAMMAR_ERR",
                "target_rubric": "expression_2",
                "target_drop": 1.0,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    report, predictions = run_grouped_gbm(pairs, folds=3, seed=7)
    assert report["pairs"] == 12
    assert report["pairwise_accuracy"] == 1.0
    assert report["significantly_above_random"] is True
    assert all(row["correct"] for row in predictions)


def test_g2_lightgbm_ranker_separates_clear_bidirectional_pairs():
    pytest.importorskip("lightgbm")
    pairs = []
    for index in range(12):
        chosen = _feature_row(target_gain=1.0, non_target_drop=0.0)
        rejected = _feature_row(target_gain=-1.0, non_target_drop=0.5)
        pairs.append(
            {
                "pair_id": f"essay-{index}:stage1",
                "essay_id": f"essay-{index}",
                "stage_gap": 1,
                "corruption_op": "SHUFFLE_FLOW",
                "target_rubric": "organization_1",
                "target_drop": 1.0,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    report, predictions = run_grouped_lightgbm_ranker(pairs, folds=3, seed=7)
    assert report["backend"] == "lightgbm.LGBMRanker"
    assert report["text_used_by_model"] is False
    assert report["split_unit"] == "essay_id"
    assert report["pairwise_accuracy"] == 1.0
    assert report["by_stage_gap"]["1"]["pairs"] == 12
    assert all(row["correct"] for row in predictions)


def test_g2_lightgbm_ranker_can_ablate_target_gain():
    pytest.importorskip("lightgbm")
    pairs = []
    for index in range(12):
        chosen = _feature_row(target_gain=1.0, non_target_drop=0.0)
        rejected = _feature_row(target_gain=-1.0, non_target_drop=0.5)
        pairs.append(
            {
                "pair_id": f"essay-{index}:stage1",
                "essay_id": f"essay-{index}",
                "stage_gap": 1,
                "corruption_op": "SHUFFLE_FLOW",
                "target_rubric": "organization_1",
                "target_drop": 1.0,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    selected = tuple(
        feature for feature in TRANSITION_FEATURES if feature != "target_gain"
    )
    report, predictions = run_grouped_lightgbm_ranker(
        pairs,
        folds=3,
        seed=7,
        transition_features=selected,
    )
    assert "target_gain" not in report["transition_features"]
    assert report["excluded_transition_features"] == ["target_gain"]
    assert all("target_gain" not in row["feature"] for row in report["top_feature_importances"])
    assert report["pairwise_accuracy"] == 1.0
    assert all(row["correct"] for row in predictions)


def test_g2_lightgbm_ranker_rejects_unknown_ablation_feature():
    with pytest.raises(ValueError, match="unknown transition features"):
        run_grouped_lightgbm_ranker(
            [
                {
                    "pair_id": "essay-1:stage1",
                    "essay_id": "essay-1",
                    "chosen": _feature_row(1.0, 0.0),
                    "rejected": _feature_row(-1.0, 0.5),
                }
            ],
            transition_features=("not_a_feature",),
        )


def test_human_review_is_blinded_and_remains_pending_until_filled():
    chain = {
        "record_id": "essay-1",
        "question": "질문",
        "states": ["clean", "bad-1", "bad-2", "bad-3"],
    }
    audit = [
        _audit_row("essay-1", stage=1, accepted=True),
        _audit_row("essay-1", stage=2, accepted=True),
        _audit_row("essay-1", stage=3, accepted=True),
    ]
    review, key, summary = build_human_review_pairs(
        [chain],
        audit,
        count=3,
        seed=7,
    )
    assert summary["available_valid_pairs"] == 6
    assert all("expected_preference" not in row for row in review)
    assert all(row["preference"] == "" for row in review)
    assert evaluate_human_review(review, key, required=3)["status"] == "pending"

    for row, answer in zip(review, key):
        row["preference"] = answer["expected_preference"]
    result = evaluate_human_review(review, key, required=3)
    assert result["status"] == "passed"
    assert result["agreement"] == 1.0


def test_two_human_reviews_require_independent_completion_and_adjudication():
    keys = [
        {"pair_id": "pair-1", "expected_preference": "A"},
        {"pair_id": "pair-2", "expected_preference": "B"},
    ]
    rater_one = [
        {"pair_id": "pair-1", "preference": "A"},
        {"pair_id": "pair-2", "preference": "B"},
    ]
    incomplete_rater_two = [
        {"pair_id": "pair-1", "preference": "A"},
        {"pair_id": "pair-2", "preference": ""},
    ]
    report, disagreements = evaluate_two_human_reviews(
        rater_one,
        incomplete_rater_two,
        keys,
    )
    assert report["status"] == "pending_raters"
    assert disagreements == []

    rater_two = [
        {"pair_id": "pair-1", "preference": "B"},
        {"pair_id": "pair-2", "preference": "B"},
    ]
    report, disagreements = evaluate_two_human_reviews(rater_one, rater_two, keys)
    assert report["status"] == "pending_adjudication"
    assert report["unresolved_disagreements"] == 1
    assert disagreements[0]["pair_id"] == "pair-1"
    assert "expected_preference" not in disagreements[0]

    report, disagreements = evaluate_two_human_reviews(
        rater_one,
        rater_two,
        keys,
        adjudication_rows=[
            {"pair_id": "pair-1", "adjudicated_preference": "A"}
        ],
    )
    assert disagreements == []
    assert report["status"] == "passed"
    assert report["final_agreement"] == 1.0


def _audit_row(
    essay_id: str,
    *,
    stage: int = 1,
    accepted: bool,
) -> dict:
    clean_rubrics = {
        "task_1": 5.0,
        "content_1": 5.0,
        "content_2": 5.0,
        "content_3": 5.0,
        "organization_1": 5.0,
        "organization_2": 5.0,
        "expression_1": 5.0,
        "expression_2": 5.0,
    }
    corrupted_rubrics = {**clean_rubrics, "expression_2": 4.0}
    features = {"word_Cnt": 100.0}
    return {
        "essay_id": essay_id,
        "stage_k": stage,
        "accepted": accepted,
        "text_before": "문법이 올바른 문장이다.",
        "text": "문법이가 올바른 문장이다.",
        "measured_rubrics_before": clean_rubrics,
        "measured_rubrics": corrupted_rubrics,
        "measured_features_before": features,
        "measured_features": features,
        "reverse_action": "STYLE_REFINE",
        "target_rubric": "expression_2",
        "corruption_op": "INJECT_GRAMMAR_ERR",
        "target_drop": 1.0,
        "edits": [
            {
                "operation": "replace",
                "target_span": "문법이",
                "text": "문법이가",
            }
        ],
    }


def _feature_row(target_gain: float, non_target_drop: float) -> dict:
    return {
        "action_type": "STYLE_REFINE",
        "target_rubric": "expression_2",
        "target_gain": target_gain,
        "target_gap_reduction": target_gain,
        "non_target_drop": non_target_drop,
        "evidence_match": 0.9,
        "edit_ratio": 0.1,
        "goal_preservation": 0.9,
        "emb_sim": 0.9,
    }
