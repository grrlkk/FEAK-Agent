from feak_tc.corruption.g2 import (
    TRANSITION_FEATURES,
    build_gbm_pairs,
    build_human_review_pairs,
    evaluate_human_review,
    run_grouped_gbm,
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
