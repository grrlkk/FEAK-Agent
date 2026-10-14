from feak_tc.rv.rebuild import (
    apply_replacements,
    build_state_groups,
    generate_replacement,
    select_regeneration_rows,
    validate_replacement,
    wrong_target_spec,
)
from feak_tc.rv.schema import CANDIDATE_TYPES, LABEL_FIELDS


def test_selective_targets_include_all_wrong_and_only_add_detail_over_edit():
    rows = _rows()

    selected = select_regeneration_rows(rows)

    assert {row["sample_id"] for row in selected} == {
        "state-add:wrong_target",
        "state-add:over_edit",
        "state-style:wrong_target",
    }


def test_wrong_target_generation_retries_target_repair_and_validates_edit():
    rows = _rows()
    groups = build_state_groups(rows)
    row = groups["state-style"]["wrong_target"]
    calls = []

    def fake_request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"revised_text": row["before_text"].replace("반복 반복", "반복")}
        return {
            "revised_text": row["before_text"].replace(
                "보조 문장이다.", "보조 근거를 구체적인 사례와 함께 설명한 문장이다."
            )
        }

    text, metrics = generate_replacement(
        row,
        groups["state-style"],
        {
            "model": "test",
            "max_attempts": 2,
            "validation": {"wrong_target_min_edit_ratio": 0.02},
        },
        request_fn=fake_request,
    )

    assert len(calls) == 2
    assert "반복 반복" in text
    assert metrics["edit_ratio_from_current"] >= 0.02
    assert "content_2" in calls[-1]["user"]


def test_wrong_target_late_retry_names_a_safe_sentence():
    rows = _rows()
    groups = build_state_groups(rows)
    row = groups["state-style"]["wrong_target"]
    prompts = []

    def fake_request(**kwargs):
        prompts.append(kwargs["user"])
        if len(prompts) < 3:
            return {"revised_text": row["before_text"]}
        return {
            "revised_text": row["before_text"].replace(
                "보조 문장이다.", "보조 근거를 구체적으로 설명하는 문장이다."
            )
        }

    generate_replacement(
        row,
        groups["state-style"],
        {"model": "test", "max_attempts": 3},
        request_fn=fake_request,
    )

    assert "required_sentence_to_revise" not in prompts[1]
    assert "required_sentence_to_revise" in prompts[2]


def test_strong_variant_requires_multiple_safe_sentences_and_larger_edit():
    rows = _rows()
    groups = build_state_groups(rows)
    row = groups["state-style"]["wrong_target"]
    prompt = __import__(
        "feak_tc.rv.rebuild", fromlist=["build_regeneration_prompt"]
    ).build_regeneration_prompt(
        row,
        groups["state-style"],
        generation_variant="strong",
    )

    assert "required_non_target_sentences_to_revise" in prompt
    try:
        validate_replacement(
            row,
            groups["state-style"],
            row["before_text"].replace("보조 문장이다.", "보조 설명 문장이다."),
            {},
            generation_variant="strong",
        )
    except ValueError as exc:
        assert "wrong_target edit ratio" in str(exc)
    else:
        raise AssertionError("strong variant accepted a tiny edit")


def test_add_detail_over_edit_requires_restored_sentences_and_reference_distance():
    rows = _rows()
    groups = build_state_groups(rows)
    row = groups["state-add"]["over_edit"]
    candidate = (
        "빠진 세부 근거입니다. 원래 주장이다. 전혀 근거 없는 내용을 크게 덧붙이고 "
        "기존의 보조 설명을 삭제해 버렸다."
    )

    metrics = validate_replacement(
        row,
        groups["state-add"],
        candidate,
        {
            "max_length_ratio": 10.0,
            "over_edit_min_reference_edit_ratio": 0.08,
        },
    )

    assert metrics["edit_ratio_from_reference"] >= 0.08


def test_apply_replacements_marks_every_row_pending_and_preserves_legacy_labels():
    rows = _rows()
    replacements = {
        "state-add:wrong_target": {
            "text": "새로운 wrong target 후보 문장이다. 원래 주장이다.",
            "metrics": {"edit_ratio_from_current": 0.3},
        }
    }

    result = apply_replacements(
        rows,
        replacements,
        dataset_version="rv-pilot-v2-test",
        model="test-model",
    )

    assert all(row["training_eligible"] is False for row in result)
    assert all(row["label_status"] == "pending_instance_relabel" for row in result)
    assert all(set(row["legacy_labels_v1"]) == set(LABEL_FIELDS) for row in result)
    replaced = next(row for row in result if row["sample_id"] == "state-add:wrong_target")
    assert replaced["provenance"]["candidate_method"] == "llm_selective_regeneration_v2"
    assert wrong_target_spec("content_2")["requested_wrong_target_action"] == "STYLE_REFINE"


def _rows():
    rows = []
    for state_id, operation, action, target, current, correct, spans in (
        (
            "state-add",
            "DELETE_SPECIFICS",
            "ADD_DETAIL",
            "content_2",
            "원래 주장이다.",
            "빠진 세부 근거입니다. 원래 주장이다.",
            [{"operation": "delete", "target_span": "빠진 세부 근거입니다.", "text": ""}],
        ),
        (
            "state-style",
            "INJECT_LEX_REPEAT",
            "STYLE_REFINE",
            "expression_1",
            "핵심 문장에 반복 반복 표현이 있다. 보조 문장이다.",
            "핵심 문장에 표현이 있다. 보조 문장이다.",
            [{"operation": "insert_after", "target_span": "핵심 문장에", "text": "반복 반복"}],
        ),
    ):
        for candidate_type in CANDIDATE_TYPES:
            after = {
                "correct_repair": correct,
                "partial_repair": correct + " 일부",
                "wrong_target": current + " 기존 wrong",
                "over_edit": correct + " 기존 over",
                "further_corruption": current + " 추가 훼손",
                "no_edit": current,
            }[candidate_type]
            rows.append(
                {
                    "dataset_version": "v1",
                    "sample_id": f"{state_id}:{candidate_type}",
                    "essay_id": state_id,
                    "state_id": state_id,
                    "stage_k": 1,
                    "question": "질문",
                    "before_text": current,
                    "after_text": after,
                    "target_rubric": target,
                    "intended_action": action,
                    "intent": "intent",
                    "corruption_type": operation,
                    "changed_spans": spans,
                    "candidate_type": candidate_type,
                    "candidate_source": "llm" if candidate_type in {"wrong_target", "over_edit"} else "trajectory_current",
                    **{field: "pass" for field in LABEL_FIELDS},
                    "weak_supervision": True,
                    "label_source": "v1",
                    "provenance": {},
                }
            )
    return rows
