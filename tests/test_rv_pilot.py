import json
from pathlib import Path

from feak_tc.rv.generation import generate_llm_candidate_texts
from feak_tc.rv.labels import build_weak_labels
from feak_tc.rv.pilot import (
    ResolvedTransition,
    audit_corruption_data,
    build_candidate_rows,
    build_pilot_report,
    resolve_training_rows,
    select_pilot_anchors,
)
from feak_tc.rv.schema import CANDIDATE_TYPES


def _labels(
    target_fulfillment: str,
    preservation: str,
    edit_appropriateness: str,
    action_consistency: str,
) -> dict[str, str]:
    return {
        "target_fulfillment": target_fulfillment,
        "preservation": preservation,
        "edit_appropriateness": edit_appropriateness,
        "action_consistency": action_consistency,
    }


LABEL_MAPPING = {
    "correct_repair": _labels("pass", "pass", "pass", "pass"),
    "partial_repair": _labels("partial", "pass", "pass", "pass"),
    "wrong_target": _labels("fail", "pass", "partial", "fail"),
    "over_edit": _labels("pass", "partial", "fail", "partial"),
    "further_corruption": _labels("fail", "fail", "fail", "fail"),
    "no_edit": _labels("fail", "pass", "fail", "fail"),
}


def test_resolve_audit_and_build_six_candidate_rows(tmp_path: Path):
    training_row, chain = _fixture("essay-1", "DELETE_SPECIFICS")
    raw_path = tmp_path / "chains.jsonl"
    raw_path.write_text(json.dumps(chain, ensure_ascii=False) + "\n", encoding="utf-8")

    resolved, resolution = resolve_training_rows([training_row], [raw_path])
    audit = audit_corruption_data([training_row], resolved, resolution)
    anchor = resolved[0]
    llm_texts = _llm_texts(anchor.current_text)
    rows = build_candidate_rows(
        anchor,
        llm_texts,
        LABEL_MAPPING,
        dataset_version="test-v1",
        label_source="test-weak-labels",
        llm_model="gpt-5-mini-test",
        llm_validation_cfg={"max_length_ratio": 3.0},
    )

    assert resolution["resolved_exact_rows"] == 1
    assert resolution["rows_with_next_state"] == 1
    assert audit["decision_summary"]["regeneration_needed"] == []
    assert audit["field_audit"]["x0"]["classification"] == "metadata_enrichment_needed"
    assert len(rows) == 6
    assert {row["candidate_type"] for row in rows} == set(CANDIDATE_TYPES)
    assert len({row["after_text"] for row in rows}) == 6
    assert all(row["weak_supervision"] is True for row in rows)
    by_type = {row["candidate_type"]: row for row in rows}
    assert by_type["correct_repair"]["after_text"] == anchor.previous_text
    assert by_type["partial_repair"]["candidate_source"] == "corruption_edit_replay"
    assert by_type["partial_repair"]["after_text"] not in {
        anchor.previous_text,
        anchor.current_text,
    }
    assert by_type["further_corruption"]["after_text"] == anchor.next_text
    assert by_type["no_edit"]["after_text"] == anchor.current_text
    assert by_type["wrong_target"]["action_consistency"] == "fail"
    assert len(by_type["correct_repair"]["changed_spans"]) == 2

    report = build_pilot_report(
        rows,
        requested_essays=1,
        audit_path="audit.json",
        schema_path="schema.json",
        output_path="pilot.jsonl",
    )
    assert report["passed"] is True
    assert report["candidate_types"] == {name: 1 for name in sorted(CANDIDATE_TYPES)}


def test_select_pilot_anchors_is_balanced_and_essay_unique():
    operators = [
        "DELETE_SPECIFICS",
        "INJECT_LEX_REPEAT",
        "INSERT_OFFTOPIC",
        "SHUFFLE_FLOW",
    ]
    anchors = []
    for index, operator in enumerate(operators):
        row, chain = _fixture(f"essay-{index}", operator)
        anchors.append(
            ResolvedTransition(
                row=row,
                chain=chain,
                source_path=f"source-{index}.jsonl",
                exact_match_count=1,
            )
        )

    selected = select_pilot_anchors(anchors, sample_size=4, seed=17)

    assert len(selected) == 4
    assert len({anchor.essay_id for anchor in selected}) == 4
    assert {anchor.row["corruption_op"] for anchor in selected} == set(operators)

    one = select_pilot_anchors(anchors, sample_size=1, seed=17)
    assert len(one) == 1


def test_partial_replay_supports_insert_and_move_edits():
    insert_anchor = _replay_anchor(
        "INSERT_OFFTOPIC",
        previous="A. B. C.",
        current="A. X. B. Y. C.",
        edits=[
            {"operation": "insert_after", "target_span": "A.", "text": "X."},
            {"operation": "insert_after", "target_span": "B.", "text": "Y."},
        ],
    )
    move_anchor = _replay_anchor(
        "SHUFFLE_FLOW",
        previous="A. B. C. D.",
        current="A. D. B. C.",
        edits=[
            {"operation": "move_after", "target_span": "B.", "text": "D."},
            {"operation": "move_after", "target_span": "C.", "text": "B."},
        ],
    )

    assert insert_anchor.partial_text == "A. X. B. C."
    assert move_anchor.partial_text == "A. C. D. B."


def test_llm_candidate_generation_retries_a_trajectory_copy():
    row, chain = _fixture("essay-retry", "DELETE_SPECIFICS")
    anchor = ResolvedTransition(row, chain, "source.jsonl", 1)
    calls = []

    def fake_request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "wrong_target": anchor.current_text,
                "over_edit": anchor.current_text + " 전체를 크게 다시 썼다.",
            }
        return _llm_texts(anchor.current_text)

    result = generate_llm_candidate_texts(
        anchor,
        {
            "model": "gpt-5-mini-test",
            "max_attempts": 2,
            "candidate_repair_attempts": 1,
            "validation": {"max_length_ratio": 3.0},
        },
        request_fn=fake_request,
    )

    assert len(calls) == 2
    assert set(result) == {"wrong_target", "over_edit"}
    assert calls[0]["json_schema"]["additionalProperties"] is False


def test_weak_label_builder_rejects_ground_truth_ambiguity():
    labels = build_weak_labels(
        "partial_repair",
        LABEL_MAPPING,
        label_source="candidate-type-test",
    )

    assert labels == {
        "target_fulfillment": "partial",
        "preservation": "pass",
        "edit_appropriateness": "pass",
        "action_consistency": "pass",
        "weak_supervision": True,
        "label_source": "candidate-type-test",
    }


def _fixture(essay_id: str, operator: str) -> tuple[dict, dict]:
    previous = "주장은 분명하다. 첫째 근거는 구체적이다. 둘째 근거도 충분하다."
    current = "주장은 분명하다. 근거가 있다."
    following = "주장은 분명하다. 근거가 있다. 흐름과 무관한 문장을 덧붙였다."
    step = {
        "corruption_op": operator,
        "operator": operator,
        "target_rubric": "content_2",
        "reverse_action": "ADD_DETAIL",
        "intent": "ADD_SUPPORTING_EXPLANATION",
        "edits": [
            {"operation": "delete", "target_span": "첫째 근거는 구체적이다.", "text": ""},
            {"operation": "delete", "target_span": "둘째 근거도 충분하다.", "text": ""},
        ],
    }
    next_step = {
        "corruption_op": "INSERT_OFFTOPIC",
        "operator": "INSERT_OFFTOPIC",
        "target_rubric": "organization_2",
        "reverse_action": "DELETE_OR_FOCUS",
        "intent": "REMOVE_REDUNDANCY",
        "edits": [
            {"operation": "insert_after", "target_span": "근거가 있다.", "text": "무관한 문장"}
        ],
    }
    row = {
        "essay_id": essay_id,
        "chain_id": f"{essay_id}:g1",
        "transition_id": f"{essay_id}:g1:stage1",
        "stage_k": 1,
        "question": "근거를 들어 설명하시오.",
        "corruption_op": operator,
        "target_rubric": "content_2",
        "reverse_action": "ADD_DETAIL",
        "intent": "ADD_SUPPORTING_EXPLANATION",
        "text_before": previous,
        "text": current,
        "edits": step["edits"],
    }
    chain = {
        "record_id": essay_id,
        "question": row["question"],
        "states": [previous, current, following],
        "steps": [step, next_step],
    }
    return row, chain


def _llm_texts(current: str) -> dict[str, str]:
    return {
        "wrong_target": current + " 문장 끝 표현만 정돈하였다.",
        "over_edit": current + " 주제 밖의 새 사례와 결론까지 대폭 다시 작성하였다.",
    }


def _replay_anchor(
    operator: str,
    *,
    previous: str,
    current: str,
    edits: list[dict],
) -> ResolvedTransition:
    row = {
        "essay_id": f"replay-{operator}",
        "chain_id": f"replay-{operator}:g1",
        "stage_k": 1,
        "corruption_op": operator,
        "target_rubric": "organization_1",
        "reverse_action": "RESTRUCTURE",
        "intent": "RESTORE_LOGICAL_ORDER",
        "text_before": previous,
        "text": current,
        "edits": edits,
    }
    chain = {
        "record_id": row["essay_id"],
        "question": "문장을 고치시오.",
        "states": [previous, current, current + " Z."],
        "steps": [row, {"corruption_op": "INSERT_OFFTOPIC"}],
    }
    return ResolvedTransition(row, chain, "source.jsonl", 1)
