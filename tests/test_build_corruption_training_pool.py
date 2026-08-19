from scripts.build_corruption_training_pool import build_training_pool


def test_training_pool_preserves_base_and_filters_cross_pool_duplicates():
    base = [
        _row("base1", 1, "DELETE_SPECIFICS", 0.3),
        _row(
            "base2",
            1,
            "INSERT_OFFTOPIC",
            0.4,
            edit_text="기존 풀에 예약된 고유 삽입 문장이다.",
        ),
    ]
    additional = [
        _row("base1", 1, "SHUFFLE_FLOW", 9.0),
        _row(
            "dup-edit",
            1,
            "INSERT_OFFTOPIC",
            8.0,
            edit_text="기존 풀에 예약된 고유 삽입 문장이다.",
        ),
        _row(
            "context",
            1,
            "SHUFFLE_FLOW",
            7.0,
            before="앞 문장. 기존 풀에 예약된 고유 삽입 문장이다.",
        ),
        _row("new1", 1, "SHUFFLE_FLOW", 0.9),
        _row("new2", 2, "INJECT_LEX_REPEAT", 0.8, edit_text="표현 표현 표현이 필요하다."),
    ]
    cfg = {
        "artifact_audit": {"enabled": False},
        "semantic_artifact_audit": {"enabled": False},
        "balance": {"enabled": True, "max_operator_fraction": 0.75},
    }

    rows, report = build_training_pool(base, additional, cfg, target_size=4)

    assert report["passed"] is True
    assert report["base_retained"] == 2
    assert report["additional_selected"] == 2
    assert report["skipped"] == {
        "base_edit_in_additional_context": 1,
        "duplicate_generated_edit": 1,
        "duplicate_natural_key": 1,
    }
    assert len({row["transition_id"] for row in rows}) == 4
    assert {row["transition_id"] for row in rows} == {
        "base1:g1:stage1",
        "base2:g1:stage1",
        "new1:g1:stage1",
        "new2:g1:stage2",
    }


def _row(
    essay_id: str,
    stage: int,
    operator: str,
    drop: float,
    *,
    edit_text: str | None = None,
    before: str = "이전 상태이다.",
) -> dict:
    edits = []
    if edit_text:
        edits.append({"operation": "insert_after", "text": edit_text})
    return {
        "essay_id": essay_id,
        "chain_id": f"{essay_id}:g1",
        "stage_k": stage,
        "corruption_op": operator,
        "target_drop": drop,
        "accepted": True,
        "text_before": before,
        "text": before + f" 변경 {essay_id} {stage}",
        "edits": edits,
    }
