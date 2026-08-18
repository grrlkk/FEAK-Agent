from feak_tc.corruption.quality import (
    audit_edit_artifacts,
    audit_operator_balance,
    audit_semantic_edit_artifacts,
    canonical_edit_signature,
)
from scripts.filter_corruption_g1 import _calibrate_operator_thresholds


def test_artifact_audit_detects_variable_repetition_template():
    rows = [
        _insert_row("essay-1", "사과, 사과, 사과를 계속 중요하게 생각해야 한다."),
        _insert_row("essay-2", "학교, 학교, 학교를 계속 중요하게 생각해야 한다."),
        _insert_row("essay-3", "권리, 권리, 권리를 계속 중요하게 생각해야 한다."),
    ]
    report = audit_edit_artifacts(
        rows,
        {
            "min_distinct_essays": 3,
            "max_distinct_essay_fraction": 0.02,
            "word_ngram_sizes": [5, 6, 7],
            "char_ngram_size": 20,
        },
    )
    assert report["passed"] is False
    assert any(
        item["kind"] == "exact_template" and item["distinct_essays"] == 3
        for item in report["violations"]
    )
    assert set(report["violations"][0]["affected_pair_ids"]) == {
        "essay-1:stage1",
        "essay-2:stage1",
        "essay-3:stage1",
    }
    assert canonical_edit_signature(rows[0]["edits"][0]["text"]).startswith("<REP>")


def test_artifact_quarantine_rejects_entire_recurring_cluster():
    from scripts.filter_corruption_g1 import _quarantine_artifacts

    rows = [
        _accepted_insert_row("essay-1", "어제 본 영화의 결말이 아직도 생각난다."),
        _accepted_insert_row("essay-2", "어제 본 영화의 결말이 아직도 기억난다."),
        _accepted_insert_row("essay-3", "어제 본 영화의 결말이 아직도 선명하다."),
        _accepted_insert_row("essay-4", "서로 겹치지 않는 고유한 문장이다."),
    ]

    accepted, report, quarantine = _quarantine_artifacts(
        rows,
        {
            "min_distinct_essays": 3,
            "max_distinct_essay_fraction": 0.02,
            "word_ngram_sizes": [5],
            "char_ngram_size": 100,
        },
    )

    assert report["passed"] is True
    assert [row["essay_id"] for row in accepted] == ["essay-4"]
    assert quarantine["rejected_steps"] == 3
    assert all(row["acceptance_reason"] == "corpus_artifact" for row in rows[:3])


def test_artifact_audit_ignores_unique_generated_edits():
    rows = [
        _insert_row("essay-1", "정원에서 발견한 작은 돌의 무늬가 눈에 띄었다."),
        _insert_row("essay-2", "버스 정류장 근처의 낡은 간판이 갑자기 떠올랐다."),
        _insert_row("essay-3", "서랍 안에 넣어 둔 파란색 단추를 다시 찾았다."),
    ]
    report = audit_edit_artifacts(
        rows,
        {
            "min_distinct_essays": 3,
            "max_distinct_essay_fraction": 0.02,
            "word_ngram_sizes": [5, 6, 7],
            "char_ngram_size": 20,
        },
    )
    assert report["passed"] is True


def test_semantic_artifact_audit_detects_paraphrase_cluster():
    rows = [
        _retrieval_row("essay-1", "영화의 마지막 장면이 오래 기억에 남았다.", "d1"),
        _retrieval_row("essay-2", "그 영화의 결말이 계속 머릿속에 떠올랐다.", "d2"),
        _retrieval_row("essay-3", "영화 결말의 인상이 아직도 선명하게 남아 있다.", "d3"),
        _retrieval_row("essay-4", "화산재는 바람을 따라 넓게 퍼질 수 있다.", "d4"),
    ]
    vectors = {
        rows[0]["edits"][0]["text"]: [1.0, 0.0],
        rows[1]["edits"][0]["text"]: [0.99, 0.01],
        rows[2]["edits"][0]["text"]: [0.98, 0.02],
        rows[3]["edits"][0]["text"]: [0.0, 1.0],
    }
    report = audit_semantic_edit_artifacts(
        rows,
        {
            "enabled": True,
            "similarity_threshold": 0.95,
            "min_distinct_essays": 3,
            "max_first_token_fraction": 1.0,
            "max_cue_fraction": 1.0,
        },
        encode_fn=lambda texts: [vectors[text] for text in texts],
    )
    assert report["passed"] is False
    assert report["violation_count"] == 1
    assert report["violations"][0]["distinct_essays"] == 3


def test_semantic_artifact_audit_recomputes_relevance_with_injected_encoder():
    row = _retrieval_row("essay-1", "화산은 뜨거운 용암을 분출한다.", "d1")
    row["question"] = "화산은 어떤 현상인가요?"
    row["text_before"] = "화산 활동을 설명하는 글이다."
    vectors = {
        row["edits"][0]["text"]: [1.0, 0.0],
        row["question"]: [1.0, 0.0],
        row["text_before"]: [0.0, 1.0],
    }
    report = audit_semantic_edit_artifacts(
        [row],
        {
            "enabled": True,
            "relevance_enabled": True,
            "max_question_similarity": 0.5,
            "max_source_similarity": 0.5,
        },
        encode_fn=lambda texts: [vectors[text] for text in texts],
    )
    assert report["passed"] is False
    assert report["relevance_similarity_source"] == "audit_recomputed"
    assert report["relevance_violation_count"] == 1


def test_operator_balance_reports_dominant_class():
    rows = [
        {"corruption_op": "INJECT_LEX_REPEAT"},
        {"corruption_op": "INJECT_LEX_REPEAT"},
        {"corruption_op": "INJECT_LEX_REPEAT"},
        {"corruption_op": "DELETE_SPECIFICS"},
    ]
    report = audit_operator_balance(rows, {"max_operator_fraction": 0.4})
    assert report["passed"] is False
    assert report["dominant_operator"] == "INJECT_LEX_REPEAT"
    assert report["dominant_fraction"] == 0.75


def test_operator_threshold_calibration_keeps_largest_balanced_pool():
    rows = [
        _drop_row("INJECT_LEX_REPEAT", 0.7),
        _drop_row("INJECT_LEX_REPEAT", 0.6),
        _drop_row("INJECT_LEX_REPEAT", 0.4),
        _drop_row("DELETE_SPECIFICS", 0.4),
        _drop_row("INSERT_OFFTOPIC", 0.4),
        _drop_row("SHUFFLE_FLOW", 0.4),
    ]
    report = _calibrate_operator_thresholds(
        rows,
        thresholds=(0.225, 0.3, 0.4, 0.5),
        max_operator_fraction=0.4,
    )
    recommended = report["recommended"]
    assert recommended["steps"] == 5
    assert recommended["by_operator"]["INJECT_LEX_REPEAT"] == 2
    assert recommended["dominant_fraction"] == 0.4


def _insert_row(essay_id: str, text: str) -> dict:
    return {
        "essay_id": essay_id,
        "stage_k": 1,
        "corruption_op": "INJECT_LEX_REPEAT",
        "edits": [
            {
                "operation": "insert_after",
                "target_span": "기존 문장이다.",
                "text": text,
            }
        ],
    }


def _drop_row(operator: str, target_drop: float) -> dict:
    return {
        "corruption_op": operator,
        "target_drop": target_drop,
        "quality_checks": {},
    }


def _accepted_insert_row(essay_id: str, text: str) -> dict:
    row = _insert_row(essay_id, text)
    row.update(
        {
            "accepted": True,
            "acceptance_reason": "accepted",
            "quality_checks": {},
        }
    )
    return row


def _retrieval_row(essay_id: str, text: str, donor_id: str) -> dict:
    return {
        "essay_id": essay_id,
        "question": f"{essay_id}의 질문",
        "stage_k": 1,
        "corruption_op": "INSERT_OFFTOPIC",
        "edits": [
            {
                "operation": "insert_after",
                "text": text,
                "distractor_record_id": donor_id,
                "distractor_question": f"{donor_id}의 다른 질문",
            }
        ],
    }
