from scripts.run_two_llm_blind_review import (
    _blind_quality_report,
    _find_disagreements,
    _request_decision,
    _review_pair_ids_for_transitions,
    _validate_decision,
    _validate_same_blind_forms,
)


def _decision(preference="B"):
    return {
        "preference": preference,
        "confidence": 80,
        "notes": "판단 근거다.",
        "local_fluency_a": 4,
        "local_fluency_b": 3,
        "canned_artifact_a": False,
        "canned_artifact_b": False,
    }


def _row(pair_id="pair-1"):
    return {
        "pair_id": pair_id,
        "essay_id": "essay-1",
        "question": "질문",
        "text_a": "첫 번째 글",
        "text_b": "두 번째 글",
        "preference": "",
    }


def test_request_decision_excludes_non_public_key_fields():
    captured = {}

    def requester(**kwargs):
        captured.update(kwargs)
        return _decision("B")

    row = {**_row(), "expected_preference": "SECRET_KEY_MARKER"}
    decision = _request_decision(
        row,
        model="gpt-4.1-2025-04-14",
        max_attempts=1,
        timeout=10,
        requester=requester,
    )

    assert decision["preference"] == "B"
    assert "SECRET_KEY_MARKER" not in captured["user"]
    assert "expected_preference" not in captured["user"]


def test_validate_decision_normalizes_and_rejects_bad_values():
    assert _validate_decision(
        {**_decision(" tie "), "confidence": 50, "notes": "비슷하다."}
    )["preference"] == "TIE"

    try:
        _validate_decision(
            {**_decision("C"), "confidence": 50, "notes": "잘못된 선택"}
        )
    except ValueError as exc:
        assert "invalid preference" in str(exc)
    else:
        raise AssertionError("invalid preference should fail")


def test_forms_match_and_disagreements_do_not_include_text_or_key():
    first = [{**_row(), "preference": "A"}]
    second = [{**_row(), "preference": "B"}]
    _validate_same_blind_forms(first, second)

    disagreements = _find_disagreements(first, second)
    assert disagreements == [
        {
            "pair_id": "pair-1",
            "rater_one_preference": "A",
            "rater_two_preference": "B",
            "adjudicated_preference": "",
            "notes": "",
        }
    ]
    assert "text_a" not in disagreements[0]
    assert "expected_preference" not in disagreements[0]


def test_blind_quality_report_uses_hidden_key_only_after_reviews():
    first = [{**_row(), **_decision("A")}]
    second = [{**_row(), **_decision("A"), "local_fluency_b": 2, "canned_artifact_b": True}]
    report = _blind_quality_report(
        first,
        second,
        [{"pair_id": "pair-1", "expected_preference": "A"}],
    )
    assert report["rater_one"]["local_fluency_ge3_rate"] == 1.0
    assert report["rater_two"]["canned_artifact_rate"] == 1.0


def test_review_scope_keeps_only_transitions_in_final_dataset():
    keys = [
        {"pair_id": "p1", "essay_id": "e1", "corrupted_stage": 2},
        {"pair_id": "p2", "essay_id": "e2", "corrupted_stage": 1},
    ]
    accepted = [{"essay_id": "e2", "stage_k": 1}]

    assert _review_pair_ids_for_transitions(keys, accepted) == {"p2"}
