from feak_tc.rv.relabel import (
    CANDIDATE_CODES,
    aggregate_relabels,
    build_relabel_packets,
    relabel_packet_digest,
    request_relabel,
    strict_candidate_quality,
    validate_relabel,
)
from feak_tc.rv.schema import CANDIDATE_TYPES, LABEL_FIELDS


def test_packets_hide_types_and_labels_and_randomize_six_codes():
    rows = _rows()

    public, key = build_relabel_packets(rows, seed=7)

    assert len(public) == len(key) == 1
    assert {item["candidate_code"] for item in public[0]["candidates"]} == set(
        CANDIDATE_CODES
    )
    assert {item["candidate_type"] for item in key[0]["candidates"]} == set(
        CANDIDATE_TYPES
    )
    packet_text = str(public[0])
    assert "candidate_type" not in packet_text
    assert "target_fulfillment" not in packet_text
    changed = dict(public[0], intent="changed")
    assert relabel_packet_digest(public[0]) != relabel_packet_digest(changed)


def test_request_validates_all_codes_once_and_sends_no_key():
    public, _ = build_relabel_packets(_rows(), seed=9)
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return _response()

    result = request_relabel(
        public[0],
        model="test",
        max_attempts=1,
        timeout=1,
        requester=fake_request,
    )

    assert len(result["judgments"]) == 6
    assert "candidate_type" not in captured["user"]
    invalid = _response()
    invalid["judgments"][-1]["candidate_code"] = "C1"
    try:
        validate_relabel(invalid)
    except ValueError as exc:
        assert "exactly once" in str(exc)
    else:
        raise AssertionError("duplicate candidate code was accepted")


def test_quality_gate_requires_both_external_type_judges():
    key = [
        {
            "review_id": "review",
            "candidate_a": {"sample_id": "wrong", "candidate_type": "wrong_target"},
            "candidate_b": {"sample_id": "over", "candidate_type": "over_edit"},
        }
    ]
    results = {
        "m1": [_type_review("wrong_target", "over_edit")],
        "m2": [_type_review("other", "over_edit")],
    }

    quality = strict_candidate_quality(key, results)

    assert quality["wrong"]["passed"] is False
    assert quality["over"]["passed"] is True


def test_aggregate_uses_majority_and_excludes_failed_generated_candidate():
    rows = _rows()
    _, key = build_relabel_packets(rows, seed=5)
    results = {
        "m1": [_model_result(key[0], "pass")],
        "m2": [_model_result(key[0], "pass")],
        "m3": [_model_result(key[0], "partial")],
    }
    generated_quality = {
        row["sample_id"]: {
            "passed": row["candidate_type"] != "wrong_target",
            "reason": "test",
        }
        for row in rows
        if row["candidate_type"] in {"wrong_target", "over_edit"}
    }

    all_rows, train_rows, report = aggregate_relabels(
        rows,
        key,
        results,
        generated_quality,
        dataset_version="v2-test",
    )

    assert len(all_rows) == 6
    assert len(train_rows) == 5
    assert all(row[field] == "pass" for row in all_rows for field in LABEL_FIELDS)
    wrong = next(row for row in all_rows if row["candidate_type"] == "wrong_target")
    assert wrong["training_eligible"] is False
    assert wrong["candidate_quality_gate"]["passed"] is False
    assert report["training_candidates"] == 5
    assert report["inter_rater_fleiss_kappa"]["target_fulfillment"] == -0.5


def _rows():
    rows = []
    for candidate_type in CANDIDATE_TYPES:
        rows.append(
            {
                "dataset_version": "v2",
                "sample_id": f"state:{candidate_type}",
                "essay_id": "essay",
                "state_id": "state",
                "stage_k": 1,
                "question": "질문",
                "before_text": "현재 글",
                "after_text": f"{candidate_type} 수정 글",
                "target_rubric": "content_1",
                "intended_action": "ADD_DETAIL",
                "intent": "근거 추가",
                "corruption_type": "DELETE_SPECIFICS",
                "changed_spans": [{"operation": "delete", "target_span": "근거"}],
                "candidate_type": candidate_type,
                "candidate_source": "llm",
                **{field: "fail" for field in LABEL_FIELDS},
                "weak_supervision": True,
                "label_source": "legacy",
                "provenance": {},
            }
        )
    return rows


def _response(label="pass"):
    return {
        "judgments": [
            {
                "candidate_code": code,
                "observed_candidate_type": "other",
                **{field: label for field in LABEL_FIELDS},
                "confidence": 80,
                "notes": "관찰 근거",
            }
            for code in CANDIDATE_CODES
        ]
    }


def _model_result(key, label):
    response = _response(label)
    response["review_id"] = key["review_id"]
    return response


def _type_review(candidate_a, candidate_b):
    return {
        "review_id": "review",
        "candidate_a": {"inferred_candidate_type": candidate_a},
        "candidate_b": {"inferred_candidate_type": candidate_b},
    }
