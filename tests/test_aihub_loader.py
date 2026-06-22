import json

from feak_tc.data.aihub_loader import (
    compute_rubric_score_means,
    load_aihub_records,
    normalize_aihub_record,
)


def test_normalize_aihub_record_handles_nested_fields():
    raw = {
        "id": "essay-001",
        "metadata": {
            "prompt": "Explain your position.",
            "topic": "copyright",
            "grade": 6,
        },
        "essay": {"text": "첫 문장입니다.\n둘째 문장입니다."},
        "features": {"morph_count": 12, "sentence_count": 2},
        "scores": {
            "rater_1": {"grammar": 2, "vocabulary": 3},
            "rater_2": {"grammar": 4, "vocabulary": 1},
            "content": {"score": 3},
        },
        "feedback": [{"rubric": "grammar", "comment": "문법 오류를 줄이세요."}],
        "rubric_definitions": {"grammar": "문법 정확성"},
    }

    record = normalize_aihub_record(raw, "sample.json")

    assert record.essay_id == "essay-001"
    assert record.prompt == "Explain your position."
    assert record.topic == "copyright"
    assert record.grade == 6
    assert record.text == "첫 문장입니다.\n둘째 문장입니다."
    assert record.features["morph_count"] == 12
    assert record.rubric_scores_mean["grammar"] == 3.0
    assert record.rubric_scores_mean["vocabulary"] == 2.0
    assert record.rubric_scores_mean["content"] == 3.0
    assert record.expert_feedback[0]["rubric"] == "grammar"


def test_normalize_aihub_record_handles_missing_optional_fields():
    raw = {"content": "본문만 있는 글입니다."}

    record = normalize_aihub_record(raw, "fallback.json")

    assert record.essay_id == "fallback"
    assert record.text == "본문만 있는 글입니다."
    assert record.prompt is None
    assert record.topic is None
    assert record.features == {}
    assert record.rubric_scores_raw == {}
    assert record.rubric_scores_mean == {}
    assert record.expert_feedback == []
    assert record.rubric_definitions == {}


def test_load_aihub_records_supports_top_level_list_and_limit(tmp_path):
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps(
            [
                {"essay_id": "a", "text": "첫 번째 글", "evaluations": [{"criterion": "grammar", "score": 1}]},
                {"essay_id": "b", "text": "두 번째 글", "evaluations": [{"criterion": "grammar", "score": 3}]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = load_aihub_records(tmp_path, limit=1)

    assert len(records) == 1
    assert records[0].essay_id == "a"
    assert records[0].raw_path.endswith("records.json#0")
    assert records[0].rubric_scores_mean["grammar"] == 1.0


def test_compute_rubric_score_means_handles_criterion_score_pairs():
    scores = {
        "items": [
            {"criterion": "organization", "score": 1},
            {"criterion": "organization", "score": 3},
            {"criterion": "expression", "score": "2"},
        ]
    }

    means = compute_rubric_score_means(scores)

    assert means == {"organization": 2.0, "expression": 2.0}
