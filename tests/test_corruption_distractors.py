import json

from feak_tc.corruption import generate_chain
from feak_tc.corruption.distractors import (
    _build_assignments,
    generate_retrieval_payload,
    is_standalone_distractor_sentence,
    split_sentence_units,
)
from feak_tc.corruption.operators import OPERATOR_SPECS


def test_sentence_splitter_preserves_decimal_expressions():
    assert split_sentence_units("10.1kg부터 탈 수 있다. 다음 문장이다.") == [
        "10.1kg부터 탈 수 있다.",
        "다음 문장이다.",
    ]


def test_standalone_distractor_rejects_donor_context_dependencies():
    rejected = [
        "그러면 전쟁을 하는 나라의 인구도 급격히 줄어든다.",
        "이로 인한 가장 큰 문제가 독거노인 문제이다.",
        "두번째로 소개드릴 문화유산은 청자 오리모양 연적입니다.",
        "바다 생물을 오랫동안 위협한다는 뜻이 된다.",
        "갈라파고스는 원래 등록됐었는데 지금은 해지된 상태이다.",
        "개혁의 결과로, 우선 군사적 개혁이 이루어졌다.",
    ]

    assert not any(is_standalone_distractor_sentence(text) for text in rejected)


def test_retrieval_assignment_is_global_unique_and_records_provenance(tmp_path):
    bank = tmp_path / "bank.jsonl"
    records = _records()
    bank.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    assignments = _build_assignments(str(bank), 2, 1, 15, 80, 0.12, True)
    assigned = [item.text for pair in assignments.values() for item in pair]
    assert len(assigned) == len(set(assigned)) == 8

    record = records[0]
    spec = {
        **OPERATOR_SPECS["INSERT_OFFTOPIC"],
        "edits_per_step": 2,
        "distractor_bank_path": str(bank),
        "distractor_min_chars": 15,
        "distractor_max_chars": 80,
        "distractor_max_lexical_jaccard": 0.12,
    }
    first = generate_retrieval_payload(
        record_id=record["record_id"],
        question=record["question"],
        text=record["text"],
        source_text=record["text"],
        spec=spec,
    )
    second = generate_retrieval_payload(
        record_id=record["record_id"],
        question=record["question"],
        text=record["text"],
        source_text=record["text"],
        spec=spec,
    )
    assert first == second
    assert len({edit["distractor_record_id"] for edit in first["edits"]}) == 2
    assert all(
        edit["distractor_record_id"] != record["record_id"]
        and edit["distractor_question"] != record["question"]
        for edit in first["edits"]
    )


def test_chain_retrieval_generator_does_not_call_llm(monkeypatch, tmp_path):
    bank = tmp_path / "bank.jsonl"
    records = _records()
    bank.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "feak_tc.corruption.chain.request_json",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    cfg = {
        "seed": 7,
        "depth": 1,
        "llm": {"max_attempts": 3},
        "normalization": {"enabled": False},
        "validity": {
            "min_ratio_vs_source": 0.6,
            "max_ratio_vs_source": 1.8,
            "insertion_min_chars": 10,
            "insertion_max_chars": 250,
            "offtopic_source_ngram_size": 8,
        },
        "operators": {
            "INSERT_OFFTOPIC": {
                **OPERATOR_SPECS["INSERT_OFFTOPIC"],
                "generation_modes": ["retrieval"],
                "edits_per_step": 2,
                "distractor_bank_path": str(bank),
                "distractor_min_chars": 15,
                "distractor_max_chars": 80,
                "distractor_max_lexical_jaccard": 0.12,
            }
        },
    }
    chain = generate_chain(records[0], cfg)
    assert chain["status"] == "ok"
    step = chain["steps"][0]
    assert step["generator"] == "retrieval"
    assert step["model"] is None
    assert all("distractor_record_id" in edit for edit in step["edits"])


def _records():
    return [
        {
            "record_id": "r1",
            "question": "인권을 설명하세요.",
            "text": (
                "인권은 모든 사람에게 보장되어야 하는 기본 권리이다. "
                "부당한 차별을 막는 제도는 사회 구성원을 보호한다. "
                "존엄성을 존중하는 태도는 공동체의 신뢰를 높인다. "
                "자유로운 의사 표현은 민주 사회의 토대가 된다."
            ),
        },
        {
            "record_id": "r2",
            "question": "화산을 설명하세요.",
            "text": (
                "화산 내부의 마그마는 지각의 틈을 따라 올라온다. "
                "분출한 용암은 식으면서 단단한 암석으로 변한다. "
                "화산재는 바람을 타고 넓은 지역까지 퍼질 수 있다. "
                "지질학자는 분화 징후를 관측해 위험을 예측한다."
            ),
        },
        {
            "record_id": "r3",
            "question": "별자리를 설명하세요.",
            "text": (
                "별자리는 밤하늘에서 별을 묶어 만든 모양이다. "
                "북극성은 예전 항해자들이 방향을 찾는 기준이 되었다. "
                "계절이 바뀌면 관찰할 수 있는 별자리도 달라진다. "
                "천문대의 망원경은 희미한 천체의 빛을 모은다."
            ),
        },
        {
            "record_id": "r4",
            "question": "전통 음악을 설명하세요.",
            "text": (
                "전통 음악은 고유한 장단과 선율을 지닌다. "
                "북과 장구는 연주에 힘찬 리듬을 더한다. "
                "판소리의 소리꾼은 긴 이야기를 노래로 전달한다. "
                "지역마다 이어 온 민요에는 생활의 정서가 담겨 있다."
            ),
        },
    ]
