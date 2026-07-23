import pytest

from feak_tc.corruption import OPERATOR_SPECS, generate_chain
from feak_tc.corruption.operators import parse_and_apply

VALIDITY = {
    "min_ratio_vs_source": 0.6,
    "max_ratio_vs_source": 1.8,
    "insertion_min_chars": 10,
    "insertion_max_chars": 250,
}
INTENSITY = {
    "drop_detail_spans": 2,
    "insert_offtopic_count": 2,
    "verbose_spans": 1,
    "shuffle_moves": 1,
    "flatten_spans": 2,
}

ESSAY = (
    "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다. "
    "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다. "
    "또한 헌법 제10조는 인간의 존엄과 가치를 명시하고 있다. "
    "인권은 국가가 함부로 제한할 수 없다. "
    "장애인 이동권 시위는 인권 보장의 실제 사례로 자주 언급된다. "
    "우리는 서로의 인권을 존중하며 살아야 한다."
)


def test_drop_detail_deletes_multiple_exact_spans():
    spans = [
        "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다.",
        "장애인 이동권 시위는 인권 보장의 실제 사례로 자주 언급된다.",
    ]
    new_text, edits = parse_and_apply(
        "DROP_DETAIL", {"target_spans": spans}, ESSAY, ESSAY, VALIDITY, INTENSITY
    )
    assert all(span not in new_text for span in spans)
    assert [e["operation"] for e in edits] == ["delete", "delete"]

    with pytest.raises(ValueError):
        parse_and_apply("DROP_DETAIL", {"target_spans": spans[:1]}, ESSAY, ESSAY, VALIDITY, INTENSITY)


def test_insert_offtopic_applies_two_novel_sentences():
    payload = {
        "edits": [
            {"anchor_span": "인권은 국가가 함부로 제한할 수 없다.",
             "insertion": "어제는 날씨가 좋아서 공원에 다녀왔다."},
            {"anchor_span": "우리는 서로의 인권을 존중하며 살아야 한다.",
             "insertion": "요즘 유행하는 드라마는 정말 재미있다."},
        ]
    }
    new_text, edits = parse_and_apply("INSERT_OFFTOPIC", payload, ESSAY, ESSAY, VALIDITY, INTENSITY)
    assert "공원에 다녀왔다" in new_text and "드라마는 정말 재미있다" in new_text
    assert len(edits) == 2


def test_verbose_repeat_must_lengthen():
    payload = {"edits": [{"target_span": "인권은 국가가 함부로 제한할 수 없다.", "replacement": "인권은 소중하다."}]}
    with pytest.raises(ValueError):
        parse_and_apply("VERBOSE_REPEAT", payload, ESSAY, ESSAY, VALIDITY, INTENSITY)


def test_shuffle_flow_moves_sentence():
    payload = {
        "moves": [
            {"moved_span": "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다.",
             "anchor_span": "인권은 국가가 함부로 제한할 수 없다."}
        ]
    }
    new_text, edits = parse_and_apply("SHUFFLE_FLOW", payload, ESSAY, ESSAY, VALIDITY, INTENSITY)
    assert new_text.index("함부로 제한할 수 없다") < new_text.index("태어나면서부터 가지는")
    assert edits[0]["operation"] == "move_after"


def test_rejects_span_not_in_text():
    with pytest.raises(ValueError):
        parse_and_apply(
            "DROP_DETAIL",
            {"target_spans": ["글에 없는 문장이다.", "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다."]},
            ESSAY, ESSAY, VALIDITY, INTENSITY,
        )


def test_generate_chain_records_reverse_action_labels(monkeypatch):
    payloads = iter(
        [
            {"target_spans": [
                "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다.",
                "장애인 이동권 시위는 인권 보장의 실제 사례로 자주 언급된다.",
            ]},
            {"edits": [
                {"anchor_span": "인권은 국가가 함부로 제한할 수 없다.",
                 "insertion": "요즘 유행하는 드라마는 정말 재미있다."},
                {"anchor_span": "우리는 서로의 인권을 존중하며 살아야 한다.",
                 "insertion": "어제 저녁에는 맛있는 피자를 먹었다."},
            ]},
            {"moves": [
                {"moved_span": "우리는 서로의 인권을 존중하며 살아야 한다.",
                 "anchor_span": "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다."}
            ]},
        ]
    )

    def fake_request_json(**kwargs):
        return next(payloads)

    monkeypatch.setattr("feak_tc.corruption.chain.request_json", fake_request_json)

    import random

    cfg = {"seed": 1, "depth": 3, "llm": {}, "validity": VALIDITY, "intensity": INTENSITY}
    rng = random.Random(0)
    monkeypatch.setattr(rng, "sample", lambda seq, k: ["DROP_DETAIL", "INSERT_OFFTOPIC", "SHUFFLE_FLOW"])
    chain = generate_chain({"record_id": "r1", "question": "인권이란?", "text": ESSAY}, cfg, rng=rng)

    assert chain["status"] == "ok"
    assert len(chain["states"]) == 4
    assert [s["reverse_action"] for s in chain["steps"]] == ["ADD_DETAIL", "DELETE_OR_FOCUS", "RESTRUCTURE"]
    assert all(s["edits"] for s in chain["steps"])
    assert {spec["reverse_action"] for spec in OPERATOR_SPECS.values()} == {
        "ADD_DETAIL", "DELETE_OR_FOCUS", "COMPRESS", "RESTRUCTURE", "STYLE_REFINE"
    }
