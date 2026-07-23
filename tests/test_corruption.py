import pytest

from feak_tc.corruption import OPERATOR_SPECS, generate_chain
from feak_tc.corruption.operators import parse_and_apply

VALIDITY = {
    "min_ratio_vs_source": 0.6,
    "max_ratio_vs_source": 1.8,
    "insertion_min_chars": 10,
    "insertion_max_chars": 250,
}

ESSAY = (
    "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다. "
    "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다. "
    "인권은 국가가 함부로 제한할 수 없다. "
    "우리는 서로의 인권을 존중하며 살아야 한다."
)


def test_drop_detail_deletes_exact_span():
    span = "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다."
    new_text, edits = parse_and_apply("DROP_DETAIL", {"target_span": span}, ESSAY, ESSAY, VALIDITY)

    assert span not in new_text
    assert edits[0]["operation"] == "delete"
    assert edits[0]["target_span"] == span


def test_insert_offtopic_requires_novel_complete_sentence():
    anchor = "인권은 국가가 함부로 제한할 수 없다."
    insertion = "어제는 날씨가 좋아서 공원에 다녀왔다."
    new_text, edits = parse_and_apply(
        "INSERT_OFFTOPIC", {"anchor_span": anchor, "insertion": insertion}, ESSAY, ESSAY, VALIDITY
    )
    assert insertion in new_text
    assert new_text.index(insertion) > new_text.index(anchor)

    with pytest.raises(ValueError):
        parse_and_apply(
            "INSERT_OFFTOPIC",
            {"anchor_span": anchor, "insertion": "짧은 연결어라서"},
            ESSAY, ESSAY, VALIDITY,
        )


def test_verbose_repeat_must_lengthen():
    span = "인권은 국가가 함부로 제한할 수 없다."
    with pytest.raises(ValueError):
        parse_and_apply("VERBOSE_REPEAT", {"target_span": span, "replacement": "인권은 소중하다."}, ESSAY, ESSAY, VALIDITY)


def test_shuffle_flow_moves_sentence():
    moved = "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다."
    anchor = "인권은 국가가 함부로 제한할 수 없다."
    new_text, edits = parse_and_apply(
        "SHUFFLE_FLOW", {"moved_span": moved, "anchor_span": anchor}, ESSAY, ESSAY, VALIDITY
    )
    assert new_text.index(anchor) < new_text.index(moved)
    assert edits[0]["operation"] == "move_after"


def test_rejects_span_not_in_text():
    with pytest.raises(ValueError):
        parse_and_apply("DROP_DETAIL", {"target_span": "글에 없는 문장이다."}, ESSAY, ESSAY, VALIDITY)


def test_generate_chain_records_reverse_action_labels(monkeypatch):
    payloads = iter(
        [
            {"target_span": "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다."},
            {"anchor_span": "인권은 국가가 함부로 제한할 수 없다.",
             "insertion": "요즘 유행하는 드라마는 정말 재미있다."},
            {"moved_span": "우리는 서로의 인권을 존중하며 살아야 한다.",
             "anchor_span": "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다."},
        ]
    )

    def fake_request_json(**kwargs):
        return next(payloads)

    monkeypatch.setattr("feak_tc.corruption.chain.request_json", fake_request_json)

    import random

    cfg = {"seed": 1, "depth": 3, "llm": {}, "validity": VALIDITY}
    rng = random.Random(0)
    # Force a deterministic operator order matching the fake payloads.
    monkeypatch.setattr(rng, "sample", lambda seq, k: ["DROP_DETAIL", "INSERT_OFFTOPIC", "SHUFFLE_FLOW"])
    chain = generate_chain({"record_id": "r1", "question": "인권이란?", "text": ESSAY}, cfg, rng=rng)

    assert chain["status"] == "ok"
    assert len(chain["states"]) == 4
    assert [s["reverse_action"] for s in chain["steps"]] == ["ADD_DETAIL", "DELETE_OR_FOCUS", "RESTRUCTURE"]
    assert all(s["edits"] for s in chain["steps"])
    assert {spec["reverse_action"] for spec in OPERATOR_SPECS.values()} == {
        "ADD_DETAIL", "DELETE_OR_FOCUS", "COMPRESS", "RESTRUCTURE", "STYLE_REFINE"
    }
