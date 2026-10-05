"""Authored fixtures plus Section 6.4 live Bareun acceptance (explicit opt-in)."""

import os

import pytest

from verak.src.analyzer import Analyzer
from verak.src.schemas import Profile, Sentence, Token
from verak.v3.common import load_config
from verak.v3.ko import KoreanStructure, annotate, render, render_with_budget
from verak.v3.ko.annotation import initial_conjunction, read_lexicons
from verak.v3.ko.patterns import predicate_features


def morphs(pairs):
    """Authored tag sequences, not human-verified corpus annotations."""
    result, offset = [], 0
    for pair in pairs.split():
        form, tag = pair.rsplit("/", 1)
        result.append(Token(form, tag, offset, offset + len(form)))
        offset += len(form) + 1
    return result


def fixture_structure(rows):
    sentences, offset, parts = [], 0, []
    annotator = Analyzer(None)
    for i, (paragraph, pairs) in enumerate(rows):
        tokens = morphs(pairs)
        text = " ".join(token.form for token in tokens)
        for token in tokens:
            token.start += offset
            token.end += offset
        sentence = Sentence(str(i), paragraph, offset, offset + len(text), text, tokens)
        annotator._annotate(sentence, sentences)
        sentences.append(sentence)
        parts.append(text)
        offset += len(text) + 1
    text = "\n".join(parts)
    return annotate(text, Profile(sentences, [], "bareun", "authored_fixture"))


@pytest.mark.parametrize("pairs,polarity,modality,uncertain", [
    ("느끼/VV 어/EC 지/VX ㄹ/ETM 수/NNB 밖에/JX 없/VA 다/EF", "POS", "NECESSITY", []),
    ("느끼/VV 어/EC 지/VX ㄹ/ETM 수/NNB 밖에/JX 있/VA 다/EF", "POS", None, ["modality_ill_formed"]),
    ("가/VV 지/EC 않/VX 을/ETM 수/NNB 없/VA 다/EF", "POS", "NECESSITY", []),
    ("가/VV ㄹ/ETM 수/NNB 있/VA 다/EF", "POS", "POSSIBLE", []),
    ("가/VV ㄹ/ETM 수/NNB 없/VA 다/EF", "NEG", "IMPOSSIBLE", []),
    ("가/VV ㄹ/ETM 수/NNB 는/JX 없/VA 다/EF", "NEG", "IMPOSSIBLE", []),
    ("가/VV ㄹ/ETM 수/NNB 도/JX 있/VA 다/EF", "POS", "POSSIBLE", []),
    ("가/VV ㄹ/ETM 수/NNB 가/JKS 있/VA 다/EF", "POS", "POSSIBLE", []),
    ("지키/VV 어야/EC 하/VX ㄴ다/EF", "POS", "OBLIGATION", []),
    ("가/VV 아야/EC 되/VV ㄴ다/EF", "POS", "OBLIGATION", []),
    ("안/MAG 가/VV ㄴ다/EF", "NEG", None, []),
    ("못/MAG 가/VV ㄴ다/EF", "NEG", None, []),
    ("가/VV 지/EC 못하/VX ㄴ다/EF", "NEG", None, []),
    ("가/VV 지/EC 않/VX 는다/EF", "NEG", None, []),
    ("그렇/VA 지/EC 가/JKS 않/VX 다/EF", "NEG", None, []),
    ("가/VV 지/EC 는/JX 않/VX 는다/EF", "NEG", None, []),
    ("방법/NNG 이/JKS 없/VA 다/EF", "NEG", None, []),
    ("학생/NNG 이/JKS 아니/VCN 다/EF", "NEG", None, []),
    ("비/NNG 가/JKS 안/MAG 오/VV 아서/EC 여행/NNG 을/JKO 가/VV ㄴ다/EF", "POS", None, []),
    ("없/VA 는/ETM 자료/NNG 를/JKO 찾/VV 았/EP 다/EF", "POS", None, []),
    ("가/VV ㄹ/ETM 수/NNB 없/VA 는/ETM 곳/NNG 을/JKO 찾/VV 았/EP 다/EF", "POS", None, []),
    ("글/NNG", "unknown", None, ["polarity", "modality"]),
])
def test_main_predicate_patterns(pairs, polarity, modality, uncertain):
    assert predicate_features(morphs(pairs)) == (polarity, modality, uncertain)


def test_antecedents_same_paragraph_dedup_ambiguity_and_topic_edges():
    structure = fixture_structure([
        (1, "학생/NNG 들/XSN 은/JX 공부하/VV ㄴ다/EF"),
        (1, "학생/NNG 은/JX 읽/VV 는다/EF"),
        (1, "복습/NNG 을/JKO 하/VV ㄴ다/EF"),
        (2, "복습/NNG 을/JKO 하/VV ㄴ다/EF"),
    ])
    assert structure.annotations[2].antecedent.status == "resolved"
    assert structure.annotations[2].antecedent.targets == ["S2"]
    assert {(e.kind, e.src, e.dst) for e in structure.edges} >= {
        ("REF", "S3", "S2"), ("TOPIC", "S3", "S2"), ("TOPIC", "S2", "S1")}
    assert structure.annotations[3].antecedent.status == "none"
    ambiguous = fixture_structure([(1, "민수/NNP 는/JX 읽/VV 는다/EF"),
                                   (1, "영희/NNP 는/JX 쓰/VV ㄴ다/EF"),
                                   (1, "웃/VV 는다/EF")])
    assert ambiguous.annotations[2].antecedent.status == "ambiguous"
    assert not any(e.kind == "REF" and e.src == "S3" for e in ambiguous.edges)


def test_narrator_fallback_and_realized_unmarked_quantifier():
    structure = fixture_structure([(1, "나/NP 는/JX 읽/VV 는다/EF"),
                                   (2, "웃/VV 는다/EF"),
                                   (2, "둘/NR 다/MAG 즐겁/VA 다/EF")])
    assert structure.annotations[1].antecedent.status == "resolved"
    assert "antecedent" in structure.annotations[1].uncertain
    assert structure.annotations[2].subject.realized
    assert not any(e.kind == "REF" and e.src == "S3" for e in structure.edges)


def test_style_relations_rendering_and_unchanged_source():
    structure = fixture_structure([(1, "학생/NNG 은/JX 읽/VV 는다/EF"),
                                   (1, "규칙/NNG 을/JKO 지키/VV 면/EC 좋/VA 다/EF"),
                                   (2, "그러나/MAJ 결과/NNG 는/JX 다르/VA ㅂ니다/EF")])
    assert structure.dominant_style == "한다"
    assert structure.annotations[-1].style == "합니다"
    assert "style_shift" in structure.annotations[-1].uncertain
    assert structure.annotations[1].connectives[0].candidates == ["CONDITION"]
    assert any(e.kind == "REL" and e.src == "S3" and e.dst == "S2" for e in structure.edges)
    before = structure.to_dict()
    full = render(structure, compact=False, max_line_chars=100)
    compact = render(structure, compact=True)
    assert "[P2]" in full and "?" in full and "극성:POS" not in full
    assert all(len(line) <= 100 for line in full.splitlines())
    assert structure.annotations[0].text not in compact
    assert render_with_budget(structure, len, 10)["mode"] == "compact"
    assert render_with_budget(structure, len, 10000)["mode"] == "compact"
    assert render_with_budget(structure, len, 10000)["eligible"]
    assert not render_with_budget(structure, len, 10)["eligible"]
    assert structure.to_dict() == before
    assert [a.sid for a in fixture_structure([(1, "읽/VV 는다/EF")]).annotations] == ["S1"]


def test_conjunction_phrase_boundaries_and_connective_ambiguity():
    lexicons = read_lexicons()
    assert initial_conjunction("예를  들어, 학생들은 공부한다.", lexicons["conjunction"]).relation == "EXAMPLE"
    assert initial_conjunction("그래서인지 좋았다.", lexicons["conjunction"]) is None
    structure = fixture_structure([(1, "오/VV 아서/EC 늦/VV 었/EP 다/EF")])
    assert structure.annotations[0].connectives[0].candidates == ["CAUSE", "SEQUENCE"]


def test_compact_is_uniform_preserves_edges_and_decision_flags_without_truncation():
    structure = fixture_structure([
        (1, "학생/NNG 은/JX 공부하/VV ㄴ다/EF"),
        (1, "지키/VV 면/EC 웃/VV 는다/EF"),
        (1, "그러나/MAJ 결과/NNG 가/JKS 좋/VA 지/EC 않/VX 습니다/EF"),
    ])
    text = render(structure)
    assert text.startswith("[주문체:한다]")
    assert text.count("한다") == 1
    assert "문체:합니다" in text and "화제:학생 은" in text
    assert "결과 가" not in text  # Realized JKS is omitted from the compact flags.
    assert "주어∅:resolved→S1" in text
    assert "CONDITION" in text and "CONTRAST" in text
    assert "극성:NEG" in text and "극성:POS" not in text
    # Unmapped auxiliary EC has no relation-bearing display; its uncertainty is kept.
    assert "지(" not in text and "?" in text
    for edge in structure.edges:
        assert f"{edge.kind}→{edge.dst}({edge.label})" in text
    structure.annotations[0].subject_candidates[0].surface = "학생" * 200 + "은"
    long_view = render_with_budget(structure, len, 100)
    assert "학생" * 200 + "은" in long_view["text"]
    assert "…" not in long_view["text"]
    assert long_view["over_budget"] and not long_view["eligible"]
    assert render_with_budget(structure, len, 100000)["mode"] == "compact"


def test_topic_normalization_preserves_attached_negative_prefix():
    text = "불평등은 심하다."
    sentence = Sentence("1.1", 1, 0, len(text), text, [
        Token("불", "XPN", 0, 1), Token("평등", "NNG", 1, 3), Token("은", "JX", 3, 4),
        Token("심하", "VA", 5, 7), Token("다", "EF", 7, 8), Token(".", "SF", 8, 9)])
    Analyzer(None)._annotate(sentence, [])
    structure = annotate(text, Profile([sentence], [], "bareun", "authored_fixture"))
    assert structure.annotations[0].topic == "불평등"
    assert structure.annotations[0].subject.surface == "불평등은"


@pytest.fixture(scope="module")
def live_structure():
    if os.getenv("VERAK_LIVE_TAGS") != "1":
        pytest.skip("Real Bareun integration: VERAK_LIVE_TAGS=1; keys from environment")
    return KoreanStructure.from_config(load_config())


@pytest.mark.parametrize("text,expected", [
    ("규칙을 지키면 사고가 줄어든다.", "CONDITION"),
    ("규칙을 지키니까 사고가 줄어든다.", "CAUSE"),
])
def test_live_section64_connectives(live_structure, text, expected):
    assert any(expected in conn.candidates for conn in live_structure.analyze(text).annotations[0].connectives)


@pytest.mark.parametrize("text,modality,uncertain", [
    ("생동감이 느껴질 수밖에 없다.", "NECESSITY", False),
    ("생동감이 느껴질 수밖에 있다.", None, True),
])
def test_live_section64_modality(live_structure, text, modality, uncertain):
    ann = live_structure.analyze(text).annotations[0]
    assert ann.modality == modality
    assert ("modality_ill_formed" in ann.uncertain) == uncertain


def test_live_section64_ellipsis(live_structure):
    structure = live_structure.analyze("학생들은 열심히 공부한다. 매일 복습을 한다.")
    assert structure.annotations[1].antecedent.status == "resolved"
    assert structure.annotations[1].antecedent.targets == ["S1"]
    assert {(edge.kind, edge.src, edge.dst) for edge in structure.edges} >= {
        ("REF", "S2", "S1"), ("TOPIC", "S2", "S1")}


def test_live_section64_realized_quantifier(live_structure):
    structure = live_structure.analyze("민수는 책을 샀다. 영희는 영화를 봤다. 둘 다 즐거웠다.")
    assert structure.annotations[2].subject.realized
    assert structure.annotations[2].subject.surface == "둘 다"
    assert not any(edge.kind == "REF" and edge.src == "S3" for edge in structure.edges)


def test_live_section64_conjunction(live_structure):
    assert live_structure.analyze("그러나 결과는 달랐다.").annotations[0].initial_conj.relation == "CONTRAST"


def test_live_section64_style(live_structure):
    structure = live_structure.analyze("학생들은 공부한다. 매일 복습한다. 이 방법은 효과적입니다.")
    assert structure.annotations[-1].style == "합니다"
    assert structure.dominant_style == "한다"


@pytest.mark.parametrize("text,polarity,modality", [
    ("사실을 알 수는 없다.", "NEG", "IMPOSSIBLE"),
    ("경제 발달이 중지될 수도 있다.", "POS", "POSSIBLE"),
    ("그렇지가 않다.", "NEG", None),
])
def test_live_focus_particles_in_predicate_chain(live_structure, text, polarity, modality):
    ann = live_structure.analyze(text).annotations[0]
    assert (ann.polarity, ann.modality) == (polarity, modality)


def test_live_negative_prefix_is_preserved(live_structure):
    ann = live_structure.analyze("사회적 불평등은 여전히 심하다.").annotations[0]
    assert "불평등" in ann.topic
    assert "불평등은" in ann.subject.surface
