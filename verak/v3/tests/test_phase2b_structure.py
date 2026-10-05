"""Phase 2b fixes 1–8. Authored fixtures; no corpus gold labels fabricated."""

from dataclasses import fields

import pytest

from verak.src.analyzer import Analyzer
from verak.src.schemas import Profile, Sentence, Token
from verak.v3.ko import annotate, render
from verak.v3.ko.annotation import initial_conjunction, read_lexicons
from verak.v3.ko.levels import (ANNOTATION_LEVELS, COHESION_CHANGE_LEVELS, cohesion_change,
                               corruption_target_eligible, dependent_eligible, relation_eligible)
from verak.v3.ko.types import Annotation
from verak.v3.tests.test_ko_annotation import fixture_structure


@pytest.mark.parametrize("aux", "있 주 보 버리 가 오 싶 않 못하 말".split())
def test_auxiliary_ec_has_no_relation(aux):
    ann = fixture_structure([(1, f"하/VV 고/EC {aux}/VX 다/EF")]).annotations[0]
    assert ann.connectives[0].classification == "AUX"
    assert ann.connectives[0].candidates == []
    assert not relation_eligible(ann.connectives[0])


def test_auxiliary_through_focus_particle_and_lexical_predicate_not_aux():
    a = fixture_structure([(1, "가/VV 지/EC 는/JX 않/VX 는다/EF")]).annotations[0]
    assert a.connectives[0].classification == "AUX" and a.polarity == "NEG"
    b = fixture_structure([(1, "읽/VV 고/EC 가/VV ㄴ다/EF")]).annotations[0]
    assert b.connectives[0].classification == "AMBIGUOUS" and b.connectives[0].candidates


@pytest.mark.parametrize("form,expected", [(f, r) for fs, r in [
    ("면 으면", "CONDITION"), ("니까 으니까 므로 으므로", "CAUSE"),
    ("지만", "CONTRAST"), ("더라도 어도 아도 여도", "CONCESSION"),
    ("려고 으려고", "PURPOSE")] for f in fs.split()])
def test_unambiguous_relation_contract(form, expected):
    conn = fixture_structure([(1, f"가/VV {form}/EC 좋/VA 다/EF")]).annotations[0].connectives[0]
    assert conn.candidates == [expected] and relation_eligible(conn)


@pytest.mark.parametrize("form", "고 아서 어서 는데 며 으며".split())
def test_ambiguous_not_eligible(form):
    conn = fixture_structure([(1, f"가/VV {form}/EC 좋/VA 다/EF")]).annotations[0].connectives[0]
    assert conn.classification == "AMBIGUOUS" and not relation_eligible(conn)


@pytest.mark.parametrize("phrase", ["이처럼", "이와 같이", "이와같이", "이와  같이", "이렇게", "또", "또한",
    "게다가", "더구나", "결국", "그래도", "반대로", "한편", "그렇다면", "그러면", "그러므로",
    "이 때문에", "이때문에", "이로 인해", "이로인해", "왜냐하면"])
def test_initial_conjunction_variants(phrase):
    assert initial_conjunction(phrase + " 결과가 달라졌다.", read_lexicons()["conjunction"])
    assert initial_conjunction("또한번 보자.", read_lexicons()["conjunction"]) is None


@pytest.mark.parametrize("ending,style", [("ㄴ가", "한다"), ("을까", "한다"), ("니", "해"),
    ("나", "해"), ("지", "해"), ("거든", "해"), ("잖아", "해"), ("습니까", "합니다"),
    ("나요", "해요"), ("까요", "해요"), ("unmapped", "unknown")])
def test_interrogative_colloquial_styles(ending, style):
    assert fixture_structure([(1, f"가/VV {ending}/EF")]).annotations[0].style == style


def test_polite_particle_and_mixed_multi_unit():
    s = fixture_structure([(1, "가/VV ㄹ까/EF 요/JX ./SF 좋/VA 다/EF ./SF")])
    ann = s.annotations[0]
    assert ann.multi_unit and ann.style == "mixed"
    assert [v["style"] for v in ann.final_endings] == ["해요", "한다"]
    assert not corruption_target_eligible(ann) and "복수종결?" in render(s)


@pytest.mark.parametrize("noun", "것 점 수 때 데 바 등 측 중 뿐 사실 요즘 현재 오늘날 물론 지금 오늘 결국".split())
def test_excluded_mentions(noun):
    ann = fixture_structure([(1, f"{noun}/NNG 은/JX 좋/VA 다/EF")]).annotations[0]
    assert not ann.subject_candidates


def test_relative_clause_subject_excluded_without_dropping_outer_topic():
    s = fixture_structure([(1, "학생/NNG 은/JX 내/NP 가/JKS 읽/VV 은/ETM 책/NNG 을/JKO 보/VV ㄴ다/EF")])
    assert [c.lemma for c in s.annotations[0].subject_candidates] == ["학생"]


def test_each_anonymized_occurrence_is_distinct_entity():
    text = "#@이름#는 #@이름#는 간다."
    tokens = [Token("#@", "SW", 0, 2), Token("이름", "NNG", 2, 4), Token("#", "SW", 4, 5),
              Token("는", "JX", 5, 6), Token("#@", "SW", 7, 9), Token("이름", "NNG", 9, 11),
              Token("#", "SW", 11, 12), Token("는", "JX", 12, 13), Token("가", "VV", 14, 15),
              Token("ㄴ다", "EF", 14, 16), Token(".", "SF", 16, 17)]
    sentence = Sentence("1.1", 1, 0, len(text), text, tokens)
    Analyzer(None)._annotate(sentence, [])
    structure = annotate(text, Profile([sentence], [], "bareun", "fixture"))
    a = structure.annotations[0]
    assert len(a.subject_candidates) == 2
    assert len({c.entity_id for c in a.subject_candidates}) == 2
    assert all(c.surface == "#@이름#는" for c in a.subject_candidates)
    assert "#@이름#는@0:5" in render(structure) and "#@이름#는@7:12" in render(structure)


def test_edge_confidence_same_paragraph_distance_and_multi_unit():
    s = fixture_structure([(1, "나/NP 는/JX 읽/VV 는다/EF"),
                           (1, "웃/VV 는다/EF"), (2, "웃/VV 는다/EF"),
                           (2, "웃/VV 는다/EF ./SF 울/VV ㄴ다/EF")])
    ref = {e.src: e for e in s.edges if e.kind == "REF"}
    assert ref["S2"].confidence == "HIGH" and dependent_eligible(ref["S2"])
    assert ref["S3"].confidence == ref["S4"].confidence == "LOW"
    assert not dependent_eligible(ref["S3"])
    assert f"REF→S1({ref['S3'].label})?" in render(s)


def test_all_annotation_fields_and_change_types_have_levels_not_in_view():
    s = fixture_structure([(1, "학생/NNG 은/JX 가/VV 지/EC 도/JX 않/VX 습니다/EF")])
    value = s.to_dict()
    fields_in_schema = {f.name for f in fields(Annotation)}
    assert fields_in_schema == set(ANNOTATION_LEVELS) | {"uncertain"}
    assert fields_in_schema == set(value["annotations"][0]["field_levels"])
    assert set(ANNOTATION_LEVELS.values()) == {"WORD", "SENTENCE", "TEXT"}
    assert s.annotations[0].focus_particles[-1]["level"] == "WORD"
    assert all(cohesion_change(kind)["level"] == level for kind, level in COHESION_CHANGE_LEVELS.items())
    with pytest.raises(KeyError): cohesion_change("unregistered")
    assert not any(level in render(s) for level in ("WORD", "SENTENCE", "TEXT"))
