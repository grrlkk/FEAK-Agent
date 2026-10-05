"""Phase 2c contracts on authored fixtures and the requested cached regression."""

from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import pytest

from verak.src.schemas import Profile, Sentence, Token
from verak.v3.common import load_config, read_json
from verak.v3.ko import annotate_structural, render, render_with_budget
from verak.v3.ko.coarse import CLOSED_CONJUNCTIONS, coarse_conjunction, relation_swap_allowed
from verak.v3.ko.structural import (FIELD_LEVELS, StructuralAnnotation, dependency_changes,
                                    subject_observation)
from verak.v3.phase2 import restore_profile


def fixture(rows, **kwargs):
    sentences, parts, offset = [], [], 0
    for paragraph, pairs in rows:
        tokens, pos = [], offset
        for pair in pairs.split():
            form, tag = pair.rsplit("/", 1)
            tokens.append(Token(form, tag, pos, pos + len(form)))
            pos += len(form) + 1
        text = " ".join(t.form for t in tokens)
        sentences.append(Sentence(str(len(sentences) + 1), paragraph, offset, offset + len(text), text, tokens))
        parts.append(text)
        offset += len(text) + 1
    return annotate_structural("\n".join(parts), Profile(sentences, [], "bareun", "authored"), **kwargs)


@pytest.mark.parametrize("subject", ["이/NP 는/JX", "이것/NP 은/JX", "그것/NP 은/JX", "우리/NP 는/JX",
    "나/NP 는/JX", "이러하/VA ㄴ/ETM 학생/NNG 은/JX", "학생/NNG 이/JKS", "학생/NNG 도/JX"])
def test_marked_np_survives_relative_clause(subject):
    s = fixture([(1, subject + " 좋/VA ㄴ/ETM 일/NNG 을/JKO 하/VV ㄴ다/EF")])
    assert s.annotations[0].subject_omitted is False
    assert not any(e["kind"] == "DEP" for e in s.edges)


def test_valid_6862_s31_ineun_regression():
    # The full original essay is kept local, never copied into the repository.
    path = Path(__file__).resolve().parents[1] / "outputs/phase2/bareun_profiles/6862.json"
    if not path.exists():
        pytest.skip("Private cached valid:6862 profile unavailable; authored 이는 regression still runs")
    sentence = restore_profile(read_json(path)["profile"]).sentences[30]
    assert sentence.text.startswith("이는")
    assert subject_observation(sentence)[0] is False


@pytest.mark.parametrize("pairs", ["일/NNG 을/JKO 하/VV ㄴ다/EF", "공부/NNG 하/XSV ㄴ다/EF"])
def test_clear_omissions(pairs):
    assert fixture([(1, pairs)]).annotations[0].subject_omitted


@pytest.mark.parametrize("pairs", ["나/NP 또한/MAG 공부하/VV ㄴ다/EF", "둘/NR 다/MAG 좋/VA 다/EF",
                                    "오늘/NNG 의/JKG 과제/NNG"])
def test_uncertain_unmarked_subject_or_fragment_is_not_omission(pairs):
    ann = fixture([(1, pairs)]).annotations[0]
    assert not ann.subject_omitted and ann.omission_uncertain


def test_marker_subjects_are_distinct_and_preserved():
    s = fixture([(1, "#@이름#/SW 은/JX 공부하/VV ㄴ다/EF"), (1, "#@이름#/SW 은/JX 웃/VV 는다/EF")])
    assert all(not a.subject_omitted for a in s.annotations)
    ids = [a.subject_evidence[0]["entity_id"] for a in s.annotations]
    assert ids[0] != ids[1]


def test_nominal_display_preserves_negative_prefix_and_plural_suffix():
    text = "불평등은 크다. 학생들은 웃는다."
    tokens = [Token("불", "XPN", 0, 1), Token("평등", "NNG", 1, 3), Token("은", "JX", 3, 4),
              Token("크", "VA", 5, 6), Token("다", "EF", 6, 7), Token(".", "SF", 7, 8),
              Token("학생", "NNG", 9, 11), Token("들", "XSN", 11, 12), Token("은", "JX", 12, 13),
              Token("웃", "VV", 14, 15), Token("는다", "EF", 15, 17)]
    s = Sentence("S1", 1, 0, len(text), text, tokens)
    omitted, evidence, _ = subject_observation(s)
    assert not omitted
    assert [e["surface"] for e in evidence] == ["불평등은", "학생들은"]
    assert all(text[e["span"][0]:e["span"][1]] == e["surface"] for e in evidence)


def test_anonymous_possessive_np_display_keeps_occurrence():
    text = "#@이름#네 반은 좋다."
    tokens = [Token("#@", "SW", 0, 2), Token("이름", "NNG", 2, 4), Token("#", "SW", 4, 5),
              Token("네", "NNG", 5, 6), Token("반", "NNG", 7, 8), Token("은", "JX", 8, 9),
              Token("좋", "VA", 10, 11), Token("다", "EF", 11, 12)]
    _, evidence, _ = subject_observation(Sentence("S1", 1, 0, len(text), text, tokens))
    assert evidence[0]["surface"] == "#@이름#네 반은"
    assert evidence[0]["entity_id"] == "#@이름#@0"


def test_structural_predecessors_cross_paragraph_and_stable_move():
    rows = [(1, "학생/NNG 은/JX 공부하/VV ㄴ다/EF"), (1, "복습하/VV ㄴ다/EF"), (2, "웃/VV 는다/EF")]
    source = fixture(rows, include_debug=True)
    assert [(a.predecessor_id, a.cross_paragraph) for a in source.annotations] == [(None, False), ("S1", False), ("S2", True)]
    assert {(e["kind"], e["src"], e["dst"]) for e in source.edges} == {("DEP", "S2", "S1"), ("DEP", "S3", "S2")}
    moved = fixture([rows[0], rows[2], rows[1]], sentence_ids=["S1", "S3", "S2"])
    changes = dependency_changes(source, moved)
    assert len(changes) == 2
    assert changes[0]["message"] == "S3 (subject omitted): preceding sentence changed S2 → S1"
    assert not any("broken" in c["message"] for c in changes)
    explicit = fixture([rows[0], rows[2], (1, "학생/NNG 이/JKS 복습하/VV ㄴ다/EF")], sentence_ids=["S1", "S3", "S2"])
    assert not any(c["sid"] == "S2" for c in dependency_changes(source, explicit))
    active = source.to_dict()
    assert "antecedent" not in str(active) and "REF" not in str(active) and "TOPIC" not in str(active["edges"])
    assert "phase2b_antecedents_debug_only" in source.to_dict(include_debug=True)["debug"]
    assert "REF" not in render(source) and "선행사" not in render(source)
    with pytest.raises(TypeError):
        dependency_changes({}, source)


def test_document_initial_dep_has_null_predecessor():
    s = fixture([(1, "웃/VV 는다/EF")])
    assert s.edges[0]["dst"] is None and "DEP→START" in render(s)


@pytest.mark.parametrize("form,expected", [("면", "CONDITION"), ("거든", "CONDITION"),
    ("니까", "CAUSE"), ("므로", "CAUSE"), ("지만", "ADVERSATIVE"), ("더라도", "ADVERSATIVE"),
    ("어도", "ADVERSATIVE"), ("려고", "PURPOSE"), ("고자", "PURPOSE")])
def test_coarse_ec(form, expected):
    c = fixture([(1, f"읽/VV {form}/EC 이해하/VV ㄴ다/EF")]).annotations[0].connectives[0]
    assert c["classification"] == "UNAMBIGUOUS" and c["coarse_class"] == expected


@pytest.mark.parametrize("pairs", ["예/NNG 를/JKO 들/VV 면/EC", "다시/MAG 말하/VV 면/EC",
    "바꾸/VV 어/EC 말하/VV 면/EC"])
def test_fixed_expressions_do_not_create_condition(pairs):
    s = fixture([(1, pairs + " 좋/VA 다/EF")])
    c = next(c for c in s.annotations[0].connectives if c["form"] == "면")
    assert not c["eligible"] and c["exclusion_reason"] == "fixed_expression"


@pytest.mark.parametrize("pairs,reason", [("읽/VV 어도/EC 되/VV ㄴ다/EF", "permission_not_concession"),
    ("읽/VV 거든/EC ./SF", "conditional_use_unconfirmed"), ("먹/VV 고/EC 있/VX 다/EF", "auxiliary")])
def test_excluded_ec(pairs, reason):
    c = fixture([(1, pairs)]).annotations[0].connectives[0]
    assert not c["eligible"] and c["exclusion_reason"] == reason


def test_nominal_cause_does_not_invent_ec_and_swap_is_cross_class_only():
    ann = fixture([(1, "오/VV 기/ETN 때문/NNB 에/JKB 늦/VV 었/EP 다/EF")]).annotations[0]
    c = ann.connectives[0]
    assert c["kind"] == "construction" and c["coarse_class"] == "CAUSE"
    assert relation_swap_allowed(c, {"eligible": True, "coarse_class": "CONDITION"})
    assert not relation_swap_allowed(c, {"eligible": True, "coarse_class": "CAUSE"})
    assert not relation_swap_allowed(c, {"eligible": False, "coarse_class": "CONDITION"})


@pytest.mark.parametrize("form,relation", list(CLOSED_CONJUNCTIONS.items()))
def test_closed_conjunctions(form, relation):
    c = coarse_conjunction(form.replace(" ", "  ") + ", 결과는 같았다.", {})
    assert c["eligible"] and c["coarse_class"] == relation
    assert coarse_conjunction("학생들은 " + form + " 결과를 확인했다.", {}) is None


def test_open_conjunction_visible_but_not_eligible():
    c = coarse_conjunction("그리고 결과를 보았다.", {"그리고": "ADDITION"})
    assert c["visible_relation"] == "ADDITION" and not c["eligible"]
    assert coarse_conjunction("그래서인지 결과가 좋다.", {}) is None


@pytest.mark.parametrize("embedded", ["오/VV 니/EF 고/JKQ", "오/VV ㄹ까/EF 를/JKO",
    '“/SS 오/VV 습니다/EF ”/SS 라고/JKQ'])
def test_style_only_last_outside_ef_and_multi_unit_is_independent(embedded):
    s = fixture([(1, embedded + " 묻/VV 는다/EF")])
    a = s.annotations[0]
    assert a.style == s.dominant_style == "한다" and a.multi_unit
    assert not a.off_style and len([e for e in a.final_endings if e["selected"]]) == 1
    a.multi_unit = False
    assert a.style == "한다"  # A corruption eligibility flag never changes register.


@pytest.mark.parametrize("ending", ["ㄹ까", "을까", "니"])
@pytest.mark.parametrize("previous,expected", [("다", "한다"), ("어", "해"), ("습니다", "unknown")])
def test_ambiguous_ending_uses_previous_final_style(ending, previous, expected):
    s = fixture([(1, f"가/VV {previous}/EF"), (2, f"오/VV {ending}/EF")])
    assert s.annotations[1].style == expected


def test_ambiguous_first_sentence_unknown_and_polite_particle_explicit():
    assert fixture([(1, "가/VV ㄹ까/EF")]).annotations[0].style == "unknown"
    assert fixture([(1, "가/VV ㄹ까/EF 요/JX")]).annotations[0].style == "해요"


def test_quoted_only_ef_is_not_a_sentence_style_and_majority():
    s = fixture([(1, '“/SS 오/VV 습니다/EF ”/SS'), (1, "좋/VA 다/EF"),
                 (1, "좋/VA 다/EF"), (1, "좋/VA 습니다/EF")])
    assert s.annotations[0].style == "unknown"
    assert s.dominant_style == "한다"
    assert [a.off_style for a in s.annotations] == [False, False, False, True]


def test_active_fields_levels_compact_budget_and_equal_weights():
    assert set(FIELD_LEVELS) == {f.name for f in fields(StructuralAnnotation)} - {"uncertain"}
    s = fixture([(1, "읽/VV 면/EC 좋/VA 다/EF"), (1, "학생/NNG 은/JX 공부하/VV ㄴ다/EF")])
    before = deepcopy(s.to_dict())
    text = render(s)
    assert text.count("한다") == 1 and "CONDITION" in text and "WORD" not in text
    assert not render_with_budget(s, len, 1)["eligible"]
    assert s.to_dict() == before
    config = load_config()["structure_policy"]
    assert config["level_weights"] == {"WORD": 1, "SENTENCE": 1, "TEXT": 1}
    assert config["dependent_edges"] == "DEP_only"
    assert config["dependency_recovery_from_phase"] == 4
    assert config["dependency_recovery"] == "predecessor_restored_or_explicit_subject"
