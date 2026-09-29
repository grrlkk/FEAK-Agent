import os

import pytest

from verak.src.analyzer import Analyzer, BareunBackend
from verak.src.change_info import align, extract, surface_diff
from verak.src.schemas import Profile, Sentence, Token


def plain_profile(text, parts=None, tokens=None):
    parts = parts or [text]
    sentences, offset = [], 0
    for i, part in enumerate(parts):
        start = text.index(part, offset)
        end = start + len(part)
        sentences.append(Sentence(f"1.{i+1}", 1, start, end, part, (tokens or {}).get(i, [])))
        offset = end
    return Profile(sentences, [], "manual_fixture", "1")


def test_morpheme_connection_argument_deletion_noun_and_negation_are_not_lost():
    a, b = "비가 와서 늦어졌다.", "비가 왔고 늦어졌다."
    ap = plain_profile(a, tokens={0: [Token("오", "VV", 3, 4), Token("아서", "EC", 3, 5)]})
    bp = plain_profile(b, tokens={0: [Token("오", "VV", 3, 4), Token("았", "EP", 3, 4), Token("고", "EC", 4, 5)]})
    units = extract(a, b, ap, bp)
    assert "connective" in units[0].types
    assert units[0].before_text == a and units[0].after_text == b
    a, b = "책을 읽었다.", "신문을 읽지 않았다."
    ap = plain_profile(a, tokens={0: [Token("책", "NNG", 0, 1), Token("을", "JKO", 1, 2)]})
    bp = plain_profile(b, tokens={0: [Token("신문", "NNG", 0, 2), Token("을", "JKO", 2, 3), Token("않", "VX", 7, 8)]})
    assert {"noun", "negation"} <= set(extract(a, b, ap, bp)[0].types)


def test_argument_reinsertion_remains_a_candidate_and_never_resolves_recoverability():
    a, b = "발표했다.", "학생들은 발표했다."
    ap = plain_profile(a)
    bp = plain_profile(b, tokens={0: [Token("학생", "NNG", 0, 2), Token("들", "XSN", 2, 3), Token("은", "JX", 3, 4)]})
    inserted = extract(a, b, ap, bp)
    assert "arg_insert" in inserted[0].types
    assert "argument_reference_and_recoverability_not_resolved" in inserted[0].uncertain
    assert "recoverable" not in inserted[0].interpretations
    deleted = extract(b, a, bp, ap)
    assert "arg_delete" in deleted[0].types


def test_sentence_insertion_deletion_merge_split_and_surface_boundaries():
    a, b = "비가 왔다.", "비가 왔다. 우산을 폈다."
    ap, bp = plain_profile(a), plain_profile(b, ["비가 왔다.", "우산을 폈다."])
    assert any(u.alignment == "0:1" for u in extract(a, b, ap, bp))
    assert any(u.alignment == "1:0" for u in extract(b, a, bp, ap))
    a, b = "비가 왔다. 길이 젖었다.", "비가 와서 길이 젖었다."
    ap, bp = plain_profile(a, ["비가 왔다.", "길이 젖었다."]), plain_profile(b)
    assert any("merge" in u.types for u in extract(a, b, ap, bp))
    assert any("split" in u.types for u in extract(b, a, bp, ap))
    a, b = "비가 왔다.  길이 젖었다.", "비가 왔다.\n길이 젖었다."
    parts = ["비가 왔다.", "길이 젖었다."]
    units = extract(a, b, plain_profile(a, parts), plain_profile(b, parts))
    assert units and all(u.types == ["spacing"] for u in units)
    for unit in units:
        assert a[slice(*unit.before_span)] == unit.before_text
        assert b[slice(*unit.after_span)] == unit.after_text


def test_spacing_label_requires_all_actual_changes_to_be_whitespace():
    a, b = "한  문장이다.", "한 문장이다."
    assert extract(a, b, plain_profile(a), plain_profile(b))[0].types == ["spacing"]
    c = "다른 문장이다."
    assert "spacing" not in extract(a, c, plain_profile(a), plain_profile(c))[0].types


@pytest.fixture
def live_analyzer():
    if os.getenv("VERAK_LIVE_TAGS") != "1":
        pytest.skip("Opt-in real Bareun integration")
    from dotenv import load_dotenv
    load_dotenv("/home/chanwoo/essay_scoring_llm/.env", override=False)
    return Analyzer(BareunBackend())


@pytest.mark.parametrize("text,style,relation", [
    ("학생은 책을 읽는다.", "해라체", None),
    ("학생이 책을 읽습니다.", "하십시오체", None),
    ("우리는 책을 읽어요.", "해요체", None),
    ("비가 와서 늦어졌다.", "해라체", "원인"),
    ("비가 왔고 늦어졌다.", "해라체", "나열"),
    ("민수도 참석하지만 영희만 발표한다.", "해라체", "대조"),
    ("학교에서는 뛰면 안 된다.", "해라체", "조건"),
    ("친구가 오니까 문을 열자.", "해라체", "원인"),
    ("이 길로 가십시오.", "하십시오체", None),
    ("학생들은 자료를 모았다. 다음 날 발표했다.", "해라체", None),
])
def test_ten_authored_profile_examples(live_analyzer, text, style, relation):
    profile = live_analyzer.profile(text)
    assert all(style in s.style_candidates for s in profile.sentences)
    if relation:
        assert any(relation in c["candidates"] for s in profile.sentences for c in s.connectives)
    if len(profile.sentences) == 2:
        assert profile.sentences[1].antecedent_candidates[0]["text"] == "학생들은"
        assert "no_subject_marker_does_not_prove_ellipsis" in profile.sentences[1].uncertain
    assert all(text[s.start:s.end] == s.text for s in profile.sentences)


def test_real_changed_units_and_unicode_offsets(live_analyzer):
    pairs = [
        ("비가 와서 늦어졌다.", "비가 왔고 늦어졌다.", "connective"),
        ("학생들은 자료를 모았다. 이후 발표했다.", "학생들은 자료를 모았다. 이후 학생들은 발표했다.", "arg_insert"),
        ("학생들은 자료를 모았다. 이후 학생들은 발표했다.", "학생들은 자료를 모았다. 이후 발표했다.", "arg_delete"),
        ("학생은  책을 읽는다.", "학생은 책을 읽는다.", "spacing"),
        ("학생이 책을 읽는다.", "학생이 책을 읽습니다.", "ender"),
        ("민수가 책을 읽는다.", "민수만 책을 읽는다.", "particle_focus"),
        ("책을 읽는다.", "책을 읽지 않는다.", "negation"),
    ]
    for a, b, label in pairs:
        units = extract(a, b, live_analyzer.profile(a), live_analyzer.profile(b), live_analyzer.lexicons["focus"])
        assert any(label in u.types for u in units), (a, b, units)
    text = "  학생들은 자료를 모았다.\r\n\n 😀 이후 발표했다.  "
    result = live_analyzer.profile(text)
    assert all(text[s.start:s.end] == s.text for s in result.sentences)
    assert result.sentences[-1].antecedent_candidates
