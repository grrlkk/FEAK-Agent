import json
from pathlib import Path

import pytest

from verak.src.diagnoser import (Diagnoser, FEEDBACK_ALIASES, expand_goal_range, parse_diagnosis,
                                 set_goal, user_prompt, validate_goal)
from verak.src.judge import candidate_side, judge_pair
from verak.src.llm import JSONFailure, LLM
from verak.src.reviser import hard_checks, replace_span
from verak.src.run_single import load_samples, process_sample, read_sample
from verak.src.schemas import (Evidence, GlobalJudgment, Goal, GoalDraft, GoalRange, GoalResponse, PlannedChange,
                               Profile, Replacement, RUBRICS, Sentence, Unit, UnitJudgment)
from verak.src.spellcheck import SpellChecker, attach, compare, parse_bareun


def goal(text="학생들이가 발표했다."):
    return Goal(rubric="어법적절성", target_sents=["1.1"], span_before=[0, len(text)], intent="중복 조사를 고친다",
                evidence=[Evidence(source="draft", quote=text)], allowed_changes=[])


def raw_diagnosis():
    return "8 7 6 5 4 3 2 1\n### Feedback:\n" + "\n".join(f"- {r}: 피드백 {i}" for i, r in enumerate(RUBRICS))


def test_training_input_is_parsed_without_gold_scores_or_keyword_leakage():
    sample = read_sample({"user": "질문: 문항\n에세이:  글이다.\r\n끝이다.\n핵심 키워드: SECRET",
                          "assistant": "GOLD", "grader_1_scores": [9] * 8}, 7)
    assert sample["draft"] == " 글이다.\r\n끝이다."
    assert "SECRET" not in json.dumps(sample) and "GOLD" not in json.dumps(sample)
    assert user_prompt(sample["question"], sample["draft"]) == "질문: 문항\n에세이:  글이다.\r\n끝이다."


def test_diagnosis_requires_all_eight_integer_scores_and_all_feedback():
    parsed = parse_diagnosis(raw_diagnosis())
    assert list(parsed["scores"].values()) == list(range(8, 0, -1))
    assert len(parsed["feedback"]) == 8
    for text in (raw_diagnosis().replace("8 7", "10 7"), raw_diagnosis().replace("8 7", "8.1 7"),
                 raw_diagnosis().replace("### Feedback:", "feedback"), raw_diagnosis().rsplit("\n", 1)[0]):
        with pytest.raises(ValueError): parse_diagnosis(text)


def test_scorer_retries_parse_failures_twice_with_same_input():
    calls, records = [], []
    def generate(system, user):
        calls.append((system, user))
        return "bad" if len(calls) < 3 else raw_diagnosis()
    result = Diagnoser(generate, records.append).diagnose("문항", "글이다.")
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert all("핵심 키워드" not in user for _, user in calls)
    assert result["scores"]["어법적절성"] == 1


def test_actual_ft_feedback_headers_are_mapped_without_changing_scores():
    raw = raw_diagnosis()
    for alias, canonical in FEEDBACK_ALIASES.items():
        raw = raw.replace(f"- {canonical}:", f"- {alias}:")
    parsed = parse_diagnosis(raw)
    assert set(parsed["feedback"]) == set(RUBRICS)
    assert list(parsed["scores"].values()) == list(range(8, 0, -1))


def test_goal_offsets_are_calculated_and_evidence_source_is_enforced():
    text = "  앞 문장.\r\n학생들이가 발표했다.  "
    start = text.index("학생")
    profile = Profile([Sentence("1.1", 1, 2, 7, text[2:7], []),
                       Sentence("2.1", 2, start, len(text)-2, text[start:-2], [])], [], "fixture", "1")
    draft = GoalDraft(rubric="어법적절성", target_sents=["2.1"], intent="조사 오류 수정",
                      evidence=[Evidence(source="draft", quote="학생들이가")], allowed_changes=[])
    selected = validate_goal(draft, "문항", text, profile, "어법적절성")
    assert selected.span_before == [start, len(text)-2]
    updated = replace_span(text, selected, "학생들이 발표했다.")
    assert updated[:start].encode() == text[:start].encode()
    assert updated[-2:].encode() == text[-2:].encode()
    draft.evidence[0].source = "question"
    with pytest.raises(ValueError): validate_goal(draft, "문항", text, profile, "어법적절성")
    draft.evidence[0].source = "draft"
    draft.target_sents = ["2.1", "1.1"]
    with pytest.raises(ValueError): validate_goal(draft, "문항", text, profile, "어법적절성")


def test_explicit_scope_markers_and_no_change_are_hard_constraints():
    text = "#@이름#은 글을 쓴다."
    request = goal(text)
    pattern = r"#@[^#\r\n]+#"
    assert "anonymization_marker_changed" in hard_checks(text, "민수는 글을 쓴다.", request, pattern)[1]
    assert "no_change" in hard_checks(text, text, request, pattern)[1]
    assert hard_checks(text, "#@이름#은 글을 씁니다.", request, pattern)[0]  # Style is not a hard rule.
    request.span_before = [0, 5]
    assert "outside_allocated_scope" in hard_checks(text, "글을 쓴다.", request)[1]


def test_goal_endpoints_include_intermediate_sentences_without_another_generation():
    text = "앞. 가운데. 뒤."
    profile = Profile([Sentence("1.1", 1, 0, 2, "앞.", []),
                       Sentence("1.2", 1, 3, 7, "가운데.", []),
                       Sentence("1.3", 1, 8, 10, "뒤.", [])], [], "fixture", "1")
    selection = GoalRange(rubric="어법적절성", first_sentence="1.1", last_sentence="1.3",
                          intent="표현 정리", evidence_ids=["draft:1.2", "question"],
                          allowed_changes=[PlannedChange(scope_sentence="1.2", expected="표현 명료화", reason="문맥에 맞춤")])
    draft = expand_goal_range(selection, profile, "문항")
    assert draft.target_sents == ["1.1", "1.2", "1.3"]
    assert draft.evidence == [Evidence(source="draft", quote="가운데."), Evidence(source="question", quote="문항")]
    assert draft.allowed_changes[0].scope_quote == "가운데."
    assert validate_goal(draft, "문항", text, profile, "어법적절성").span_before == [0, 10]
    selection.evidence_ids = ["feedback:1"]
    with pytest.raises(ValueError): expand_goal_range(selection, profile, "문항")
    selection.evidence_ids = ["question"]
    selection.allowed_changes[0].scope_sentence = "2.1"
    with pytest.raises(ValueError): expand_goal_range(selection, profile, "문항")
    selection.allowed_changes = []
    selection.first_sentence, selection.last_sentence = "1.3", "1.1"
    with pytest.raises(ValueError): expand_goal_range(selection, profile, "문항")
    selection.first_sentence = "2.1"
    with pytest.raises(ValueError): expand_goal_range(selection, profile, "문항")


def spell_report(issues):
    return {"status": "available", "issues": issues}


def test_spelling_cache_retries_and_unavailable_is_not_clean(tmp_path):
    calls, records = [], []
    def fail(text): calls.append(text); raise TimeoutError()
    cfg = {"endpoint": "https://fixture.invalid", "min_interval_s": 0, "cache": str(tmp_path / "cache.jsonl")}
    checker = SpellChecker(cfg, fail, records.append)
    assert checker.check("글") is None
    assert len(calls) == 3
    assert checker.report("글")["status"] == "unavailable" and len(calls) == 3
    assert compare("글", "다른 글", checker.report("글"), spell_report([]))["status"] == "unavailable"
    good = SpellChecker(cfg, lambda text: {"origin": text, "revised": text}, records.append)
    assert good.check("글") == []
    def never(text): raise AssertionError("Should use persistent cache")
    assert SpellChecker(cfg, never).check("글") == []


def test_spelling_fixed_introduced_and_persistent_with_offset_shift():
    before, after = "틀린 글. 오류다.", "맞는 새 글. 오류다. 새오류."
    fixed = dict(start=0, end=2, original="틀린", suggestion="맞는", category="TYPO")
    persistent = dict(start=6, end=8, original="오류", suggestion="오류수정", category="TYPO")
    shifted = dict(persistent, start=8, end=10)
    introduced = dict(start=13, end=16, original="새오류", suggestion="수정", category="TYPO")
    result = compare(before, after, spell_report([fixed, persistent]), spell_report([shifted, introduced]))
    assert result["fixed"] == [fixed] and result["introduced"] == [introduced]
    assert result["persistent"] == [{"before": persistent, "after": shifted}]
    unit = Unit("U1", [0, 3], [0, 4], "틀린 ", "맞는 새", ["noun"], [], [], [], "1:1", [])
    attach([unit], result)
    assert unit.spelling["fixed"] == [fixed] and not unit.spelling["introduced"]


def test_spellcheck_rejects_invalid_offsets_and_preserves_surface_only_corrections():
    assert parse_bareun("가 나", {"origin": "가 나", "revised": "가나"})[0]["original"] == " "
    with pytest.raises(ValueError):
        parse_bareun("가 나", {"origin": "가 나", "revised": "가나", "revised_blocks": [
            {"origin": {"content": "가", "begin_offset": 2}, "revised": "나"}]})


class JudgeSpy:
    def __init__(self, side="equal", fail=False): self.calls, self.side, self.fail = [], side, fail
    def request(self, schema, prompt, payload, **kwargs):
        self.calls.append((schema, prompt, payload, kwargs))
        if self.fail: raise JSONFailure("failed")
        if schema is UnitJudgment:
            result = UnitJudgment(goal_valid="pass", goal_improved="pass", selective="pass", meaning="pass", issues=[])
        else:
            result = GlobalJudgment(preference=self.side, issues=[])
        kwargs["validate"](result)
        return result


def judge(spy):
    old, new = "학생들이가 발표했다.", "학생들이 발표했다."
    unit = Unit("U1", [0, len(old)], [0, len(new)], old, new, ["particle_case"],
                [{"kind": "TEST_MORPH"}], [{"TEST_INTERPRETATION": True}], [], "1:1", [],
                {"status": "available", "fixed": [], "introduced": []})
    return judge_pair("pair", "발표 과정을 설명하시오", old, new, goal(old), [unit],
        {"status": "available", "fixed": [], "introduced": []}, llm=spy, unit_prompt="UNIT PROMPT",
        global_prompt="GLOBAL PROMPT")


def test_judge_information_is_the_only_condition_difference_and_no_scores_leak():
    spy = JudgeSpy()
    results = judge(spy)
    assert len(spy.calls) == 6
    assert len({r["candidate_side"] for r in results.values()}) == 1
    for offset in (0, 1):
        calls = spy.calls[offset::2]
        assert len({call[1] for call in calls}) == 1
        payloads = [call[2] for call in calls]
        assert set(payloads[1]) - set(payloads[0]) == {"surface_diff"}
        assert set(payloads[2]) - set(payloads[1]) == {"korean_changes", "spelling"}
        for key in payloads[0]: assert all(p[key] == payloads[0][key] for p in payloads)
        for p in payloads[:2]:
            assert "TEST_MORPH" not in json.dumps(p) and "TEST_INTERPRETATION" not in json.dumps(p)
            assert "spelling" not in p and "diagnosis" not in p and "scores" not in p
        assert all(call[3]["retries"] == 2 for call in calls)
    assert all(r["accept"] for r in results.values())  # Equal + actual goal improvement is permitted.


def test_global_ab_choice_is_mapped_to_candidate_and_failure_becomes_unknown():
    side = candidate_side("pair", "학생들이가 발표했다.", "학생들이 발표했다.", 42)
    assert all(r["global"] == "better" for r in judge(JudgeSpy(side)).values())
    opposite = "B" if side == "A" else "A"
    assert all(not r["accept"] for r in judge(JudgeSpy(opposite)).values())
    failed = JudgeSpy(fail=True)
    assert all(r["global"] == "unknown" and not r["accept"] for r in judge(failed).values())
    assert len(failed.calls) == 6


def test_llm_validation_retry_uses_identical_requests_and_does_not_leak_last_answer():
    calls = []
    def client(**kwargs): calls.append(kwargs); return {"bad": "OLD REASON MUST NOT LEAK"}
    llm = LLM({"provider": "openai", "model": "gpt-5-mini"}, client=client)
    with pytest.raises(JSONFailure):
        llm.request(Replacement, "prompt", {"text": "원문"}, role="j_units", retries=2)
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert "OLD REASON" not in calls[2]["user"]


def test_pipeline_freezes_one_candidate_spelling_failure_does_not_stop_or_adopt():
    old, new = "학생들이가 발표했다.", "학생들이 발표했다."
    class AnalyzerStub:
        lexicons = {"focus": {}}
        def profile(self, text):
            return Profile([Sentence("1.1", 1, 0, len(text), text, [])], [], "fixture", "1")
    class Generator:
        calls = []
        def request(self, schema, prompt, payload, **kwargs):
            self.calls.append((schema, payload))
            if schema is GoalResponse:
                g = GoalRange(rubric="어법적절성", intent="중복 조사를 고친다", allowed_changes=[],
                              evidence_ids=["draft:1.1"], first_sentence="1.1", last_sentence="1.1")
                result = GoalResponse(goal=g, reason="조사 중복")
            else: result = Replacement(replacement=new)
            if kwargs.get("validate"): kwargs["validate"](result)
            return result
    generator, spy, pairs = Generator(), JudgeSpy(), []
    def fail(text): raise TimeoutError()
    result = process_sample({"id": "sample", "question": "문항", "draft": old}, analyzer=AnalyzerStub(),
        diagnoser=Diagnoser(lambda system, user: raw_diagnosis()), generator=generator, judge_llm=spy,
        spellchecker=SpellChecker({"endpoint": "fixture", "min_interval_s": 0}, fail),
        config={"anonymization_pattern": r"#@[^#]+#", "seed": 42,
                "judge_info": ["criteria_only", "surface_diff", "korean"]},
        prompts={key: key for key in ("goal", "revise", "j_units", "j_global")}, on_pair=pairs.append)
    assert result["status"] == "completed" and result["returned_text"] == old
    assert result["candidate"] == new and len(pairs) == 1 and len(generator.calls) == 2
    assert result["spelling"]["status"] == "unavailable" and len(spy.calls) == 6
    assert all(call[2]["after"] == new for call in spy.calls[::2])
