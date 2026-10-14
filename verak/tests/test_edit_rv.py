"""Edit coverage, exact surfaces, deterministic gates and fail-closed verdicts."""
from copy import deepcopy
import random
import re

import pytest

from verak.src.change_info import (readable_edits, edit_verification_info,
                                    plan_edit_budget, surface_diff)
from verak.src.judge import judge_scoped
from verak.src.schemas import (ScopePlan, ResolvedTarget, Evidence, Profile, Sentence, Token,
                               EditJudgment, EditJudgments, EditIssue)


def profile(text):
    sentences = []
    for i, match in enumerate(re.finditer(r'[^.!?\n]+[.!?]?', text)):
        old = match.group()
        value = old.strip()
        if not value:
            continue
        a = match.start() + len(old) - len(old.lstrip())
        end = a + len(value)
        tokens = []
        for token in re.finditer(r'없|있|습니다|는다|다|면|니까|에서|에게|를|에|는', value):
            form = token.group()
            tag = ('VA' if form in {'없', '있'} else 'EF' if form in {'습니다', '는다', '다'}
                   else 'EC' if form in {'면', '니까'} else 'JKB')
            tokens.append(Token(form, tag, a+token.start(), a+token.end()))
        style = ['하십시오체'] if value.endswith('습니다.') else ['해라체'] if value.endswith('다.') else []
        sentences.append(Sentence(str(i), 1, a, end, value, tokens, style_candidates=style))
    return Profile(sentences, [], 'fixture', '1')


def plan(text, action='REWRITE', target=None, goal='어색한 표현을 고친다', reason='해당 문장만 고친다'):
    target = target or (0, text.find('.')+1)
    a, b = target
    return ScopePlan(rubric='어법적절성', goal=goal, scope='sentence', action=action,
        target=ResolvedTarget(start=a, end=b, text=text[a:b], insertion_at=b if action == 'ADD' else None,
                              pieces=[]), preserve=[], evidence=[Evidence(source='draft', quote=text[a:b])],
        minimal_scope_reason=reason)


def info(before, after, contract):
    return edit_verification_info(before, after, contract, profile(before), profile(after))


class NoLLM:
    def request(self, *args, **kwargs):
        raise AssertionError('Mechanical violation must not call the LLM')


class Respond:
    def __init__(self, modify=lambda rows: rows):
        self.modify, self.payload = modify, None

    def request(self, schema, prompt, payload, **kwargs):
        self.payload = deepcopy(payload)
        rows = [EditJudgment(edit_id=edit['id'], necessity='pass', preservation='pass',
                groundedness='pass', meaning='pass', korean_consistency='pass',
                reason='원문 근거에 따른 필요한 표현 수정', issues=[]) for edit in payload['edits']]
        response = schema(edits=self.modify(rows))
        kwargs['validate'](response)
        return response


def judge(before, after, contract, llm=None):
    return judge_scoped('이유를 설명하시오', before, after, contract, surface_diff(before, after),
                         info(before, after, contract), llm=llm or NoLLM(), prompt='rv')


def test_spacing_and_negation_are_separate_readable_edits_with_exact_transitions():
    before, after = '생동감이 느껴질 수 밖에 없다. 뒤 문장이다.', '생동감이 느껴질 수밖에 있다. 뒤 문장이다.'
    data = info(before, after, plan(before))
    assert [(e['before_text'], e['after_text']) for e in data['edits']] == [('수 밖에', '수밖에'), ('없다.', '있다.')]
    changes = data['edits'][1]['transitions']
    assert next(t for t in changes if t['kind'] == 'negation')['before'][0]['form'] == '없'
    assert next(t for t in changes if t['kind'] == 'negation')['after'][0]['form'] == '있'
    assert all('tokens' not in e for e in data['edits'])


def test_add_is_anchored_and_split_into_actual_sentences_without_relocating_repeated_endings():
    before = '원문이다. 뒤다.'
    contract = plan(before, 'ADD', reason='1~2문장을 추가한다')
    addition = ' 추가다. 설명이다.'
    point = contract.target.insertion_at
    after = before[:point] + addition + before[point:]
    data = info(before, after, contract)
    assert len(data['edits']) == 2
    assert all(e['before_span'] == [point, point] and e['before_text'] == '' for e in data['edits'])
    assert ''.join(e['after_text'] for e in data['edits']) == addition
    assert data['counts']['budgeted_sentence_count'] == 2
    assert not data['hard_reasons']


@pytest.mark.parametrize('reason,maximum', [
    ('한 문장을 추가한다', 1), ('하나의 문장을 추가한다', 1), ('문장 하나만 추가', 1),
    ('한두 문장으로 보강', 2), ('1~2 sentences', 2), ('1~2문장 추가', 2),
    ('두 문장으로 분리', 2), ('문장만 고친다', None),
])
def test_budget_from_existing_plan_without_new_planner_fields(reason, maximum):
    assert plan_edit_budget(plan('원문이다.', reason=reason))['max_sentences'] == maximum


@pytest.mark.parametrize('reason', [
    '하나 이상의 완결된 예시 문장을 추가한다', '한 문장 이상을 추가한다',
    '문장 하나 이상을 추가한다', '최소 두 문장을 추가한다', '적어도 2문장이 필요하다',
    '2 sentences or more', 'at least 2 sentences',
])
def test_lower_bound_or_unspecified_budget_never_becomes_an_invented_cap(reason):
    before = '원문이다. 뒤다.'
    contract = plan(before, 'ADD', reason=reason)
    data = info(before, '원문이다. 예시다. 설명이다. 뒤다.', contract)
    assert data['budget']['max_sentences'] is None
    assert 'sentence_budget_exceeded' not in data['hard_reasons']


def test_five_sentences_violate_one_or_two_sentence_add_before_llm():
    before = '원문이다. 뒤다.'
    contract = plan(before, 'ADD', reason='1~2문장 추가면 충분하다')
    after = '원문이다. 하나다. 둘이다. 셋이다. 넷이다. 다섯이다. 뒤다.'
    result = judge(before, after, contract)
    assert result['verdict'] == 'fail' and not result['accept']
    assert 'sentence_budget_exceeded' in result['hard_reasons']


@pytest.mark.parametrize('action,after', [
    ('REWRITE', '앞 문장입니다. 뒤 문장이다.'),
    ('ADD', '앞 문장이다. 새로운 설명입니다. 뒤 문장이다.'),
])
def test_known_register_change_rejects_without_llm(action, after):
    # Actual Bareun's 합쇼체 EF is 습니다 or ㅂ니다; use an explicit fixture observation.
    before = '앞 문장이다. 뒤 문장이다.'
    contract = plan(before, action)
    p = profile(after)
    changed = next(s for s in p.sentences if '입니다' in s.text)
    changed.style_candidates = ['하십시오체']
    data = edit_verification_info(before, after, contract, profile(before), p)
    result = judge_scoped('문항', before, after, contract, surface_diff(before, after), data,
                         llm=NoLLM(), prompt='rv')
    assert 'speech_level_changed' in result['hard_reasons']
    assert result['korean_consistency'] == 'fail'


def test_same_register_ending_change_is_evidence_not_automatic_failure():
    before, after = '나는 먹는다. 뒤다.', '나는 먹었다. 뒤다.'
    data = info(before, after, plan(before))
    assert 'speech_level_changed' not in data['hard_reasons']
    assert any(t['kind'] == 'final_ending' for e in data['edits'] for t in e['transitions'])


def test_ambiguous_style_is_not_guessed_into_a_hard_violation():
    before, after = '앞이다. 뒤다.', '앞입니다. 뒤다.'
    left, right = profile(before), profile(after)
    left.sentences[0].style_candidates = ['해라체', '해요체']
    right.sentences[0].style_candidates = ['하십시오체']
    data = edit_verification_info(before, after, plan(before), left, right)
    assert 'speech_level_changed' not in data['hard_reasons']
    assert data['style_checks'][0]['status'] == 'unknown'


@pytest.mark.parametrize('action,after', [
    ('ADD', '다른 원문이다. 추가다. 뒤다.'),
    ('REWRITE', '원문이다. 다른 문장이다.'),
    ('DELETE', '새로 쓴 글이다. 뒤다.'),
])
def test_scope_and_action_violations_never_reach_llm(action, after):
    before = '원문이다. 뒤다.'
    result = judge(before, after, plan(before, action))
    assert result['verdict'] == 'fail' and result['skipped'] == 'mechanical_constraint'


@pytest.mark.parametrize('label', ['fail', 'unknown'])
def test_one_problematic_edit_rejects_whole_candidate(label):
    before, after = '수 밖에 없다. 뒤다.', '수밖에 있다. 뒤다.'
    def change(rows):
        rows[1].meaning = label
        rows[1].issues = [EditIssue(requirement='meaning', before_quote='없다.', after_quote='있다.', reason='부정 의미 변화')]
        return rows
    result = judge(before, after, plan(before), Respond(change))
    assert len(result['edits']) == 2 and result['edits'][0]['verdict'] == 'pass'
    assert result['verdict'] == label and not result['accept']


def test_fail_takes_priority_over_unknown_and_all_pass_is_required():
    before, after = '수 밖에 없다. 뒤다.', '수밖에 있다. 뒤다.'
    def change(rows):
        for row, label in zip(rows, ['unknown', 'fail']):
            row.meaning = label
            row.issues = [EditIssue(requirement='meaning', before_quote='', after_quote='', reason='의미 검증 근거')]
        return rows
    assert judge(before, after, plan(before), Respond(change))['verdict'] == 'fail'
    assert judge(before, after, plan(before), Respond())['accept']


@pytest.mark.parametrize('modify', [lambda rows: rows[:1], lambda rows: [rows[0], rows[0]],
                                    lambda rows: [*rows, rows[0].model_copy(update={'edit_id': 'E9'})]])
def test_missing_duplicate_or_invented_edit_verdict_is_invalid(modify):
    before, after = '수 밖에 없다. 뒤다.', '수밖에 있다. 뒤다.'
    with pytest.raises(ValueError, match='every edit'):
        judge(before, after, plan(before), Respond(modify))


def test_fabricated_morphological_sentence_quote_is_invalid():
    before = '겉으로 참고 기다리되, 재촉했다. 뒤다.'
    after = '겉으로는 인내하며 기다리되, 재촉했다. 뒤다.'
    def change(rows):
        rows[0].korean_consistency = 'fail'
        rows[0].issues = [EditIssue(requirement='korean_consistency', before_quote='',
                                  after_quote='인내하며 되,', reason='형태소 조각을 잘못 인용')]
        return rows
    with pytest.raises(ValueError, match='quote'):
        judge(before, after, plan(before), Respond(change))


def test_unchanged_distant_sentences_and_full_profiles_are_not_sent():
    before = '대상이다. 가까운 문장이다. 먼 문장이다. 비밀123이다.'
    after = before.replace('대상', '수정 대상')
    llm = Respond()
    judge(before, after, plan(before), llm)
    assert not {'before', 'after', 'profile', 'scores', 'korean_changes'} & llm.payload.keys()
    assert '비밀123' not in str(llm.payload)


def test_edit_extraction_covers_random_insert_delete_replace_with_whitespace_and_repetition():
    rng = random.Random(7)
    for _ in range(400):
        before = ''.join(rng.choice('가나 다.\n') for _ in range(30)) + ' 뒤다.'
        a, b = sorted(rng.sample(range(30), 2))
        after = before[:a] + ''.join(rng.choice('가나 다.\n') for _ in range(rng.randrange(8))) + before[b:]
        contract = plan(before, target=(a, b))
        edits = readable_edits(before, after, contract)
        reconstructed = before
        for edit in reversed(edits):
            x, y = edit['before_span']
            assert edit['before_text'] == before[x:y]
            assert edit['after_text'] == after[slice(*edit['after_span'])]
            reconstructed = reconstructed[:x] + edit['after_text'] + reconstructed[y:]
        assert reconstructed == after
