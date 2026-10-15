"""v4.5 inference fixes and both completion definitions, without model calls."""
from copy import deepcopy
import json

import pytest

from verak.v4.content_env import dumps
from verak.v4.environment_v45 import V45Environment, match_span, CONTEXT, OUTPUT
from verak.v4.maps_v45 import filter_map, validate, protected_map, EmptyMapError
from verak.v4.smoke45_infer import completion, run_case, summarize, LocalBackend, CONDITIONS
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row


@pytest.fixture(autouse=True)
def no_contract_io(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **k: {})


def environment(text, role='revision'):
    env = V45Environment(row(text), FakeBoundaryAnalysis())
    if role == 'revision':
        env.delegate([{'location': ['P1'], 'action': 'EDIT', 'instruction': '검토한다.', 'needs_search': 'no'}], 1)
    else: env.start_korean([])
    return env


def edit(env, old, new, role='revision'):
    return env.step(dumps({'action': 'EDIT', 'sentence': 'S1', 'old': old, 'new': new}), role)


def test_exact_first_normalized_offsets_and_ambiguity():
    span, data = match_span('가  나 그리고 가 나', '가 나')
    assert data['mode'] == 'exact' and span == (9, 12)
    text = '앞 가\t\u00a0 나 뒤'
    span, data = match_span(text, '가 나')
    assert data['matched'] and text[slice(*span)] == '가\t\u00a0 나'
    for text, old in [('가  나 가\t나', '가 나'), ('가 나 가 나', '가 나'), ('aaaaa', 'aaa'), ('내용 없음', '내용이')]:
        assert match_span(text, old)[0] is None
    assert match_span('내용이 있음', '내용 이')[0] is None  # no removal of word spaces
    assert match_span('가  나', ' ')[0] is None  # no whitespace-only fuzzy fallback


def test_fallback_changes_only_original_span_keeps_raw_action_and_undo():
    env = environment('앞의 가\t\u00a0 나 뒤는 유지한다.')
    original = env.document.text
    result = edit(env, '가 나', '다 라')
    assert result['valid'] and result['span_match']['matched']
    assert result['value']['old'] == '가 나'
    assert env.document.text == '앞의 다 라 뒤는 유지한다.'
    assert env.step(dumps({'action': 'UNDO'}), 'revision')['valid']
    assert env.document.text == original


def test_fallback_retains_mask_scope_boundary_and_noop_guards():
    env = environment('#@이름#은  학생이다.', 'korean')
    old = '#@이름#은 학생이다.'
    assert not edit(env, old, '#@다른이름#은 학생이다.', 'korean')['valid']
    assert not edit(env, old, '#@이름#은 학생이다. 설명한다.', 'korean')['valid']
    result = edit(env, old, '#@이름#은  학생이다.', 'korean')
    assert not result['valid'] and '고칠 것이 없으면 STOP 한다.' in result['error']
    assert env.controller_stop and env.closed
    env = environment('가  나다. 다른 문장이다.')
    env.scope = {'S1a'}
    assert not edit(env, '가 나다', '다 라다')['valid']
    assert not env.step('{"action":"UNDO"}{"action":"UNDO"}', 'revision')['valid']


def test_fallback_revision_split_not_korean_split():
    env = environment('이것은  길고 복잡하다.')
    result = edit(env, '이것은 길고 복잡하다.', '이것은 길다. 복잡하다.')
    assert result['valid'] and len(result['created_sids']) == 1
    assert result['split_notice']['kind'] == 'sentence_split'
    assert len(env.document.units) == 2


def paragraphs(n=2):
    return [{'id': f'P{i}', 'sentences': [{'id': f'S{i}', 'text': '문장이다.'},
              {'id': f'S{i}a', 'text': '다음이다.'}]} for i in range(1, n+1)]


def graph(n=2):
    return {'genre': 'argumentative', 'paragraph_roles': [
        {'paragraph': f'P{i}', 'role': 'body', 'key_sentence': f'S{i}'} for i in range(1, n+1)],
        'sentence_relations': [], 'paragraph_relations': [], 'off_topic': []}


def edge(source, target, kind='support'):
    return {'source': source, 'target': target, 'type': kind}


def test_map_salvages_wrong_key_role_unknown_fields_and_bad_edges():
    value = graph()
    value['paragraph_roles'][0].update(key_sentence='S2', role='invalid', detail='ignore')
    value['paragraph_roles'].append({'paragraph': 'P99', 'role': 'body', 'key_sentence': 'S99'})
    value['off_topic'] = ['S2a', 'S99', 'S2a']
    value['sentence_relations'] = [edge('S1a', 'S1'), edge('S1a', 'S2'), edge('S1a', 'S2', 'example'), edge('S99', 'S2')]
    before = deepcopy(value)
    result, info = validate(dumps({'action': 'MAP', 'map': value}), paragraphs())
    clean = result['map']
    assert value == before
    assert clean['paragraph_roles'][0] == {'paragraph': 'P1', 'role': None, 'key_sentence': None}
    assert len(clean['sentence_relations']) == 2 and clean['off_topic'] == ['S2a']
    assert info['dropped_by_reason'] == {'outgoing_support_limit': 1, 'unknown_source_ID': 1}
    assert info['dropped_fields_by_reason']['key_not_in_paragraph'] == 1
    assert info['dropped_fields_by_reason']['role_not_allowed_for_genre'] == 1
    protected_map(clean, paragraphs())


def test_map_edge_filter_identical_to_training_and_long_essay_rule():
    from verak.v4.prep3_maps import filter_edges
    value = graph(7)
    value['sentence_relations'] = [edge('S1', 'S2'), edge('S1a', 'S2a'), edge('S2a', 'S2'),
                                 edge('S2a', 'S2', 'example'), edge('S1', 'S2')]
    clean, info = filter_map(value, paragraphs(7))
    expected = filter_edges(value, {'question': '', 'genre': 'argumentative', 'paragraphs': paragraphs(7)})
    assert clean == expected['map'] and info['dropped_edges'] == expected['dropped_edges']
    assert info['dropped_by_reason']['long_cross_paragraph_missing_main_or_key'] == 1
    assert info['dropped_by_reason']['duplicate_relation'] == 1


def test_map_keeps_valid_edges_despite_missing_or_broken_metadata():
    value = {'genre': [], 'paragraph_roles': 'bad', 'off_topic': [None, {}],
             'sentence_relations': [edge('S1', 'Q', 'main')], 'paragraph_relations': None}
    clean, info = filter_map(value, paragraphs())
    assert clean['genre'] is None and len(clean['sentence_relations']) == 1
    assert all(p['role'] is None for p in clean['paragraph_roles'])
    result, protection = protected_map(clean, paragraphs())
    assert result['genre'] is None and protection['protection']['seeds'] == ['S1']
    with pytest.raises(EmptyMapError) as exc:
        filter_map({'genre': 'argumentative', 'sentence_relations': [edge('S99', 'S1')]}, paragraphs())
    assert exc.value.diagnostics['dropped_edges'][0]['reason'] == 'unknown_source_ID'
    clean, _ = filter_map({'paragraph_roles': [{'paragraph': 'P1', 'role': None, 'key_sentence': None}]}, paragraphs())
    assert len(clean['paragraph_roles']) == 2  # supplied node accepted, missing node unknown


def test_model_blocked_with_reason_separates_both_completion_measures():
    case = {'status': 'completed', 'orchestrator_ending': 'FINISH', 'stages': [
        {'role': 'korean', 'ending': 'STOP', 'stop_status': 'blocked', 'report': {'summary': '근거가 부족하다.', 'issues': []}}]}
    assert completion(case)['no_protocol_error'] and not completion(case)['all_stages_STOP_done']
    case['stages'][0]['report']['summary'] = ' '
    assert not completion(case)['no_protocol_error']
    case['stages'][0]['report']['issues'] = ['범위 안에서 처리 불가']
    assert completion(case)['no_protocol_error']
    case['stages'][0]['stop_status'] = 'done'
    assert completion(case)['all_stages_STOP_done']
    for field, value in [('controller_events', [{}]), ('context_limits', [{}]), ('execution_failures', [{}]),
                         ('orchestrator_ending', 'step_limit'), ('status', 'error')]:
        assert not completion({**case, field: value})['no_protocol_error']
    for ending in ('step_limit', 'controller_STOP_blocked', 'invalid_format', 'prefix_context_limit'):
        other = deepcopy(case); other['stages'][0]['ending'] = ending
        assert not completion(other)['no_protocol_error']


def test_integrated_map_salvage_and_blocked_are_valid_protocol(tmp_path):
    class Tokens:
        def apply_chat_template(self, *a, **k): return [1]
        def encode(self, *a, **k): return [1]
    class Backend:
        number = 0
        def generate(self, messages, role):
            if role == 'orchestrator':
                self.number += 1
                actions = [{'action': 'SCORE'}, {'action': 'MAP', 'map': {
                    **graph(1), 'paragraph_roles': [{'paragraph': 'P1', 'role': 'body', 'key_sentence': 'S99'}]}},
                    {'action': 'FINISH', 'summary': '확인했다.', 'writer_notes': []}]
                action = actions[self.number-1]
            else: action = {'action': 'STOP', 'status': 'blocked', 'summary': '범위 내에서 더 고칠 부분을 찾지 못했다.', 'issues': []}
            return {'raw': dumps(action), 'usage': {'prompt_tokens': 1}, 'finish_reason': 'stop'}
    source = row('문장이다.'); source['genre'] = 'argumentative'
    result = run_case(source, case_id='fixture', output=tmp_path, backend=Backend(), tok=Tokens(),
        analysis=FakeBoundaryAnalysis(), score=lambda *a: {'expected': [5]*8, 'integers': [5]*8})
    assert result['completion']['no_protocol_error'] and not result['completion']['all_stages_STOP_done']
    stats = summarize([result], 1)
    assert stats['no_protocol_error'] == 1 and stats['all_stages_STOP_done'] == 0
    assert stats['roles']['korean']['model_STOP_blocked'] == 1
    assert stats['map_validation']['dropped_fields_by_reason']['key_not_in_paragraph'] == 1


def test_paired_adapters_fixed_prompts_and_increased_orchestrator_context():
    from verak.v4 import orchestrator_v44 as orch
    from verak.v4.common import sha_text
    assert sha_text(orch.PROMPT) == 'c7f608ff13a0f7968c21a894bf9b4ccea18a86ab49efb88c7e3548ecada9d6aa'
    assert CONTEXT['orchestrator'] == 16384 and OUTPUT['orchestrator'] == 4096
    for condition, epochs in CONDITIONS.items():
        client = LocalBackend(condition)
        try:
            assert client.adapters == {r: f'v45-{r}-e{e}' for r, e in epochs.items()}
            assert client.adapters['korean'] == 'v45-korean-e2'
            assert client.adapters['orchestrator'] == 'v45-orchestrator-e2'
        finally: client.close()
