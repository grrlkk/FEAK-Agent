import json

import pytest

from verak.v5 import luna_simple as s


ROW = {'source_id': s.SOURCE_IDS[0], 'question': '질문', 'original': '나는 걷는다.',
       'scores': {'expected': [3] * 8}, 'feedback': 'PRIVATE_FEEDBACK', 'human_mean': 'PRIVATE_SCORE'}
PLAN = {'thesis_quote': '나는 걷는다.', 'priorities': [{'problem': '표현', 'instruction': '명확하게'}],
        'writer_notes': ['PRIVATE_WRITER_NOTE']}


def test_private_fields_and_writer_notes_never_reach_writer():
    for role in ('one_pass', 'plan', 'revision', 'korean'):
        messages = s.generation_messages(role, ROW, plan=PLAN, revised='나는 걷는다.')
        text = json.dumps(messages, ensure_ascii=False)
        assert 'PRIVATE_' not in text
        observation = json.loads(messages[1]['content'])
        assert not {'notices', 'phi', 'feedback', 'human_mean'} & set(observation)
    assert s.generation_messages('one_pass', ROW)[1] == s.generation_messages('plan', ROW)[1]


def test_three_roles_accept_first_outputs_without_quality_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'source', lambda _: ROW)
    outputs = [PLAN, {'essay': '원문 밖 주장도 담긴 후보.'}, {'essay': '한국어 역할이 낸 최종 후보.'}]
    seen = []

    class API:
        def __init__(self, *args, **kwargs): pass
        def request(self, messages, **kwargs):
            seen.append(messages)
            return {'raw': json.dumps(outputs[len(seen)-1], ensure_ascii=False),
                    'fingerprint': str(len(seen)), 'usage': {'input_tokens': 1, 'output_tokens': 1},
                    'confirmed_usd': 0, 'elapsed_s': 1}
        def close(self): pass

    result = s.episode(s.SOURCE_IDS[0], 'three_agent', api_factory=API, directory=tmp_path)
    assert len(seen) == 3
    assert result['revision'] == outputs[1]['essay']
    assert result['final'] == outputs[2]['essay']
    assert json.loads(seen[2][1]['content'])['현재 수정본'] == outputs[1]['essay']
    assert len(list(tmp_path.glob('episodes/three_agent/*/*_observation.json'))) == 3
    # Resume reads the durable result, without calling the provider again.
    assert s.episode(s.SOURCE_IDS[0], 'three_agent', api_factory=API, directory=tmp_path) == result
    assert len(seen) == 3


def test_malformed_action_is_saved_and_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'source', lambda _: ROW)
    seen = []

    class API:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args, **kwargs):
            seen.append(1)
            return {'raw': 'malformed', 'fingerprint': 'x'}
        def close(self): pass

    with pytest.raises(ValueError):
        s.episode(s.SOURCE_IDS[0], 'one_pass', api_factory=API, directory=tmp_path)
    assert len(seen) == 1
    assert len(list(tmp_path.glob('episodes/one_pass/*/one_pass_response.json'))) == 1
    assert not list(tmp_path.glob('episodes/one_pass/*/result.json'))


def test_literal_json_newline_recovery_keeps_saved_candidate_exactly():
    raw = '{"essay": "첫 문장.\n\n다음 문단."}'
    assert s.validate_action('revision', raw)['essay'] == '첫 문장.\n\n다음 문단.'


def test_judge_blindness_and_literal_evidence():
    messages = s.judge_messages(ROW, '후보 갑.', '후보 을.')
    observation = json.loads(messages[1]['content'])
    assert set(observation) == {'문항', '원문', 'left', 'right'}
    assert 'PRIVATE_' not in json.dumps(messages)
    item = {'improved': True, 'meaning_preserved': True, 'new_content': [], 'new_link_errors': [], 'reason': '근거'}
    value = {'left': item, 'right': item, 'preference': 'tie', 'reason': '동등'}
    assert s.safe(s.validate_judgment(json.dumps(value), {'left': '후보 갑.', 'right': '후보 을.'})['left'])
    item['new_content'] = [{'quote': '없는 문장', 'reason': '내용'}]
    with pytest.raises(ValueError, match='Nonliteral'):
        s.validate_judgment(json.dumps(value), {'left': '후보 갑.', 'right': '후보 을.'})


def test_preference_requires_two_orders_and_exposes_disagreement():
    pair = ['one_pass', 'three_agent']
    first = {'mapping': {'source_id': 'x', 'comparison': pair, 'left': pair[0], 'right': pair[1]},
             'judgment': {'preference': 'right'}}
    second = {'mapping': {'source_id': 'x', 'comparison': pair, 'left': pair[1], 'right': pair[0]},
              'judgment': {'preference': 'left'}}
    assert s.preference([first, second], 'x', pair) == 'three_agent'
    second['judgment']['preference'] = 'right'
    assert s.preference([first, second], 'x', pair) == 'disagree'
    with pytest.raises(ValueError): s.preference([first], 'x', pair)


def test_evaluation_sources_are_refused_before_open():
    with pytest.raises(ValueError): s.source('valid:1')


def test_evaluation_format_recovery_preserves_both_calls_and_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'HOME', tmp_path)
    monkeypatch.setattr(s, 'source', lambda _: ROW)
    monkeypatch.setattr(s, 'results', lambda _: (None, None, {'one_pass': '후보 갑.', 'three_agent': '후보 을.'}))
    item = {'improved': True, 'meaning_preserved': True, 'new_content': [], 'new_link_errors': [], 'reason': '근거'}
    valid = {'left': item, 'right': item, 'preference': 'tie', 'reason': '동등'}
    invalid = json.loads(json.dumps(valid))
    invalid['left']['new_link_errors'] = [{'quote': '없는 인용', 'type': 'other', 'reason': '잘못된 근거'}]
    seen = []

    class API:
        def __init__(self, *args, **kwargs): pass
        def request(self, messages, **kwargs):
            seen.append((messages, kwargs))
            return {'raw': json.dumps(invalid if len(seen) == 1 else valid), 'confirmed_usd': 1}
        def close(self): pass

    monkeypatch.setattr(s, 'PaidAPI', API)
    pair = ['one_pass', 'three_agent']
    assert s.judge_one('x', pair, 0) == valid
    assert seen[0][0] == seen[1][0]
    assert seen[0][1]['item_id'] != seen[1][1]['item_id']
    assert len(list(tmp_path.glob('judgments/*/response*.json'))) == 2
    assert len(list(tmp_path.glob('judgments/*/format_retry.json'))) == 1
    assert s.judge_one('x', pair, 0) == valid
    assert len(seen) == 2
