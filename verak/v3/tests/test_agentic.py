"""Agentic pilot contracts: graph, scopes, dispatch, context and export masking."""
from copy import deepcopy
import json

import pytest

from verak.v3.agentic import graph
from verak.v3.agentic.environment import AgenticEnv, PROMPTS, context
from verak.v3.agentic.runner import run
from verak.v3.agentic.export import select, export
from verak.v3.common import load_config
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v3.tests.test_environment import StubParagraphAnalysis, StubScore


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = ''.join('<' + m['role'] + '>' + m['content'] + '</turn>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(text.encode()) if tokenize else text


@pytest.fixture
def env():
    analysis = StubParagraphAnalysis()
    doc = Document([Paragraph('P4', [Unit('S9', '주장이다.', analysis.pieces('주장이다.')[0][1]),
                                      Unit('S3', '근거이다.', analysis.pieces('근거이다.')[0][1], ' ')]),
                    Paragraph('P2', [Unit('S8', '다른 말이다.', analysis.pieces('다른 말이다.')[0][1])])], ['', '\n'])
    ep = {'episode_id': 'synthetic', 'question': '의견은?', 'question_id': 'Q:fixture', 'genre': '논증', 'document': doc}
    discourse = {'sentence_edges': [dict(source='S1', target='Q', label='addresses'),
                                    dict(source='S2', target='S1', label='supports')], 'paragraph_edges': []}
    return AgenticEnv(ep, discourse, analysis=analysis, scorer=StubScore())


def act(env, role, name, **args):
    return env.step({'action': name, 'args': args}, role)


def test_graph_intersection_q_path_and_validation(env):
    g = env.graph()
    assert g['off_topic_candidate'] == ['S3']
    assert g['unsupported'] == []
    second = deepcopy(env.discourse)
    second['sentence_edges'].pop()
    kept, counts = graph.intersection(env.discourse, second)
    assert len(kept['sentence_edges']) == 1
    assert counts['sentence_edges']['union'] == 2
    graph.validate(env.discourse, ['S1', 'S2', 'S3'], ['P1', 'P2'])
    bad = deepcopy(env.discourse)
    bad['sentence_edges'].append(dict(source='S2', target='S3', label='example_of'))
    with pytest.raises(ValueError):
        graph.validate(bad, ['S1', 'S2', 'S3'], ['P1', 'P2'])
    kept, _ = graph.intersection(bad, env.discourse)
    graph.validate(kept, ['S1', 'S2', 'S3'], ['P1', 'P2'])


def test_preview_transaction_and_dynamic_graph_undo(env):
    original = env.document.snapshot()
    ids = deepcopy(env.sids)
    result = act(env, 'composition', 'PREVIEW', action={'action': 'INSERT', 'args': {'target': 'after:S1', 'new_text': '근거이다.'}})
    assert result['valid'] and env.document.snapshot() == original and env.sids == ids
    assert not env.undo and env.executor.serial == 0
    assert act(env, 'composition', 'DELETE', target='S1')['valid']
    assert env.graph()['dangling'][0]['source'] == 'S2'
    assert 'S2' in env.graph()['off_topic_candidate']
    act(env, 'composition', 'UNDO')
    assert env.document.snapshot() == original
    assert not env.graph()['dangling']
    act(env, 'composition', 'INSERT', target='after:S1', new_text='근거이다.')
    assert 'N1' in env.graph()['off_topic_candidate']
    assert not any(e['source'] == 'N1' for e in env.graph()['sentence_edges'])


def test_role_permissions_reports_and_budget(env):
    original = env.document.text
    assert not act(env, 'orchestrator', 'DELETE', target='S1')['valid']
    assert not act(env, 'cohesion', 'MOVE', target='S1', position='after:S2')['valid']
    assert not act(env, 'cohesion', 'EDIT', target='S1', new_text='')['valid']
    assert not act(env, 'cohesion', 'EDIT', target='S1', new_text='하나이다. 둘이다.')['valid']
    assert env.document.text == original
    for _ in range(2):
        assert act(env, 'orchestrator', 'SCORE')['valid']
    assert not act(env, 'orchestrator', 'SCORE')['valid']
    for _ in range(4):
        assert act(env, 'orchestrator', 'DELEGATE', agent='composition', task='확인')['valid']
    assert not act(env, 'orchestrator', 'DELEGATE', agent='cohesion', task='확인')['valid']
    assert not act(env, 'orchestrator', 'PLAN', text='가' * 301)['valid']
    assert act(env, 'cohesion', 'REPORT', status='blocked', summary='위치 변경 필요')['valid']
    assert env.latest_report['status'] == 'blocked' and not env.done


def test_scope_and_observation_question(env):
    with pytest.raises(ValueError):
        env.resolve_scope('P1')
    env.scope = {'P4'}
    assert not act(env, 'cohesion', 'EDIT', target='S3', new_text='다르다.')['valid']
    for role in PROMPTS:
        obs = env.observation(role, 12)
        assert obs.startswith('[문항] 의견은?') and '[문항 ID] Q:fixture' in obs
    assert 'S3 | 다른 말이다.' not in env.observation('cohesion', 12)


def test_tools_do_not_charge_write_steps_and_undo_stays_in_invocation(env):
    from verak.v3.agentic.rewards import editor_reward
    env.episode.update(source=env.document.clone(), records=[], corrupted_score={'mean': 5.})
    before = env.document.clone()
    act(env, 'orchestrator', 'SCORE')
    act(env, 'orchestrator', 'PLAN', text='점검')
    act(env, 'composition', 'QUERY', target='off_topic')
    act(env, 'composition', 'PREVIEW', action={'action': 'DELETE', 'args': {'target': 'S3'}})
    r = editor_reward(env, 'composition', before, set(env.sids), env.actions, load_config())
    assert r['R_step'] == 0 and r['R_over'] == 0 and r['R'] == 0
    act(env, 'composition', 'DELETE', target='S3')
    act(env, 'composition', 'UNDO')
    r = editor_reward(env, 'composition', before, set(env.sids), env.actions, load_config())
    assert r['R_step'] == 2 and r['R_over'] == 0 and r['R'] == -.02
    env.begin_editor({'task': '확인', 'scope': 'all'})
    assert not act(env, 'cohesion', 'UNDO')['valid']


def test_final_audit_is_measured_without_forcing_finish(env):
    act(env, 'orchestrator', 'AUDIT')
    act(env, 'composition', 'DELETE', target='S3')
    a = act(env, 'orchestrator', 'FINISH', summary='완료', needs_explanation=[])
    assert a['valid'] and env.done
    assert a['result']['last_audit']['text_hash'] != a['before_hash']


def test_combined_steps_exempt_only_read_tools_and_ledgers():
    from verak.v3.agentic.accounting import combined_steps, audit_reward
    names = ['SCORE', 'QUERY', 'AUDIT', 'PREVIEW', 'PLAN', 'PROGRESS', 'DELEGATE', 'EDIT', 'UNDO', 'REPORT', 'FINISH']
    actions = [{'action': name} for name in names]
    assert combined_steps(actions) == 5
    row = {'actions': actions, 'reward': {'combined': {'R': .88, 'R_step': 2, 'weighted_components': {'steps': -.02}},
                                         'orchestrator': {'R': .86, 'final_combined_R': .88}}}
    corrected = audit_reward(row)
    assert corrected['reward']['combined']['R'] == pytest.approx(.85)
    assert row['reward']['combined']['R_step'] == 2


def test_controller_only_follows_delegate_and_finish(env, tmp_path):
    class API:
        def __init__(self):
            self.outputs = iter([
                ('DELEGATE', {'agent': 'cohesion', 'task': '확인'}),
                ('REPORT', {'status': 'blocked', 'summary': '구조 수정 필요'}),
                ('DELEGATE', {'agent': 'composition', 'task': '확인'}),
                ('REPORT', {'status': 'done', 'summary': '유지'}),
                ('AUDIT', {}), ('FINISH', {'summary': '유지', 'needs_explanation': []})])
        def request(self, messages, **kwargs):
            name, args = next(self.outputs)
            return {'raw': json.dumps({'action': name, 'args': args}), 'cost': {}, 'usage': {}}
    result = run(env, API(), Tokenizer(), load_config(), tmp_path / 'events.jsonl')
    assert result['completed']
    assert [s['role'] for s in result['sequences']] == ['cohesion', 'composition']
    assert [a['action'] for a in result['actions']] == ['DELEGATE', 'REPORT', 'DELEGATE', 'REPORT', 'AUDIT', 'FINISH']


def test_mandatory_context_never_truncates(env):
    env.plan, env.progress = '계획', '진행'
    env.latest_report = {'summary': '필수 보고'}
    messages, stats = context(env, 'orchestrator', 2, Tokenizer(), [{'old': '가' * 10000}])
    assert stats['history_dropped'] == 1
    assert '계획' in messages[1]['content'] and '필수 보고' in messages[1]['content']
    messages, _ = context(env, 'orchestrator', 2, Tokenizer(), [{'action': 'AUDIT', 'result': {}}])
    assert messages[-1]['content'].startswith('[문항] 의견은?')
    env.document.units[0].text = '가' * 10000
    with pytest.raises(ValueError, match='Mandatory'):
        context(env, 'orchestrator', 2, Tokenizer(), [])


def test_export_best_attempt_threshold_terminal_and_mask(tmp_path):
    call = {'role': 'orchestrator', 'turn': 1, 'delegation': 0,
            'messages': [{'role': 'system', 'content': '규칙'}, {'role': 'user', 'content': '보고와 도구 결과'}],
            'raw': '{"action":"FINISH","args":{"summary":"끝","needs_explanation":[]}}'}
    row = {'episode_id': 'x', 'completed': True, 'reward': {'combined': {'R': .85}},
           'actions': [{'role': 'orchestrator', 'action': 'FINISH', 'valid': True}], 'sequences': [], 'calls': [call]}
    worse = deepcopy(row)
    worse['reward']['combined']['R'] = .81
    corpus = {'x': {'records': []}}
    selected, _ = select([worse, row], corpus)
    assert selected['x', 'orchestrator'][0] == .85
    stats = export([worse, row], corpus, Tokenizer(), tmp_path)
    assert stats['roles']['orchestrator']['samples'] == 1
    saved = json.loads((tmp_path / 'orchestrator.jsonl').read_text())
    start, end = saved['assistant_spans'][0]
    assert all(v == -100 for v in saved['labels'][:start])
    assert saved['labels'][start:end] == saved['input_ids'][start:end]
    assert saved['input'] == call['messages'] and saved['target'] == call['raw']


def test_actual_prompt_limits():
    from transformers import AutoTokenizer
    config = load_config()
    from pathlib import Path
    if not (Path(config['paths']['policy_base']) / 'tokenizer_config.json').exists():
        pytest.skip('Local policy tokenizer unavailable')
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    assert all(len(tokenizer.encode(p, add_special_tokens=False)) <= 400 for p in [*PROMPTS.values(), graph.PROMPT])


def test_action_schema_rejects_malformed_preview_and_other_role_writes():
    jsonschema = pytest.importorskip('jsonschema')
    from verak.v3.agentic.schemas import action_schema
    correct = {'action': 'PREVIEW', 'args': {'action': {'action': 'MOVE', 'args': {'target': 'S1', 'position': 'after:S2'}}}}
    jsonschema.validate(correct, action_schema('composition'))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'action': 'PREVIEW', 'args': {'action': 'MOVE', 'args': {'target': 'S1', 'position': 'after:S2'}}}, action_schema('composition'))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'action': 'DELETE', 'args': {'target': 'S1'}}, action_schema('orchestrator'))
