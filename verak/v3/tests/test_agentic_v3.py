"""Behavioral regression tests for the bounded agentic v3 experiment."""
from copy import deepcopy
import json

import pytest

from verak.v3.agentic import graph
from verak.v3.agentic.data import config_for
from verak.v3.agentic.environment import AgenticEnv, prompts_for, named_sentence_ids
from verak.v3.agentic.runner import run
from verak.v3.tests.test_agentic import env, act, Tokenizer


@pytest.fixture
def v3(env):
    env.version = 3
    env.discourse['off_topic'] = ['S3']
    env.initial_graph = env.graph()
    return env


def preview_delete(env, target):
    return act(env, 'composition', 'PREVIEW', action={'action': 'DELETE', 'args': {'target': target}})


def test_explicit_off_topic_intersection_not_graph_reachability(v3):
    second = deepcopy(v3.discourse)
    second['off_topic'] = ['S2', 'S3']
    second['sentence_edges'] = []
    kept, counts = graph.intersection(v3.discourse, second)
    assert kept == {'sentence_edges': [], 'paragraph_edges': [], 'off_topic': ['S3']}
    assert counts['off_topic']['union'] == 2
    v3.discourse = kept
    assert v3.graph()['off_topic_candidate'] == ['S3']
    assert 'unsupported' not in act(v3, 'orchestrator', 'AUDIT')['result']
    act(v3, 'composition', 'INSERT', target='after:S1', new_text='설명이다.')
    assert 'N1' not in v3.graph()['off_topic_candidate']
    with pytest.raises(ValueError):
        graph.intersection(kept, {'sentence_edges': [], 'paragraph_edges': []})


def test_delete_task_id_exact_preview_current_state_and_cap(v3):
    v3.begin_editor({'task': 'S1, S2만 삭제', 'scope': 'all'})
    assert not preview_delete(v3, 'S3')['valid']
    assert not act(v3, 'composition', 'DELETE', target='S1')['valid']
    assert preview_delete(v3, 'S1')['valid']
    assert act(v3, 'composition', 'MOVE', target='S2', position='before:S1')['valid']
    assert not act(v3, 'composition', 'DELETE', target='S1')['valid']
    assert preview_delete(v3, 'S1')['valid']
    act(v3, 'composition', 'QUERY', target='off_topic')
    assert act(v3, 'composition', 'DELETE', target='S1')['valid']
    assert act(v3, 'composition', 'UNDO')['valid']
    assert preview_delete(v3, 'S2')['valid']
    assert act(v3, 'composition', 'DELETE', target='S2')['valid']
    act(v3, 'composition', 'UNDO')
    assert not act(v3, 'composition', 'DELETE', target='S1')['valid']
    v3.begin_editor({'task': 'S1 삭제', 'scope': 'all'})
    assert not act(v3, 'composition', 'DELETE', target='S1')['valid']
    assert preview_delete(v3, 'S1')['valid']
    assert act(v3, 'composition', 'DELETE', target='S1')['valid']
    assert named_sentence_ids('S1-S3, XS4, S11, N2를') == {'S1', 'S3', 'S11', 'N2'}


def test_score_first_only_and_three_delegations(v3):
    assert act(v3, 'orchestrator', 'SCORE')['valid']
    assert not act(v3, 'orchestrator', 'SCORE')['valid']
    assert not act(v3, 'orchestrator', 'DELEGATE', agent='cohesion', task='확인')['valid']
    for _ in range(3):
        assert act(v3, 'orchestrator', 'DELEGATE', agent='composition', task='확인')['valid']
    assert not act(v3, 'orchestrator', 'DELEGATE', agent='composition', task='확인')['valid']


def test_late_first_score_rejected_and_cohesion_needs_scope_notice(v3):
    act(v3, 'orchestrator', 'AUDIT')
    assert not act(v3, 'orchestrator', 'SCORE')['valid']
    v3.notices = [{'sid': 'S2', 'type': 'CONJ', 'message': '선행 문장 변경'}]
    value = act(v3, 'orchestrator', 'DELEGATE', agent='cohesion', task='S2 확인')
    assert value['valid'] and value['result']['disturbed_markers'] == v3.notices
    assert v3.disturbed_in_scope({'P2'}) == []


@pytest.mark.parametrize('report_at_notice', [False, True])
def test_final_notice_and_auto_report(v3, tmp_path, report_at_notice):
    class API:
        def request(self, messages, **kwargs):
            obs = messages[-1]['content']
            if '[역할] orchestrator' in obs:
                value = ({'action': 'FINISH', 'args': {'summary': '끝', 'needs_explanation': []}}
                         if v3.delegations else {'action': 'DELEGATE', 'args': {'agent': 'composition', 'task': '확인'}})
            elif report_at_notice and '[최종 알림]' in obs:
                value = {'action': 'REPORT', 'args': {'status': 'done', 'summary': '끝'}}
            else:
                value = {'action': 'QUERY', 'args': {'target': 'off_topic'}}
            return {'raw': json.dumps(value), 'cost': {}, 'usage': {}}
    result = run(v3, API(), Tokenizer(), config_for(version=3), tmp_path / 'events.jsonl')
    assert result['completed']
    sequence = result['sequences'][0]
    assert sequence['final_notice_sent']
    assert sequence['auto_report'] is not report_at_notice
    assert sequence['terminal'] == ('REPORT' if report_at_notice else 'AUTO_REPORT')
    assert sequence['report']['status'] == ('done' if report_at_notice else 'done (budget)')
    assert sequence['report']['origin'] == ('agent' if report_at_notice else 'controller')
    editor_calls = [c for c in result['calls'] if c['role'] == 'composition']
    assert len(editor_calls) == (15 if report_at_notice else 16)
    assert '[최종 알림]' not in editor_calls[13]['messages'][-1]['content']
    assert '[최종 알림]' in editor_calls[14]['messages'][-1]['content']


def test_v3_output_budget_and_prompt_limits():
    from transformers import AutoTokenizer
    config = config_for(version=3)
    assert config['agentic_pilot']['max_cost_usd'] == 7.
    assert config['paths']['agentic_pilot_output'].name == 'agentic_pilot_v3_protected'
    assert config_for()['paths']['agentic_pilot_output'].name == 'agentic_pilot'
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    assert all(len(tokenizer.encode(p, add_special_tokens=False)) <= 400
               for p in [*prompts_for(3).values(), graph.PROMPT_V3])
