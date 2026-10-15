"""v2 tool transactions, hand-off, role isolation, and immutable v1 prompts."""
from copy import deepcopy

import pytest

from verak.v3.agent.runner import system_prompt as v1_prompt
from verak.v3.env import RevisionEnv
from verak.v3.tests.test_environment import action, setup_env
from verak.v3.v2_ops.environment import V2RevisionEnv
from verak.v3.v2_ops.prompts import system_prompt
from verak.v3.v2_ops.protocol import validity


def make_v2(setup):
    old, episode, analysis = setup
    config = deepcopy(old.config)
    config['method_version'] = 'v2'
    return V2RevisionEnv(config, analysis=analysis), episode, analysis


def test_v2_insert_empty_paragraph_split_handoff_and_undo(setup_env):
    env, episode, analysis = make_v2(setup_env)
    observation = env.reset(episode)
    assert '[역할] 글 수정 에이전트 (GLOBAL)' in observation
    before = env.final_text()
    _, _, info = env.step(action('SPLIT', sentence_id='S1', text_1='첫 주장이다.', text_2='주장을 설명한다.'))
    assert info['action']['valid'] and info['action']['action'] == 'SPLIT'
    assert len(env.document.units) == 5
    assert analysis.refreshed[-1] == {'P3'}
    env.step(action('UNDO'))
    assert env.final_text() == before
    env.step(action('EDIT', target='S3', new_text=''))
    env.step(action('EDIT', target='S4', new_text=''))
    assert not env.document.paragraphs[1].units
    _, _, info = env.step(action('INSERT', position='before:P2', text='결론을 쓴다.'))
    assert info['action']['valid'] and info['action']['action'] == 'INSERT'
    assert len(env.document.paragraphs[1].units) == 1
    _, _, info = env.step(action('INSERT', position='after:S1', text='이를 보충한다.'))
    assert info['action']['valid'] and info['action']['cohesion_changes']
    observation, _, info = env.step(action('STOP', summary='문장 구조를 정리했다.'))
    assert info['handoff'] and env.handoff['cohesion_changes']
    assert {'INSERT', 'SPLIT'} <= {a['action'] for a in env.handoff['actions']}
    assert '[GLOBAL 인계:' in observation
    text = env.final_text()
    _, _, info = env.step(action('SPLIT', sentence_id='S1', text_1='첫 주장이다.', text_2='설명이다.'))
    assert info['action']['error_code'] == 'role_forbidden' and env.final_text() == text
    _, _, info = env.step(action('INSERT', position='before:P2', text='추가한다.'))
    assert info['action']['error_code'] == 'role_forbidden' and env.final_text() == text


def test_v2_gates_and_v1_still_rejects_new_tools(setup_env):
    old, episode, analysis = setup_env
    with pytest.raises(ValueError, match='config v2'):
        V2RevisionEnv(old.config, analysis=analysis)
    old.reset(episode)
    before = old.final_text()
    _, _, info = old.step(action('INSERT', position='before:S1', text='삽입한다.'))
    assert info['action']['error_code'] == 'unknown_action' and old.final_text() == before
    assert system_prompt('korean') == v1_prompt('korean')
    assert '글에 없는 내용은 만들지 않는다' in system_prompt('global')
    assert system_prompt('global').count('\nINSERT:') == system_prompt('global').count('\nSPLIT:') == 1
    assert validity(action('INSERT', position='before:P1', text='삽입한다.')) == (True, True)
    assert validity('{"thought":"","action":[],"args":{}}') == (True, False)


def test_v2_failed_split_is_atomic_and_budget_handoff_keeps_new_action(setup_env):
    env, episode, _ = make_v2(setup_env)
    env.config['env']['global']['max_steps'] = 2
    env.reset(episode)
    before = env.final_text()
    _, _, info = env.step(action('SPLIT', sentence_id='S1', text_1='첫째다. 둘째다.', text_2='셋째다.'))
    assert not info['action']['valid'] and env.final_text() == before
    _, _, info = env.step(action('SPLIT', sentence_id='S1', text_1='첫째다.', text_2='둘째다.'))
    assert info['handoff'] and info['action']['valid']
    assert env.handoff['actions'][-1]['action'] == 'SPLIT'
    assert env.handoff['actions'][-1]['args']['sentence_id'] == 'S1'


def test_v2_runner_executes_split_and_handoff_with_no_scorer(setup_env):
    from verak.v3.v2_ops.runner import run_episode
    from verak.v3.v2_ops.judges import LABEL_PROMPT, QC_PROMPT, RECOVERY_PROMPT
    env, episode, _ = make_v2(setup_env)
    class Backend:
        name, model = 'teacher', 'synthetic'
        def __init__(self):
            self.messages = []
        def generate(self, messages, *, episode_id, role, turn):
            self.messages.extend(messages)
            raw = (action('SPLIT', sentence_id='S1', text_1='첫 주장을 쓴다.', text_2='이를 설명한다.')
                   if role == 'global' and turn == '1:0' else action('STOP', summary='검토했다.'))
            return {'raw': raw, 'role': role, 'turn': turn, 'model': self.model}
    backend = Backend()
    result = run_episode(env, episode, backend)
    assert result['completed'] and result['reward'] is None and not result['score_calls']
    assert result['termination'] == {'global': 'STOP', 'korean': 'STOP'}
    assert result['calls'][0]['valid_json_action']
    assert result['handoff']['actions'][0]['action'] == 'SPLIT'
    assert all(m['content'] not in (LABEL_PROMPT, QC_PROMPT, RECOVERY_PROMPT) for m in backend.messages)
