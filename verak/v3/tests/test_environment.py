"""Synthetic action/protocol tests; live smoke uses only Bareun."""
from copy import deepcopy
from dataclasses import replace
import json
import re
from types import SimpleNamespace

import pytest

from verak.src.schemas import Token, Sentence, Profile
from verak.v3.common import load_config
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v3.env import RevisionEnv
from verak.v3.env.analysis import ParagraphAnalyzer
from verak.v3.env.protocol import ActionParseError, parse_action
from verak.v3.agent.runner import run_episode, system_prompt
from verak.v3.agent.backends import TeacherBackend


def action(name, **args):
    return json.dumps({'thought': '확인', 'action': name, 'args': args}, ensure_ascii=False)


class StubParagraphAnalysis:
    def __init__(self):
        self.refreshed = []

    def pieces(self, text):
        if '\n' in text:
            raise ValueError('one paragraph')
        pieces = []
        end = 0
        for m in re.finditer(r'[^.!?]+[.!?]?', text):
            sentence = m.group().strip()
            if sentence:
                start = m.start()+len(m.group())-len(m.group().lstrip())
                tokens = [Token('ㄴ다', 'EF', max(0, len(sentence)-3), len(sentence)-1)]
                pieces.append((sentence, tokens, text[end:start]))
                end = m.end()
        if not pieces:
            raise ValueError('empty')
        return pieces

    def refresh(self, doc, pids):
        self.refreshed.append(set(pids))


class StubScore:
    def __init__(self):
        self.calls = []

    def score(self, question, text):
        self.calls.append(text)
        q = 5.+text.count('새로운')*.1
        return {'expected': [q]*8, 'mean': q, 'cache_hit': False}


@pytest.fixture
def setup_env(tmp_path):
    config = load_config()
    config['paths']['phase5_output'] = tmp_path
    analysis = StubParagraphAnalysis()
    paragraphs = []
    # Nonsequential private IDs must not expose original positions at reset.
    for pid, units in [('P3', [('S8', '첫 주장을 쓴다.'), ('S4', '이를 설명한다.')]),
                       ('P1', [('S5', '다른 근거를 쓴다.'), ('S1', '결론을 쓴다.')])]:
        paragraphs.append(Paragraph(pid, [Unit(sid, text, analysis.pieces(text)[0][1], '' if i == 0 else ' ')
                                         for i, (sid, text) in enumerate(units)]))
    doc = Document(paragraphs, ['', '\n'])
    episode = {'episode_id': 'synthetic', 'question': '의견을 쓰시오.', 'genre': '논증', 'document': doc}
    env = RevisionEnv(config, analysis=analysis, scorer=StubScore())
    return env, episode, analysis


def test_reset_aliases_roles_and_no_reward_for_real(setup_env):
    env, episode, analysis = setup_env
    obs = env.reset(episode)
    assert 'S1 | 첫 주장을 쓴다.' in obs and 'S2 | 이를 설명한다.' in obs
    assert env.sentence_ids == {'S8': 'S1', 'S4': 'S2', 'S5': 'S3', 'S1': 'S4'}
    assert env.paragraph_ids == {'P3': 'P1', 'P1': 'P2'}
    assert '근거타당성' in obs and 'steps_left' in obs
    assert env.reward() is None and not env.scorer.calls
    assert analysis.refreshed == [{'P1', 'P3'}]


def test_handoff_stable_ids_and_role_restrictions(setup_env):
    env, ep, _ = setup_env
    env.reset(ep)
    original = env.final_text()
    _, done, info = env.step(action('EDIT', target='S1:첫', new_text='다음'))
    assert not done and info['action']['error_code'] == 'role_forbidden'
    assert env.final_text() == original and env.steps['global'] == 1
    env.step(action('MOVE', target='P2', position='before:P1'))
    obs, done, info = env.step(action('STOP', summary='순서 변경'))
    assert info['handoff'] and not done and env.role == 'korean'
    assert env.handoff['current_text'] == env.final_text()
    assert len(env.handoff['actions']) == 3
    assert env.handoff['cohesion_changes']
    assert '[GLOBAL 인계' in obs and 'S1 | 첫 주장을 쓴다.' in obs
    assert all(word not in obs for word in ('source_text', 'recovery_target', 'q_source'))
    _, _, info = env.step(action('UNDO'))
    assert info['action']['error_code'] == 'undo_empty'
    _, _, info = env.step(action('MOVE', target='S1', position='after:S2'))
    assert info['action']['error_code'] == 'role_forbidden'
    _, _, info = env.step(action('EDIT', target='S1:첫', new_text='처음'))
    assert info['action']['valid'] and env.role == 'korean'


def test_add_delete_undo_and_sentence_splits(setup_env):
    env, ep, analysis = setup_env
    env.reset(ep)
    original = env.final_text()
    env.step(action('EDIT', target='after:S1', new_text='새로운 근거를 쓴다. 또 설명한다.'))
    assert {'N1', 'N2'} <= set(env.sentence_ids.values())
    assert analysis.refreshed[-1] == {'P3'}
    env.step(action('UNDO'))
    assert env.final_text() == original
    env.step(action('EDIT', target='S2', new_text=''))
    assert '이를 설명한다.' not in env.final_text()
    env.step(action('UNDO'))
    assert env.final_text() == original
    env.mode = 'single'
    env.reset(ep)
    _, _, info = env.step(action('EDIT', target='S1', new_text='하나를 쓴다. 둘을 쓴다.'))
    assert info['action']['valid'] and 'S1a' in env.sentence_ids.values()


def test_sentence_range_move_preserves_ids_text_and_can_undo(setup_env):
    env, ep, analysis = setup_env
    env.reset(ep)
    original = env.final_text()
    aliases = deepcopy(env.sentence_ids)
    _, _, info = env.step(action('MOVE', target='S1-S2', position='after:S3'))
    assert info['action']['valid']
    assert [env.sentence_ids[u.sid] for u in env.document.units] == ['S3', 'S1', 'S2', 'S4']
    assert env.sentence_ids == aliases
    assert analysis.refreshed[-1] == {'P3', 'P1'}
    assert all(u.text in original for u in env.document.units)
    env.step(action('UNDO'))
    assert env.final_text() == original


def test_sentence_move_to_paragraph_and_no_id_reuse_after_undo(setup_env):
    env, ep, _ = setup_env
    env.reset(ep)
    _, _, info = env.step(action('MOVE', target='S1', position='after:P2'))
    assert info['action']['valid']
    assert [env.sentence_ids[u.sid] for u in env.document.paragraphs[-1].units] == ['S3', 'S4', 'S1']
    env.step(action('EDIT', target='before:S1', new_text='새로운 설명이다.'))
    env.step(action('UNDO'))
    env.step(action('EDIT', target='before:S1', new_text='다른 설명이다.'))
    current = [env.sentence_ids[u.sid] for u in env.document.units]
    assert 'N2' in current and 'N1' not in current


@pytest.mark.parametrize('name,args,code',[
    ('EDIT', {'target': 'S99', 'new_text': ''}, 'unknown_id'),
    ('MOVE', {'target': 'P1', 'position': 'before:P1'}, 'move_self'),
    ('MOVE', {'target': 'S2-S3', 'position': 'after:S4'}, 'range'),
    ('UNDO', {}, 'undo_empty')])
def test_invalid_actions_are_atomic_and_cost_one_step(setup_env, name, args, code):
    env, ep, _ = setup_env
    env.reset(ep)
    text = env.final_text()
    _, _, info = env.step(action(name, **args))
    assert info['action']['error_code'] == code
    assert env.final_text() == text and env.steps['global'] == 1


def test_check_stage_baselines_and_budget_handoff(setup_env):
    env, ep, _ = setup_env
    env.reset(ep)
    env.step(action('EDIT', target='after:S1', new_text='새로운 설명을 쓴다.'))
    _, _, info = env.step(action('CHECK'))
    check = info['action']['check']
    assert check['Q'] == 5.1 and check['stage_start_Q'] == 5
    assert check['delta_last_CHECK'] is None and check['noise_floor'] == .21266690
    _, _, info = env.step(action('CHECK'))
    assert info['handoff'] and env.termination['global'] == 'max_checks'
    _, _, info = env.step(action('CHECK'))
    assert info['action']['check']['delta_stage_start'] == 0
    assert info['action']['check']['stage_start_Q'] == 5.1
    env.step(action('CHECK'))
    assert env.done and env.termination['korean'] == 'max_checks'


def test_steps_and_consecutive_errors(setup_env):
    env, ep, _ = setup_env
    env.config['env']['global']['max_steps'] = 1
    env.reset(ep)
    env.step(action('UNDO'))
    assert env.role == 'korean' and env.termination['global'] == 'max_steps'
    for _ in range(3):
        env.step(action('EDIT', target='after:S1', new_text='추가한다.'))
    assert env.done and env.termination['korean'] == 'errors'
    assert env.steps['korean'] == 3


def test_korean_cannot_add_delete_split_and_marker_edits(setup_env):
    env, ep, _ = setup_env
    ep['document'].units[0].text = '#@이름#이 글을 쓴다.'
    env.reset(ep)
    env.step(action('STOP', summary='인계'))
    _, _, info = env.step(action('EDIT', target='S1:이름', new_text='성명'))
    assert info['action']['error_code'] == 'anonymization'
    _, _, info = env.step(action('EDIT', target='S1', new_text='#@이름#이 쓴다. 더 쓴다.'))
    assert info['action']['error_code'] == 'role_forbidden'
    _, _, info = env.step(action('EDIT', target='S1:글', new_text='문장'))
    assert info['action']['valid'] and '#@이름#' in env.final_text()


@pytest.mark.parametrize('raw', ['```json\n{}\n```', '{}{}', '[]', '{"a":1,"a":2}', '{"a":NaN}'])
def test_one_strict_json_object(raw):
    with pytest.raises(ActionParseError):
        parse_action(raw)


class StubBackend:
    name = 'teacher'
    model = 'stub'
    context_limit = 250000

    def __init__(self, outputs):
        self.outputs, self.sent = iter(outputs), []

    def generate(self, messages, **context):
        self.sent.append(deepcopy(messages))
        return {'raw': next(self.outputs), 'cost': {}, 'messages': deepcopy(messages), **context}


def test_runner_parse_retry_once_and_prefix_unchanged(setup_env):
    env, ep, _ = setup_env
    env.mode = 'single'
    backend = StubBackend(['{bad', action('CHECK'), action('STOP', summary='완료')])
    result = run_episode(env, ep, backend)
    assert result['completed'] and result['steps']['single'] == 2
    assert result['actions_by_role']['single'][0]['parse_retry_used']
    assert len(result['calls']) == 3
    for earlier, later in zip(backend.sent, backend.sent[1:]):
        assert earlier == later[:len(earlier)]
    assert result['reward'] is None


def test_runner_two_failed_parses_count_one_error(setup_env):
    env, ep, _ = setup_env
    env.mode = 'single'
    result = run_episode(env, ep, StubBackend(['bad', 'bad', action('STOP', summary='완료')]))
    assert result['completed'] and result['steps']['single'] == 2
    assert result['actions_by_role']['single'][0]['error_code'] == 'parse_error'


def test_role_prompts_and_history_are_separate(setup_env):
    env, ep, _ = setup_env
    result = run_episode(env, ep, StubBackend([action('STOP', summary='인계'), action('STOP', summary='완료')]))
    assert result['completed']
    assert set(result['messages_by_role']) == {'global', 'korean'}
    assert result['messages_by_role']['global'][0]['content'] == system_prompt('global')
    assert result['messages_by_role']['korean'][0]['content'] == system_prompt('korean')
    assert '[GLOBAL 인계' in result['messages_by_role']['korean'][1]['content']
    assert all(word not in system_prompt(r) for r in ('global', 'korean', 'single')
               for word in ('corruption', 'recovery_target', '숨겨진 정답'))


def test_teacher_shared_call_budget_cache_and_cost(tmp_path):
    config = load_config()
    class Adapter:
        def __init__(self, cfg): self.cfg = cfg
        def _load(self): pass
        def close(self): pass
        def complete(self, messages):
            return {'status': 'completed', 'raw': action('STOP', summary='끝'), 'usage': {
                'input_tokens': 200, 'output_tokens': 30, 'input_tokens_details': {'cached_tokens': 100},
                'output_tokens_details': {'reasoning_tokens': 10}}}
    backend = TeacherBackend(config, tmp_path, max_api_calls=1, adapter_factory=Adapter)
    messages = [{'role': 'system', 'content': '규칙'}, {'role': 'user', 'content': '글'}]
    first = backend.generate(messages, episode_id='x', role='single', turn='1')
    again = backend.generate(messages, episode_id='x', role='single', turn='1')
    assert again['replayed'] and backend.budget.used == 1
    assert first['cost']['cache_read'] == 100
    from feak_tc.runtime.openai import CallBudgetExceeded
    with pytest.raises(CallBudgetExceeded):
        backend.generate(messages, episode_id='y', role='single', turn='1')


def test_paragraph_cache_uses_neighbor_hash_and_refreshes_only_selected(tmp_path):
    class Analyzer:
        def __init__(self): self.cache, self.calls = {}, []
        def profile(self, text):
            self.calls.append(text)
            return Profile([Sentence('1.1', 1, 0, len(text), text, [])], [], 'bareun', 'fixture')
    stub = Analyzer()
    analyzer = ParagraphAnalyzer(load_config(), analyzer=stub, cache_dir=tmp_path)
    analyzer.profile('문장이다.', ['앞'])
    analyzer.profile('문장이다.', ['앞'])
    analyzer.profile('문장이다.', ['다른 앞'])
    assert len(stub.calls) == 2 and analyzer.hits == 1
