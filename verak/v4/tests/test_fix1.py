"""Targeted v4.6 boundary, lossless data rewrites and realistic recovery states."""
from copy import deepcopy
import json
import pytest

from verak.v4.content_env import dumps
from verak.v4.environment_v45 import V45Environment, occurrences
from verak.v4.environment_v46 import V46Environment, terminal_only
from verak.v4.fix1_data import widen, rejected_action, after_rejection
from verak.v4.fix1_taxonomy import classify, nearest_span
from verak.v4.render_v43 import assert_rendered
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row


@pytest.fixture(autouse=True)
def no_contract_io(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **kw: {})


def env(text, role='korean', cls=V46Environment):
    value = cls(row(text), FakeBoundaryAnalysis())
    if role == 'korean':
        value.start_korean([])
    else:
        value.delegate([{'location': ['P1'], 'action': 'EDIT', 'instruction': '배정 부분만 고친다.', 'needs_search': 'no'}], 1)
    return value


@pytest.mark.parametrize('mark', ['.', '?', '!'])
def test_only_final_missing_terminal_is_newly_accepted(mark):
    text = '#@이름#은 학생이다'
    action = {'action': 'EDIT', 'sentence': 'S1', 'old': '학생이다', 'new': '학생이다'+mark}
    old = env(text, cls=V45Environment); new = env(text)
    assert not old.step(dumps(action), 'korean')['valid']
    assert new.step(dumps(action), 'korean')['valid']
    assert new.document.text == text+mark and len(new.document.units) == 1
    assert new.step(dumps({'action': 'UNDO'}), 'korean')['valid']
    assert new.document.text == text


@pytest.mark.parametrize('text,old,new', [
    ('그는 학생임', '학생임', '학생이다.'),  # wording plus terminal is NOT the exception
    ('그는 학생이다.', '학생이다.', '학생이다.!'),
    ('그는 학생이다', '학생이다', '학생이다..'),
    ('그는 학생이다', '학생이다', '학생이다. 설명한다'),
    ('#@이름#은 학생이다', '#@이름#은 학생이다', '#@다름#은 학생이다.'),
])
def test_other_boundary_and_mask_guards_remain(text, old, new):
    value = env(text)
    result = value.step(dumps({'action': 'EDIT', 'sentence': 'S1', 'old': old, 'new': new}), 'korean')
    assert not result['valid'] and value.document.text == text


@pytest.mark.parametrize('text,old,new', [
    ('그 학생은 책을 읽었다.', '학생', '아이'),
    ('그 학생은 책을 읽었다.', '은 책', '이 글'),
    ('그 학생은  책을 읽었다.', '  ', ' '),
    ('#@이름#은 친구와 만났다.', '친구', '친구들'),
    ('가  나와 다를 읽었다.', '가 나', '가 라'),
])
def test_widening_preserves_exact_edit_result_and_unique_eojeols(text, old, new):
    action = {'action': 'EDIT', 'sentence': 'S1', 'old': old, 'new': new}
    widened, audit = widen(text, action)
    a, b = audit['original_span']; c, d = audit['widened_span']
    assert text[:a]+new+text[b:] == text[:c]+widened['new']+text[d:]
    assert len(occurrences(text, widened['old'])) == 1


@pytest.mark.parametrize('kind', ['span_not_found', 'span_ambiguous', 'no_op_edit'])
@pytest.mark.parametrize('role', ['korean', 'revision'])
def test_synthetic_rejection_matches_environment_observation(kind, role):
    value = env('학생은 학생들과 책을 일었다.', role)
    before = assert_rendered(value.public_messages(role))
    target = {'action': 'EDIT', 'sentence': 'S1', 'old': '학생은 학생들과 책을 일었다.', 'new': '학생은 학생들과 책을 읽었다.'}
    bad = rejected_action(before, target, kind)
    assert bad is not None
    expected = after_rejection(before, bad, role, kind)
    record = value.step(dumps(bad), role)
    assert not record['valid']
    actual = assert_rendered(value.public_messages(role))
    assert actual == expected
    assert classify(record, actual)['type'] == kind
    assert value.step(dumps(target), role)['valid']


def test_stop_targets_only_on_existing_done_states_and_step_accounting():
    value = env('학생은 학생들과 책을 읽었다.')
    before = assert_rendered(value.public_messages('korean'))
    done = {'action': 'STOP', 'status': 'done', 'summary': '검토를 마쳤다.', 'issues': []}
    blocked = {**done, 'status': 'blocked'}
    assert rejected_action(before, blocked, 'no_op_edit') is None
    bad = rejected_action(before, done, 'no_op_edit')
    before['steps_left'] = 3
    after = after_rejection(before, bad, 'korean', 'no_op_edit')
    assert after['steps_left'] == 2 and 'notice' in after
    before['steps_left'] = 1
    assert after_rejection(before, bad, 'korean', 'no_op_edit') is None


def test_taxonomy_distinguishes_terminal_only_and_near_copy():
    payload = {'paragraphs': [{'id': 'P1', 'sentences': [{'id': 'S1', 'text': '그 학생은 책을 읽었다'}]}]}
    action = {'action': 'EDIT', 'valid': False, 'error': 'KOREAN EDIT는 종결부호나 문장 경계를 추가할 수 없습니다.',
              'value': {'action': 'EDIT', 'sentence': 'S1', 'old': '읽었다', 'new': '읽었다.'}}
    assert classify(action, payload)['type'] == 'boundary_terminal_only'
    action['value']['new'] = '읽었습니다.'
    assert classify(action, payload)['type'] == 'boundary_other'
    near = nearest_span('그 학생은 책을 읽었다.', '학생 책을 읽었다.')
    assert near['near_copy'] and near['distance'] == 1


def test_fix1_integrated_controller_retains_forced_stop_failure(tmp_path):
    from verak.v4.fix1_infer import run_case
    class Tokens:
        def apply_chat_template(self, *args, **kwargs): return [1]
        def encode(self, *args, **kwargs): return [1]
    class Backend:
        turn = 0
        def generate(self, messages, role):
            if role == 'orchestrator':
                self.turn += 1
                value = [{'action': 'SCORE'}, {'action': 'MAP', 'map': {
                    'genre': 'argumentative', 'paragraph_roles': [{'paragraph': 'P1', 'role': 'body', 'key_sentence': 'S1'}],
                    'sentence_relations': [], 'paragraph_relations': [], 'off_topic': []}},
                    {'action': 'FINISH', 'summary': '마친다.', 'writer_notes': []}][self.turn-1]
            else:
                value = {'action': 'EDIT', 'sentence': 'S1', 'old': '학생이다.', 'new': '학생이다.'}
            return {'raw': dumps(value), 'usage': {'prompt_tokens': 1}, 'finish_reason': 'stop'}
    source = row('학생이다.'); source['genre'] = 'argumentative'
    result = run_case(source, case_id='test', output=tmp_path, backend=Backend(), tok=Tokens(),
        analysis=FakeBoundaryAnalysis(), score=lambda *a: {'expected': [5]*8, 'integers': [5]*8})
    assert not result['completion']['no_protocol_error']
    assert result['stages'][0]['ending'] == 'controller_STOP_blocked'
    assert result['stages'][0]['steps'] == 3
    assert result['final_token_document']['layout'] == result['final']['layout']


def test_old_new_only_worker_adapter_route_changes_and_settings_separate():
    from verak.v4.fix1_infer import CONDITIONS, SETTINGS
    assert CONDITIONS['old']['orchestrator'] == CONDITIONS['new']['orchestrator']
    assert CONDITIONS['old']['revision'] != CONDITIONS['new']['revision']
    assert CONDITIONS['old']['korean'] != CONDITIONS['new']['korean']
    assert [s['temperature'] for s in SETTINGS] == [0, 0, 0, .7]
    assert len({s['seed'] for s in SETTINGS[:3]}) == 3
