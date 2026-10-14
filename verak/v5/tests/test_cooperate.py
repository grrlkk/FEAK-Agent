"""Integration regressions: scope, reports, final review and strict failures."""
import json

import pytest

from verak.v4.content_env import dumps
from verak.v4.fix1_infer import run_case
from verak.v4.render_v43 import assert_rendered
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v5.cooperate_env import CooperationEnvironment


@pytest.fixture(autouse=True)
def no_contract_io(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **kw: {})


def env(role='korean'):
    value = CooperationEnvironment(row('첫 문장이다. 둘째 문장이다.'), FakeBoundaryAnalysis())
    if role == 'korean':
        value.start_korean([])
    else:
        value.delegate([{'location': ['S1'], 'action': 'EDIT', 'instruction': '첫 문장 검토', 'needs_search': 'no'}], 1)
    return value


def edit(value, old, new, role='korean', sid='S1'):
    return value.step(dumps({'action': 'EDIT', 'sentence': sid, 'old': old, 'new': new}), role)


def test_repeat_remains_invalid_but_different_edit_in_same_sentence_is_allowed():
    value = env()
    assert not edit(value, '첫', '첫')['valid']
    repeat = edit(value, '첫', '첫')
    assert not repeat['valid'] and repeat['duplicate_rejection']
    assert value.steps_used == 2 and not value.closed
    assert edit(value, '첫', '새')['valid']
    assert value.rejected_in_a_row == 0 and value.document.text.startswith('새')


def test_same_action_is_reconsidered_after_document_change():
    value = env()
    assert not edit(value, '새', '다른')['valid']
    assert edit(value, '첫', '새')['valid']
    result = edit(value, '새', '다른')
    assert result['valid'] and not result.get('duplicate_rejection')


def test_three_rejections_still_force_a_failure_and_do_not_erase_sentence():
    value = env()
    for _ in range(3): result = edit(value, '첫', '첫')
    assert not result['valid'] and value.closed and value.controller_stop['failure']
    assert value.steps_used == 3
    value.start_korean([])
    assert edit(value, '첫', '새')['valid']


def test_whole_pass_permissions_and_scope_notice_agree():
    value = env()
    observation = assert_rendered(value.public_messages('korean'))
    assert observation['whole_essay_typo_spacing_pass']
    assert observation['scope'] == 'whole_essay_form_pass'
    assert {u.sid for u in value.document.units} <= set(observation['allowed_scope'])
    assert not observation['cooperation']['observations_are_error_verdicts']
    sid = value.document.units[-1].sid
    assert edit(value, '둘째', '두 번째', sid=sid)['valid']


def test_revision_scope_and_korean_role_constraints_are_preserved():
    value = env('revision')
    sid = value.document.units[-1].sid
    assert not edit(value, '둘째', '두 번째', role='revision', sid=sid)['valid']
    assert not value.step(dumps({'action': 'MOVE', 'sentence': 'S1', 'to': {'after': sid}}), 'revision')['valid']
    value = env()
    assert not value.step(dumps({'action': 'DELETE', 'sentence': 'S1'}), 'korean')['valid']
    assert not edit(value, '첫 문장이다.', '문장이다. 더한다.')['valid']


class Tokens:
    def apply_chat_template(self, *args, **kwargs): return [1]
    def encode(self, *args, **kwargs): return [1]


class Scripted:
    def __init__(self, *, delegated=False, reject=False):
        self.observations = []
        self.delegated = delegated
        self.reject = reject
        self.count = 0

    def generate(self, messages, role):
        self.observations.append((role, messages))
        if role == 'orchestrator':
            self.count += 1
            actions = [{'action': 'SCORE'}, {'action': 'MAP', 'map': {
                'genre': 'argumentative', 'paragraph_roles': [{'paragraph': 'P1', 'role': 'body', 'key_sentence': 'S1'}],
                'sentence_relations': [], 'paragraph_relations': [], 'off_topic': []}}]
            if self.delegated:
                actions.append({'action': 'DELEGATE', 'to': 'revision', 'items': [{
                    'location': ['S1'], 'action': 'EDIT', 'instruction': '첫 문장을 검토한다.', 'needs_search': 'no'}]})
            actions += [{'action': 'FINISH', 'summary': '보고 확인', 'writer_notes': ['남은 관계 판단']}]*25
            action = actions[self.count-1]
        elif self.reject:
            action = {'action': 'EDIT', 'sentence': 'S1', 'old': '문장', 'new': '문장'}
        else:
            action = {'action': 'STOP', 'status': 'blocked', 'summary': '관계가 불확실하다.', 'issues': ['추가 설명 필요']}
        return {'raw': dumps(action), 'usage': {'prompt_tokens': 1}, 'finish_reason': 'stop'}


def run(tmp_path, backend, *, cooperation=True, capture=True):
    source = row('문장이다.'); source['genre'] = 'argumentative'
    return run_case(source, case_id='fixture', output=tmp_path, backend=backend,
        tok=Tokens(), analysis=FakeBoundaryAnalysis(),
        score=lambda *args: {'expected': [5]*8, 'integers': [5]*8},
        cooperation=cooperation, capture_states=capture)


def test_orchestrator_receives_real_worker_issues_and_confirms_final_pass(tmp_path):
    backend = Scripted(delegated=True)
    result = run(tmp_path, backend)
    assert result['completion']['no_protocol_error']
    assert not result['completion']['all_stages_STOP_done']
    assert [s['role'] for s in result['stages']] == ['revision', 'korean']
    assert result['stages'][-1]['automatic']
    observations = [json.loads(m[1]['content']) for role, m in backend.observations if role == 'orchestrator']
    reports = observations[-1]['worker_reports']
    assert [r['role'] for r in reports] == ['revision', 'korean']
    assert all(r['report']['issues'] == ['추가 설명 필요'] for r in reports)
    assert observations[-1]['integration']['final_review_pending']
    assert result['finish_snapshot'] == reports[-1]['final_snapshot']
    assert sum(r.get('finish_deferred_for_worker_report', False) for r in result['orchestrator_actions']) == 1
    korean = next(assert_rendered(m) for role, m in backend.observations if role == 'korean')
    assert korean['cooperation']['previous_worker_reports'][0]['report']['issues'] == ['추가 설명 필요']


def test_controller_stop_is_not_protocol_completion_even_if_orchestrator_finishes(tmp_path):
    result = run(tmp_path, Scripted(reject=True))
    assert result['orchestrator_ending'] == 'FINISH'
    assert not result['completion']['no_protocol_error']
    assert result['controller_events'][0]['failure']


def test_default_runner_observations_unchanged_by_capture_and_no_extra_finish(tmp_path):
    plain, captured = Scripted(), Scripted()
    a = run(tmp_path/'plain', plain, cooperation=False, capture=False)
    b = run(tmp_path/'captured', captured, cooperation=False, capture=True)
    assert plain.observations == captured.observations
    assert plain.count == captured.count == 3
    assert a['completion'] == b['completion']
    assert 'integration' not in json.loads(plain.observations[-2][1][1]['content'])
    assert 'initial_token_document' not in a and 'initial_token_document' in b


def test_mask_and_multiple_object_guards_are_unchanged():
    value = CooperationEnvironment(row('#@이름#은 학생이다.'), FakeBoundaryAnalysis())
    value.start_korean([])
    assert not edit(value, '#@이름#', '누군가')['valid']
    assert not value.step('{"action":"UNDO"}{"action":"UNDO"}', 'korean')['valid']
    assert value.document.text == '#@이름#은 학생이다.'
