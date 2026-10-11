"""CPU contracts for inference-only fixes and Orchestrator supervision."""
import json

import pytest

from verak.v4.content_env import dumps
from verak.v4.environment_v43 import V43Environment
from verak.v4.environment_v44 import V44Environment, NOOP_NOTICE
from verak.v4.orchestrator_v44 import messages, validate
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row


@pytest.fixture(autouse=True)
def no_contract_io(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **k: {})


def env(cls=V44Environment):
    obj = cls(row('#@이름#의 이야기다. 다른 문장이다.'), FakeBoundaryAnalysis())
    obj.delegate([{'location': ['P1'], 'action': 'EDIT', 'instruction': '검토한다.', 'needs_search': 'no'}], 1)
    return obj


def test_whole_mask_delete_only_v44_and_undo():
    old, new = env(V43Environment), env()
    action = dumps({'action': 'DELETE', 'sentence': 'S1'})
    original = new.document.text
    assert not old.step(action, 'revision')['valid']
    assert new.step(action, 'revision')['valid']
    assert '#@이름#' not in new.document.text
    assert new.step(dumps({'action': 'UNDO'}), 'revision')['valid']
    assert new.document.text == original


def test_mask_move_and_mask_modification_are_distinct():
    new = env()
    anchor = new.document.units[1].sid
    assert new.step(dumps({'action': 'MOVE', 'sentence': 'S1', 'to': {'after': anchor}}), 'revision')['valid']
    before = new.document.text
    assert not new.step(dumps({'action': 'EDIT', 'sentence': 'S1', 'old': '#@이름#', 'new': '#@다른이름#'}), 'revision')['valid']
    assert new.document.text == before


def test_three_rejections_are_failure_not_a_valid_model_STOP():
    new = env()
    noop = dumps({'action': 'EDIT', 'sentence': 'S1', 'old': '이야기', 'new': '이야기'})
    for index in range(3):
        result = new.step(noop, 'revision')
        assert not result['valid'] and NOOP_NOTICE in result['error']
        assert new.closed == (index == 2)
    assert new.controller_stop['status'] == 'blocked' and new.controller_stop['failure']
    assert all(a['action'] != 'STOP' for a in new.actions['revision'])
    with pytest.raises(RuntimeError):
        new.step(noop, 'revision')


def test_rejection_streak_resets_on_success_and_at_new_stage():
    new = env()
    invalid = dumps({'action': 'DELETE', 'sentence': 'S999'})
    new.step(invalid, 'revision'); new.step(invalid, 'revision')
    assert new.step(dumps({'action': 'EDIT', 'sentence': 'S1', 'old': '이야기', 'new': '기록'}), 'revision')['valid']
    assert new.rejected_in_a_row == 0
    new.step(invalid, 'revision')
    assert not new.closed
    new.start_korean([])
    assert new.rejected_in_a_row == 0
    assert not new.step(dumps({'action': 'DELETE', 'sentence': 'S1'}), 'korean')['valid']


def test_revision_scope_and_multiple_objects_stay_strict():
    new = env()
    new.scope = {'S2'}
    assert not new.step(dumps({'action': 'DELETE', 'sentence': 'S1'}), 'revision')['valid']
    assert not new.step('{"action":"UNDO"}\n{"action":"UNDO"}', 'revision')['valid']


def test_orchestrator_input_whitelist_and_strict_actions():
    paragraphs = [{'id': 'P1', 'private': 'answer', 'sentences': [{'id': 'S1', 'text': '글.', 'feedback': 'secret'}]}]
    observation = messages('질문', paragraphs, phase='score')
    assert 'secret' not in dumps(observation) and 'answer' not in dumps(observation)
    assert validate('{"action":"SCORE"}', paragraphs)[0] == {'action': 'SCORE'}
    with pytest.raises(ValueError):
        validate('{"action":"SCORE"}{"action":"FINISH"}', paragraphs)
    value = {'action': 'DELEGATE', 'to': 'korean', 'items': [
        {'location': ['S1'], 'action': 'DELETE', 'instruction': '삭제', 'needs_search': 'no'}]}
    with pytest.raises(ValueError):
        validate(dumps(value), paragraphs)


def test_shared_split_is_exact_and_has_no_cross_role_leakage():
    from verak.v4.full_data import shared_split
    import math
    sets = {'revision': {str(i) for i in range(100)}, 'korean': {str(i) for i in range(20, 120)},
            'orchestrator': {str(i) for i in range(200)}, 'oneshot': {str(i) for i in range(10, 90)}}
    held = shared_split(sets)
    assert held == shared_split(sets)
    for sources in sets.values():
        assert len(held & sources) == math.ceil(.05*len(sources))


def test_prompt_freeze_roundtrips_tuple_metadata(tmp_path, monkeypatch):
    from verak.v4 import orchestrator_v44 as orch
    class Tokens:
        def encode(self, *a, **k): return [1]
        def apply_chat_template(self, *a, **k): return [1, 2]
    monkeypatch.setattr(orch, 'FROZEN', tmp_path/'frozen.json')
    assert orch.freeze(Tokens()) == orch.freeze(Tokens())


def test_plan_target_drops_deleted_location_without_claiming_recovery(monkeypatch):
    from verak.v4 import full_data
    initial = [{'id': 'P1', 'sentences': [{'id': 'S1', 'text': '첫 문장.'}, {'id': 'S2', 'text': '다음 문장.'}]}]
    after = [{'id': 'P1', 'sentences': [{'id': 'S2', 'text': '다음 문장.'}]}]
    monkeypatch.setattr(full_data, 'paragraphs_for', lambda attempt, role, delegation, fallback: after if role == 'korean' else initial)
    plan = {'items': [{'owner': 'revision', 'location': ['S1'], 'action': 'DELETE', 'instruction': '중복 문장을 검토한다.', 'needs_search': 'no'}] + [
        {'owner': 'korean', 'location': [sid], 'action': 'EDIT', 'instruction': '띄어쓰기를 검토한다.', 'needs_search': 'no'} for sid in ('S1', 'S2')]}
    attempt = {'delegations': [{'delegation': 1, 'ending': 'STOP', 'stop_status': 'done'}],
        'korean_stage': {'ending': 'STOP', 'stop_status': 'blocked'}, 'final_paragraphs': after}
    notes = []
    samples = list(full_data.plan_samples({'source_id': 'fixture', 'question': '질문', 'paragraphs': initial},
        plan, attempt, {'kind': 'B3_plan'}, notes))
    targets = [json.loads(s['messages'][-1]['content']) for s in samples]
    assert targets[2]['items'][0]['location'] == ['S2'] and len(targets[2]['items']) == 1
    assert targets[3]['summary'] == '일부 위임이 완료되지 않았다.'
    assert notes[0]['removed_task_locations'] == [['S1']]
    assert plan['items'][1]['location'] == ['S1']


def test_new_trainer_global_target_normalization(monkeypatch):
    from verak.v3.rft1 import train as previous
    from verak.v4.full_train import distributed_action_loss
    from verak.v3.tests.test_rft1 import test_two_rank_accumulated_gradient_equals_global_target_token_mean
    monkeypatch.setattr(previous, 'distributed_action_loss', distributed_action_loss)
    test_two_rank_accumulated_gradient_equals_global_target_token_mean()


def test_integrated_controller_failure_keeps_model_STOP_denominator(tmp_path):
    from verak.v4.full_infer import run_case, summarize
    class Tokens:
        def apply_chat_template(self, *args, **kwargs): return [1]
        def encode(self, *args, **kwargs): return [1]
    class Backend:
        def generate(self, messages, role):
            # Three invalid Orchestrator actions trigger the mandatory Korean
            # pass; neither a saved plan nor records are available in this test.
            if role == 'orchestrator': raw = '{"action":"UNKNOWN"}'
            else: raw = dumps({'action': 'STOP', 'status': 'done', 'summary': '검토 완료', 'issues': []})
            return {'raw': raw, 'usage': {'prompt_tokens': 1}}
    source = row('검토할 문장이다.')
    source.update(source_id='fixture:1', genre='argumentative')
    result = run_case(source, case_id='fixture:1', output=tmp_path, backend=Backend(),
                      tok=Tokens(), analysis=FakeBoundaryAnalysis(), score=lambda *a: None)
    assert not result['complete'] and result['orchestrator_ending'] == 'controller_STOP_blocked'
    assert len(result['stages']) == 1 and result['stages'][0]['automatic']
    stats = summarize([result], 1)
    assert stats['roles']['orchestrator']['returned_actions'] == 3
    assert stats['roles']['orchestrator']['valid_actions'] == 0
    assert stats['roles']['orchestrator']['controller_STOP_blocked'] == 1
    assert stats['roles']['korean']['model_STOP_done'] == 1


def test_returned_template_failure_counts_as_invalid_action(tmp_path):
    from verak.v4.full_infer import run_case, summarize
    class Tokens:
        def apply_chat_template(self, *args, **kwargs): return [1]
        def encode(self, *args, **kwargs): return [1]
    class Backend:
        def generate(self, messages, role):
            return {'raw': '{"action":"SCORE"}', 'usage': {'prompt_tokens': 2}}
    source = row('검토할 문장이다.')
    source.update(source_id='fixture:2', genre='argumentative')
    result = run_case(source, case_id='fixture:2', output=tmp_path, backend=Backend(),
                      tok=Tokens(), analysis=FakeBoundaryAnalysis(), score=lambda *a: None)
    stats = summarize([result], 1)
    assert not result['complete']
    assert len(stats['execution_failures']) == 2
    for role in ('orchestrator', 'korean'):
        assert stats['roles'][role]['returned_actions'] == 1
        assert stats['roles'][role]['valid_action_rate'] == 0.


def test_dependency_recheck_does_not_retry_model_or_context_failures():
    from verak.v4.inference_recovery44 import dependency_failure
    context_failure = {'calls': [], 'context_limits': [{'role': 'orchestrator'}]}
    model_failure = {'calls': [{'action': {'valid': False, 'error': 'ValueError: invalid JSON'}}]}
    missing_search = {'calls': [{'action': {'valid': False,
        'error': "ModuleNotFoundError: No module named 'bm25s'"}}]}
    assert not dependency_failure(context_failure)
    assert not dependency_failure(model_failure)
    assert dependency_failure(missing_search)
