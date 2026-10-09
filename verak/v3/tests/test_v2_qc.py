from copy import deepcopy
from types import SimpleNamespace

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.common import read_json, write_json
from verak.v3.tests.test_environment import setup_env
from verak.v3.tests.test_teacher_bulk import FakeAdapter
from verak.v3.v2_ops.config import config_for, require_scorer_slot
from verak.v3.v2_ops.judges import (label_contract, qc_contract, recovery_contract, passes_qc, validate)
from verak.v3.v2_ops.operators import delete_link, label_targets
from verak.v3.v2_ops.paid import V2API
from verak.v3.v2_ops.qc import summary


def test_judge_contracts_cover_exact_sites_and_require_all_qc_fields(setup_env):
    _, episode, _ = setup_env
    source = episode['document']
    row = {'source_id': 'source', 'question': episode['question'], 'label_targets': label_targets(source)}
    _, schema = label_contract([(row, source)])
    valid = {'source': {sid: 'topic' for sid in row['label_targets']}}
    validate(valid, schema)
    bad = deepcopy(valid)
    bad['source'].pop(row['label_targets'][0])
    with pytest.raises(ValueError):
        validate(bad, schema)
    changed, record = delete_link(source, row['label_targets'][0], 'topic')
    candidate = {**row, 'episode_id': 'episode', 'operator': 'G_DEL_LINK',
        'source_text': source.text, 'corrupted_text': changed.text, 'records': [record]}
    _, qc_schema = qc_contract([candidate])
    verdict = {'damage_real': True, 'original_is_fix': True, 'recoverable_from_essay': False, 'reason': '새 정보'}
    validate({'episode': verdict}, qc_schema)
    assert not passes_qc(verdict, 'G_DEL_LINK')
    assert passes_qc(verdict, 'L_FUSE')
    messages, recovery_schema = recovery_contract(candidate, record, '삽입한 문장이다.')
    assert record['recovery_target']['original'] in messages[-1]['content']
    validate({'verdict': 'partial', 'reason': '역할은 유지됨'}, recovery_schema)


def test_shared_cap_resume_across_models_and_paid_gate(tmp_path, monkeypatch):
    from verak.v3.eval import api as api_module
    monkeypatch.setattr(api_module.socket, 'getaddrinfo', lambda *args: [])
    config = config_for(version='v2')
    config['paths']['v2_ops_output'] = tmp_path
    with pytest.raises(PermissionError, match='Proceed'):
        V2API(config, 20, kind='sol')
    assert not (tmp_path / 'api').exists()
    sol = V2API(config, 20, kind='sol', paid_approved=True, adapter_factory=FakeAdapter)
    luna = V2API(config, 20, kind='luna', paid_approved=True, adapter_factory=FakeAdapter)
    messages = [{'role': 'user', 'content': 'local mock'}]
    saved = sol.request(messages, stage='v2_labels', item_id='one')
    with sol.db() as db:
        db.execute("UPDATE calls SET status='pending',confirmed=0,reserved=?,path=NULL WHERE id=?",
                   (saved['bound_usd'], saved['phase_call']))
    outcome = luna.settle_interrupted()
    assert outcome[0]['reconciliation'] == 'recovered_durable_record'
    assert sol.accounting()['confirmed_usd'] == pytest.approx(saved['cost']['confirmed_usd'])
    assert sol.request(messages, stage='v2_labels', item_id='one')['replayed']
    with luna.db() as db:
        db.execute('UPDATE calls SET confirmed=9.9,reserved=0')
    with pytest.raises(CallBudgetExceeded):
        sol.reserve('v2_labels', 'next', 'next', .11)
    with pytest.raises(CallBudgetExceeded):
        luna.reserve('v2_teacher_1', 'next', 'next', .11)


def test_gpu1_gate_rejects_active_v1_and_never_exposes_gpu0(tmp_path, monkeypatch):
    config = {'method_version': 'v2', 'paths': {'repo': tmp_path}}
    path = tmp_path / 'verak/v3/outputs/phase7_sft/continuation_status.json'
    write_json(path, {'current': {'stage': 'evaluating_epoch_2'}})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(RuntimeError, match='queued'):
        require_scorer_slot(config)
    write_json(path, {'current': {'stage': 'completed_awaiting_final_audit'}})
    for value in ('', '0', '0,1'):
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', value)
        with pytest.raises(RuntimeError, match='physical GPU1'):
            require_scorer_slot(config)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert require_scorer_slot(config) == 0


def test_unknown_qc_never_becomes_a_failure_or_keep(tmp_path):
    plan = {'plans': {op: {split: [{'source_id': split + ':1'}]
        for split in ('agent_train', 'agent_dev')} for op in ('G_DEL_LINK', 'L_FUSE')}}
    write_json(tmp_path / 'source_plan.json', plan)
    config = {'paths': {'v2_ops_output': tmp_path}, 'v2_ops': {'qc_min_yield': .3}}
    result = summary(config)
    for row in result['operators'].values():
        assert row['decision'] == 'undetermined'
        assert row['usable_source_yield'] is None
        assert row['yield_lower_bound'] == 0 and row['yield_upper_bound'] == 1


def test_failed_client_initialization_never_leaves_a_none_client_or_reserves(tmp_path, monkeypatch):
    from verak.v3.eval import api as api_module
    from feak_tc.runtime.openai import APIUnavailable
    monkeypatch.setattr(api_module.socket, 'getaddrinfo', lambda *args: [])
    config = config_for(version='v2')
    config['paths']['v2_ops_output'] = tmp_path
    class InitiallyMissing(FakeAdapter):
        tries = 0
        def __init__(self, config):
            self._client = None
        def _load(self):
            type(self).tries += 1
            if type(self).tries == 1:
                raise APIUnavailable('synthetic missing environment')
            self._client = SimpleNamespace(responses=SimpleNamespace(create=self.create))
    api = V2API(config, 10, kind='sol', paid_approved=True, adapter_factory=InitiallyMissing)
    messages = [{'role': 'user', 'content': 'local'}]
    with pytest.raises(APIUnavailable):
        api.request(messages, stage='v2_labels', item_id='one')
    assert api.accounting()['client_attempts'] == 0
    assert api.request(messages, stage='v2_labels', item_id='one')['status'] == 'completed'
    assert api.accounting()['calls'] == 1


def test_recovery_reference_is_cached_once_and_missing_cache_is_unknown(tmp_path, setup_env):
    from concurrent.futures import ThreadPoolExecutor
    import json
    from verak.v3.v2_ops.teacher import RecoveryJudge
    _, episode, _ = setup_env
    source = episode['document']
    changed, record = delete_link(source, label_targets(source)[0], 'topic')
    config = {'paths': {'v2_ops_output': tmp_path}}
    row = {'question': episode['question'], 'corrupted_text': changed.text}
    unit = source.units[0]
    with pytest.raises(ValueError, match='Missing cached'):
        RecoveryJudge(config, row)(record, unit)
    class JudgeAPI:
        calls = 0
        def request(self, messages, **kwargs):
            self.calls += 1
            assert kwargs['effort'] == 'low'
            assert record['recovery_target']['original'] in messages[-1]['content']
            return {'phase_call': 1, 'raw': json.dumps({'verdict': 'partial', 'reason': '역할 일부 복원'})}
    api = JudgeAPI()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: RecoveryJudge(config, row, api)(record, unit), range(2)))
    assert results == [.5, .5] and api.calls == 1
    assert RecoveryJudge(config, row)(record, unit) == .5
