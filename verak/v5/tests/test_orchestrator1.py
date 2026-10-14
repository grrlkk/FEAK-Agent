"""The model intervention preserves worker requests and strict local audits."""
from contextlib import contextmanager
from copy import deepcopy
import json
import sqlite3

import pytest

from verak.v5.api import BudgetStop
from verak.v5.common import ref, write_json
from verak.v5.orchestrator1 import HybridBackend, MODEL
from verak.v5.orchestrator1_eval import make_judge_case, identity_errors, delegation_audit
from verak.v5.tests.test_cooperate import Scripted, run


@pytest.fixture(autouse=True)
def no_contract_io(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **kw: {})


class FakeAPI:
    model = MODEL
    spec = {'reasoning_effort':'none', 'temperature':0}

    def __init__(self):
        self.connection = sqlite3.connect(':memory:')
        self.connection.execute('CREATE TABLE calls (fingerprint TEXT)')
        self.script = Scripted(delegated=True)
        self.requests = []

    @contextmanager
    def db(self):
        yield self.connection

    def request(self, messages, *, item_id, max_output):
        self.requests.append((messages, item_id, max_output))
        self.connection.execute('INSERT INTO calls VALUES (?)', (item_id,))
        return {**self.script.generate(messages, 'orchestrator'),
            'usage':{'input_tokens':999, 'output_tokens':10},
            'response_model':self.model, 'fingerprint':item_id, 'status':'completed'}


def test_only_orchestrator_changes_provider_with_identical_observations(tmp_path):
    control = Scripted(delegated=True)
    baseline = run(tmp_path/'control', control)
    api, worker = FakeAPI(), Scripted(delegated=True)
    expected = control.observations[0][1]
    backend = HybridBackend(worker, api, 'fixture', expected_first_messages=expected)
    result = run(tmp_path/'hybrid', backend)
    assert result['completion'] == baseline['completion']
    assert result['final'] == baseline['final']
    assert worker.observations == [(r,m) for r,m in control.observations if r != 'orchestrator']
    assert api.script.observations == [(r,m) for r,m in control.observations if r == 'orchestrator']
    assert all(limit == 4096 for _,_,limit in api.requests)
    assert result['new_paid_calls'] == len(api.requests) > 0
    for call in result['calls']:
        audit = call['template_audit']
        if call['role'] == 'orchestrator':
            assert audit['status'] == 'different_tokenizer'
            assert audit['admission_prefix_tokens'] == 1
            assert audit['provider_input_tokens'] == 999
            assert audit['counts_are_equivalent'] is False
        else:
            assert audit['status'] == 'match'
    stages = delegation_audit(result)
    assert len(stages) == 2
    assert [s['valid_actions'] for s in stages] == [{'STOP':1}, {'STOP':1}]
    assert all(s['STOP_is_not_verified_task_completion'] for s in stages)


def test_mismatched_first_observation_and_unknown_role_never_dispatch():
    api = FakeAPI()
    backend = HybridBackend(Scripted(), api, 'fixture', expected_first_messages=[])
    with pytest.raises(ValueError, match='differs from saved control'):
        backend.generate([{'role':'user','content':'changed'}], 'orchestrator')
    with pytest.raises(ValueError, match='Unknown policy role'):
        backend.generate([], 'oneshot')
    assert not api.requests


def test_api_token_exception_does_not_disable_worker_template_checks(tmp_path):
    class BadWorker(Scripted):
        def generate(self, messages, role):
            return {**super().generate(messages, role), 'usage':{'prompt_tokens':999}}
    result = run(tmp_path, HybridBackend(BadWorker(), FakeAPI(), 'fixture'))
    assert not result['completion']['no_protocol_error']
    failed = [c for c in result['calls'] if c['action']['action']=='NOT_EXECUTED_TEMPLATE']
    assert failed and all(c['role'] != 'orchestrator' for c in failed)
    assert all(not a['valid'] for role in ('revision','korean') for a in result['actions'][role])


def test_api_provenance_and_budget_failure_remain_errors():
    api = FakeAPI(); backend = HybridBackend(Scripted(), api, 'fixture')
    response = backend.generate([], 'orchestrator')
    for replacement in ({'response_model':'other-model'}, {'usage':{'input_tokens':None}},
                        {'generation':{'provider':'local'}}):
        assert backend.audit_prompt_tokens('orchestrator', 10, {**response, **replacement})['status'] == 'invalid_api_provenance'
    def stop(*args, **kwargs): raise BudgetStop('fixture cap')
    api.request = stop
    with pytest.raises(BudgetStop): backend.generate([], 'orchestrator')
    assert backend.budget_stop


def test_api_preflight_runs_before_gpu_allocation(tmp_path, monkeypatch):
    from verak.v5 import orchestrator1
    from verak.v4 import full_run
    events = []
    monkeypatch.setattr(orchestrator1, 'ROOT', tmp_path)
    monkeypatch.setattr(orchestrator1, 'verify_design', lambda: {})
    monkeypatch.setattr(orchestrator1, 'credentials', lambda: {'luna':True,'claude':True})
    monkeypatch.setattr(orchestrator1, 'probe', lambda: events.append('api_preflight'))
    def busy(*args):
        events.append('idle_check')
        raise RuntimeError('GPUs busy')
    monkeypatch.setattr(full_run, 'idle', busy)
    with pytest.raises(RuntimeError, match='GPUs busy'): orchestrator1.gpu()
    assert events == ['api_preflight','idle_check']


def test_full_text_judging_is_blind_and_retains_only_source_evidence(tmp_path):
    path = tmp_path/'row.json'
    write_json(path, {'question':'문항', 'text':'원문', 'profile':'hidden profile'})
    case = {'case_id':'fixture', 'cohort':'smoke20', 'row':ref(path)}
    episode = {'initial':{'text':'원문'}, 'final':{'text':'수정문'},
        'score':999, 'model':'hidden model', 'actions':['hidden action'],
        'source_citations':{'search_history':[{'query':'hidden query', 'passages':[
            {'passage_id':'p1', 'title':'자료', 'text':'근거', 'url':'https://example.test', 'score':999}]}],
            'source_citations':[{'text':'주장', 'source':'p1', 'internal':'hidden'}]}}
    episodes = {'control':episode, 'luna':deepcopy(episode)}
    episodes['luna']['final']['text'] = '다른 수정문'
    request = make_judge_case(case, episodes, ['luna','control'], 0)
    payload = json.loads(request['messages'][1]['content'])
    assert request['aliases'] == {'A':'luna', 'B':'control'}
    assert payload['original'] == '원문'
    assert payload['variants'][0]['text'] == '다른 수정문'
    assert set(payload) == {'question','original','variants'}
    assert 'hidden' not in json.dumps(payload)
    assert payload['variants'][0]['source_evidence']['retrieved_passages'][0]['text'] == '근거'
    reversed_request = make_judge_case(case, episodes, ['control','luna'], 1)
    assert reversed_request['id'] != request['id']
    episodes['luna']['initial']['text'] = 'different source'
    with pytest.raises(ValueError, match='different source essays'):
        make_judge_case(case, episodes, ['control','luna'], 0)


def test_unchanged_text_cannot_be_counted_as_improved_or_strictly_preferred():
    messages = [{}, {'content':json.dumps({'original':'same', 'variants':[
        {'id':'A','text':'same'}, {'id':'B','text':'same'}]})}]
    verdict = {'preference':'tie', 'variants':[{'id':label, 'better_than_original':'no',
        'meaning_preserved':'yes', 'new_specific_claims':'none'} for label in ('A','B')]}
    assert not identity_errors(verdict, messages)
    verdict['variants'][0]['better_than_original'] = 'yes'
    verdict['preference'] = 'A'
    assert set(identity_errors(verdict, messages)) == {
        'unchanged_original_rated_better', 'identical_texts_strict_preference'}


def test_no_search_pair_uses_new_control_and_preserves_hybrid_input_parity(tmp_path):
    from verak.v5.tests.test_no_search import Backend, SEARCH, EDIT, run as no_search_run
    from verak.v5.orchestrator1 import audit_no_search
    baseline_backend = Backend([SEARCH, EDIT])
    baseline = no_search_run(tmp_path/'control', baseline_backend)
    api = FakeAPI()
    api.script = Backend([SEARCH, EDIT])
    workers = Backend([SEARCH, EDIT])
    backend = HybridBackend(workers, api, 'fixture',
                            expected_first_messages=baseline_backend.observations[0][1])
    result = no_search_run(tmp_path/'luna', backend)
    assert audit_no_search(baseline) and audit_no_search(result)
    assert result['final'] == baseline['final']
    assert result['writer_notes'] == baseline['writer_notes']
    assert result['deferred_task_count'] == 1
    assert api.script.observations == [(r,m) for r,m in baseline_backend.observations if r=='orchestrator']
    assert workers.observations == [(r,m) for r,m in baseline_backend.observations if r!='orchestrator']


@pytest.mark.parametrize('corruption', ['enabled','history','citation','action','task'])
def test_no_search_audit_fails_closed(corruption):
    from verak.v5.orchestrator1 import audit_no_search
    episode = {'search_enabled':False, 'source_citations':{}, 'actions':{}, 'stages':[]}
    if corruption == 'enabled': episode['search_enabled'] = True
    if corruption == 'history': episode['source_citations']['search_history'] = [{}]
    if corruption == 'citation': episode['source_citations']['source_citations'] = [{}]
    if corruption == 'action': episode['actions']['revision'] = [{'action':'SEARCH','valid':True}]
    if corruption == 'task': episode['stages'] = [{'assigned_tasks':[{'needs_search':'yes'}]}]
    with pytest.raises(ValueError): audit_no_search(episode)


def test_no_search_control_cannot_reuse_search_enabled_artifacts(tmp_path, monkeypatch):
    from verak.v5 import orchestrator1
    monkeypatch.setattr(orchestrator1, 'SEARCH_ENABLED', False)
    monkeypatch.setattr(orchestrator1, 'ROOT', tmp_path)
    case = {'case_id':'fixture', 'cohort':'smoke20', 'controls':{'1':{'episode':'historical_search_enabled'}}}
    with pytest.raises(FileNotFoundError): orchestrator1.control_artifacts(case)
    artifacts = {}
    for kind,folder in (('episode','attempts'), ('scores','quality_scores')):
        path = orchestrator1.control_dir(case)/folder/'fixture.json'
        write_json(path, {'search_enabled':False})
        artifacts[kind] = ref(path)
    assert orchestrator1.control_artifacts(case) == artifacts
    write_json(tmp_path/'control_complete.json', {'artifacts':{'fixture':artifacts}})
    write_json(orchestrator1.control_dir(case)/'attempts/fixture.json', {'changed':True})
    with pytest.raises(ValueError): orchestrator1.verified(orchestrator1.control_artifacts(case)['episode'])
