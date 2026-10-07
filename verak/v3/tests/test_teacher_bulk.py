from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.agent.runner import system_prompt
from verak.v3.common import file_sha, read_json, sha_text, write_json
from verak.v3.phase2 import write_jsonl
from verak.v3.train.teacher_bulk import (PHASE, BulkAPI, BulkTeacher, atomic_new,
    collection_lock, config_for, prepare, tasks)


def fixture_config(tmp_path):
    config = config_for()
    root = tmp_path / 'repo'
    paths = config['paths']
    paths.update(repo=root, active_corrupt=tmp_path / 'corpus', phase4_output=tmp_path / 'phase4',
                 phase7_teacher_output=tmp_path / 'teacher', **{PHASE + '_output': tmp_path / 'bulk'})
    write_json(paths['phase4_output'] / 'models.json', {'luna_model': 'gpt-6-luna'})
    rows = [{'episode_id': f'episode:{i}', 'source_id': f'source:{i}', 'split': 'agent_train',
             'level': f'L{i % 4 + 1}', 'records': [{'op': 'L_CONN'}], 'corrupted_layout': {'text': str(i)}} for i in range(1430)]
    write_jsonl(paths['active_corrupt'] / 'agent_train.jsonl', rows)
    source = root / 'verak/v3/example.py'
    source.parent.mkdir(parents=True)
    source.write_text('frozen')
    pilot_ids = [r['episode_id'] for r in rows[:92]]
    write_json(paths['phase7_teacher_output'] / 'design.json', {'model': 'gpt-6-luna',
        'pilot_ids': pilot_ids, 'code_sha256': {'example.py': file_sha(source)},
        'prompt_sha256': {r: sha_text(system_prompt(r)) for r in ('global', 'korean')}})
    for i, episode_id in enumerate(pilot_ids):
        write_json(paths['phase7_teacher_output'] / 'luna_low/episodes' / (episode_id.replace(':', '_') + '.json'),
            {'corpus_episode_id': episode_id, 'model': 'gpt-6-luna', 'mode': 'two_stage',
             'completed': i != 0, 'initial_layout': {'text': str(i)},
             'calls': [{'reasoning_effort': 'low', 'max_output_tokens': 1024}]})
    return config


def test_exact_attempt_matrix_reuses_saved_failure_and_freezes_contract(tmp_path):
    config = fixture_config(tmp_path)
    design, _ = prepare(config)
    before = deepcopy(design)
    work = list(tasks(design))
    assert len(work) == len(set(work)) == 2768
    assert Counter(a for a, _ in work) == {1: 1338, 2: 1430}
    assert not design['reuse_attempt_1']['episode:0']['completed']
    assert (1, 'episode:0') not in work and (2, 'episode:0') in work
    assert design['orders']['1'] != design['orders']['2']
    assert design['provider_sampling_seed'] is None
    assert prepare(config)[0] == before
    path = config['paths']['phase7_teacher_output'] / 'luna_low/episodes/episode_0.json'
    row = read_json(path)
    row['completed'] = True
    write_json(path, row)
    with pytest.raises(ValueError, match='manifest changed'):
        prepare(config)


def test_attempt_commit_and_collection_lock_cannot_overwrite(tmp_path):
    path = tmp_path / 'episode.json'
    atomic_new(path, {'completed': False})
    with pytest.raises(FileExistsError):
        atomic_new(path, {'completed': True})
    assert read_json(path) == {'completed': False}
    with collection_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another bulk'):
            with collection_lock(tmp_path):
                pass


class FakeAdapter:
    requests = []
    def __init__(self, config):
        self._client = SimpleNamespace(responses=SimpleNamespace(create=self.create))
    def _load(self):
        pass
    def create(self, **kwargs):
        self.requests.append(kwargs)
        usage = {'input_tokens': 100, 'output_tokens': 20,
                 'input_tokens_details': {}, 'output_tokens_details': {}}
        return SimpleNamespace(output_text='{}', status='completed', id='fake', model='gpt-6-luna',
            usage=SimpleNamespace(model_dump=lambda: usage))
    def close(self):
        pass


def test_attempt_two_is_fresh_but_resume_replays_and_unknown_cost_is_retained(tmp_path, monkeypatch):
    from verak.v3.eval import api as module
    monkeypatch.setattr(module.socket, 'getaddrinfo', lambda *args: [])
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path
    FakeAdapter.requests = []
    api = BulkAPI(config, 80000, phase=PHASE, adapter_factory=FakeAdapter)
    messages = [{'role': 'user', 'content': 'unchanged prompt'}]
    one = api.request(messages, stage='bulk_attempt_1', item_id='essay:global:1')
    replay = api.request(messages, stage='bulk_attempt_1', item_id='essay:global:1')
    two = api.request(messages, stage='bulk_attempt_2', item_id='essay:global:1')
    assert len(FakeAdapter.requests) == 2 and one['phase_call'] != two['phase_call']
    assert replay['replayed']
    assert all('seed' not in r and r['reasoning'] == {'effort': 'low'} and r['max_output_tokens'] == 1024
               for r in FakeAdapter.requests)
    with api.db() as db:
        db.execute("UPDATE calls SET status='pending',confirmed=0,reserved=.5 WHERE id=?", (one['phase_call'],))
    (api.output / 'requests' / f"{one['phase_call']:06}.json").unlink()
    assert len(api.settle_interrupted()) == 1
    assert api.accounting()['reserved_usd'] == .5 and api.accounting()['pending'] == 0
    with pytest.raises(RuntimeError, match='no regeneration'):
        api.request(messages, stage='bulk_attempt_1', item_id='essay:global:1')
    assert len(FakeAdapter.requests) == 2
    with api.db() as db:
        db.execute('UPDATE calls SET confirmed=12.24,reserved=0')
    with pytest.raises(CallBudgetExceeded):
        api.reserve('bulk_attempt_2', 'new', 'new', .53)


def interrupted_call(tmp_path, monkeypatch):
    """Simulate durable response followed by a crash before the DB/cache writes."""
    from verak.v3.eval import api as module
    monkeypatch.setattr(module.socket, 'getaddrinfo', lambda *args: [])
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path
    FakeAdapter.requests = []
    api = BulkAPI(config, 80000, phase=PHASE, adapter_factory=FakeAdapter)
    messages = [{'role': 'user', 'content': 'saved request'}]
    record = api.request(messages, stage='bulk_attempt_2', item_id='essay:korean:3')
    (api.output / 'cache' / (record['fingerprint'] + '.json')).unlink()
    with api.db() as db:
        db.execute("UPDATE calls SET status='pending',confirmed=0,reserved=?,path=NULL,finished=NULL WHERE id=?",
            (record['bound_usd'], record['phase_call']))
    return api, messages, deepcopy(record), api.output / 'requests' / f"{record['phase_call']:06}.json"


def test_completed_record_before_ledger_update_restores_cost_and_replays(tmp_path, monkeypatch):
    api, messages, record, path = interrupted_call(tmp_path, monkeypatch)
    outcomes = api.settle_interrupted()
    assert outcomes[0]['reconciliation'] == 'recovered_durable_record'
    account = api.accounting()
    assert account['confirmed_usd'] == pytest.approx(record['cost']['confirmed_usd'])
    assert account['reserved_usd'] == 0 and account['pending'] == 0 and account['calls'] == 1
    with api.db() as db:
        assert db.execute('SELECT status,path FROM calls').fetchone() == ('completed', str(path))
    replay = api.request(messages, stage='bulk_attempt_2', item_id='essay:korean:3')
    assert replay['raw'] == record['raw'] and replay['replayed']
    assert len(FakeAdapter.requests) == 1
    assert api.settle_interrupted() == []
    assert api.accounting()['confirmed_usd'] == pytest.approx(record['cost']['confirmed_usd'])


def test_completed_response_without_usage_keeps_billing_bound_while_replaying(tmp_path, monkeypatch):
    api, messages, record, path = interrupted_call(tmp_path, monkeypatch)
    record['usage'] = None
    record.pop('cost')
    write_json(path, record)
    api.settle_interrupted()
    account = api.accounting()
    assert account['confirmed_usd'] == 0 and account['reserved_usd'] == record['bound_usd']
    assert account['pending'] == 0
    assert api.request(messages, stage='bulk_attempt_2', item_id='essay:korean:3')['replayed']
    assert len(FakeAdapter.requests) == 1


@pytest.mark.parametrize('outcome', ['incomplete', 'timeout'])
def test_durable_failed_responses_preserve_status_and_never_resend(tmp_path, monkeypatch, outcome):
    api, messages, record, path = interrupted_call(tmp_path, monkeypatch)
    record['status'] = 'incomplete' if outcome == 'incomplete' else 'error'
    record['error_type'] = 'IncompleteResponse' if outcome == 'incomplete' else 'APITimeoutError'
    if outcome == 'timeout':
        for key in ('usage', 'cost', 'raw', 'response_id', 'response_model'):
            record.pop(key)
        record['timeout_reservation_usd'] = record['bound_usd']
    write_json(path, record)
    result = api.settle_interrupted()[0]
    assert result['status'] == record['status'] and result['path'] == str(path)
    account = api.accounting()
    assert account['pending'] == 0
    assert account['reserved_usd'] == (record['bound_usd'] if outcome == 'timeout' else 0)
    assert account['confirmed_usd'] == (0 if outcome == 'timeout' else record['cost']['confirmed_usd'])
    with pytest.raises(RuntimeError, match='no regeneration'):
        api.request(messages, stage='bulk_attempt_2', item_id='essay:korean:3')
    assert len(FakeAdapter.requests) == 1


@pytest.mark.parametrize('damage', ['missing', 'partial_json', 'partial_response', 'phase_call',
    'fingerprint', 'stage', 'item_id', 'model', 'payload', 'cost', 'usage_details'])
def test_unusable_durable_record_keeps_uncertain_reservation_without_send(tmp_path, monkeypatch, damage):
    api, messages, record, path = interrupted_call(tmp_path, monkeypatch)
    if damage == 'missing':
        path.unlink()
    elif damage == 'partial_json':
        path.write_text('{"status":')
    else:
        if damage == 'partial_response':
            record.pop('raw')
        elif damage == 'phase_call':
            record['phase_call'] += 1
        elif damage == 'payload':
            record['messages'][0]['content'] = 'different message'
        elif damage == 'cost':
            record['cost']['confirmed_usd'] += 1
        elif damage == 'usage_details':
            record['usage']['input_tokens_details'] = 'malformed'
        else:
            record[damage] = 'mismatched'
        write_json(path, record)
    result = api.settle_interrupted()[0]
    assert result['status'] == 'interrupted_unconfirmed' and result['path'] is None
    account = api.accounting()
    assert account['confirmed_usd'] == 0 and account['reserved_usd'] == record['bound_usd']
    assert account['pending'] == 0 and account['calls'] == 1
    with pytest.raises(RuntimeError, match='no regeneration'):
        api.request(messages, stage='bulk_attempt_2', item_id='essay:korean:3')
    assert len(FakeAdapter.requests) == 1


def test_policy_token_limits_fail_visibly_before_or_after_paid_response():
    class Tokenizer:
        def __init__(self, prefix, total):
            self.prefix, self.total = prefix, total
        def apply_chat_template(self, messages, **kwargs):
            return [1] * (self.prefix if kwargs.get('add_generation_prompt') else self.total)
    requests = []
    api = SimpleNamespace(model='gpt-6-luna', request=lambda *a, **k: requests.append(k) or {'raw': '{}'})
    args = {'episode_id': 'essay', 'role': 'global', 'turn': '1:0'}
    with pytest.raises(ValueError, match='Policy input'):
        BulkTeacher(api, 1, Tokenizer(7169, 7200)).generate([], **args)
    assert not requests
    with pytest.raises(ValueError, match='Generated turn'):
        BulkTeacher(api, 2, Tokenizer(7168, 8193)).generate([], **args)
    assert len(requests) == 1
    response = BulkTeacher(api, 2, Tokenizer(7168, 8192)).generate([], **args)
    assert response['policy_input_tokens'] == 7168 and response['policy_total_tokens'] == 8192


def test_parallel_reservations_never_overspend_cap(tmp_path):
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path
    config[PHASE]['max_cost_usd'] = .05
    api = BulkAPI(config, 80000, phase=PHASE, adapter_factory=FakeAdapter)
    def reserve_and_settle(i):
        try:
            call = api.reserve('bulk_attempt_1', str(i), str(i), .01)
        except CallBudgetExceeded:
            return False
        if call is None:
            return False
        accounting = api.accounting()
        assert accounting['confirmed_usd'] + accounting['reserved_usd'] <= .05 + 1e-12
        with api.db() as db:
            db.execute("UPDATE calls SET status='completed',reserved=0,confirmed=.008 WHERE id=?", (call,))
        return True
    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(reserve_and_settle, range(100)))
    account = api.accounting()
    assert sum(outcomes) in {5, 6}
    assert account['confirmed_usd'] <= .05 and account['reserved_usd'] == 0
    assert account['pending'] == 0
