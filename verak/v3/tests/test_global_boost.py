from copy import deepcopy
from types import SimpleNamespace

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.common import Example, sha_text, write_json
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v3.global_boost.config import PHASE, config_for
from verak.v3.global_boost.paid import BoostAPI
from verak.v3.global_boost.prepare import make_candidate, source_inventory
from verak.v3.phase2 import write_jsonl
from verak.v3.tests.test_teacher_bulk import FakeAdapter
from verak.v3.tests.test_environment import action, setup_env, StubParagraphAnalysis


def test_unused_sources_excluded_by_source_id_and_hash_and_preserve_phase3_policy(tmp_path, monkeypatch):
    from verak.v3.global_boost import prepare as module
    config = config_for()
    config['paths'].update(active_corrupt=tmp_path / 'active', phase3_output=tmp_path / 'phase3')
    examples = [Example(i, '문항', f'글{i}', sha_text('문항'), sha_text(f'글{i}'), '논증', False) for i in range(1, 6)]
    policy = {'quantile': 'q75', 'min_length': 500, 'max_length': 2500}
    monkeypatch.setattr(module, 'select_sources', lambda *_: (examples, policy))
    monkeypatch.setattr(module, 'excluded_sources', lambda *_: ({'valid:3': {'source_hash': examples[2].essay_hash}}, {}))
    write_jsonl(config['paths']['active_corrupt'] / 'agent_train.jsonl', [
        {'source_id': 'valid:1', 'source_hash': 'different'},
        {'source_id': 'other', 'source_hash': examples[1].essay_hash}])
    write_jsonl(config['paths']['active_corrupt'] / 'agent_dev.jsonl', [])
    write_json(config['paths']['phase3_output'] / 'sources_agent_train.json', {'rows': {
        e.id: {'essay_hash': e.essay_hash, 'eligible': e.source_line != 4, 'score': {'mean': 5.}}
        for e in examples}})
    pool, audit = source_inventory(config)
    assert [e.id for e, score in pool] == ['valid:5']
    assert audit['source_policy'] == policy
    assert audit['counts'] == {'phase3b_train_source_pool': 5, 'active_corpus_source_excluded': 2,
        'evaluation_source_excluded': 1, 'failed_source_score_excluded': 1, 'unused_eligible_sources': 1}


@pytest.mark.parametrize('operator', ['G_PARA_SWAP', 'G_SENT_MOVE'])
def test_practice_has_one_record_and_original_restoration_target(operator):
    from verak.v3.corrupt.document import BareunBank
    # No analyzer calls: moves only reuse immutable source tokens.
    source = Document([Paragraph(f'P{i}', [Unit(f'S{i*2-1}', f'첫 문장 {i}.', []),
        Unit(f'S{i*2}', f'다음 문장 {i}.', [], ' ')]) for i in range(1, 4)], ['', '\n\n', '\n\n'])
    example = Example(5, '문항', source.text, sha_text('문항'), sha_text(source.text), '논증', False)
    class Bank:
        def tokens(self, text):
            return []
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1] * 99)
    row, error = make_candidate(config_for(), example, source, {'mean': 6}, operator, Bank(), tokenizer)
    assert error is None and len(row['records']) == 1
    assert row['records'][0]['op'] == operator and row['records'][0]['level'] == 'GLOBAL'
    assert row['source_text'] == source.text and row['corrupted_text'] != source.text
    assert row['q_corrupted'] is None and row['method_version'] == 'v1'
    assert row['records'][0]['inverse'] == source.snapshot()
    again, _ = make_candidate(config_for(), example, source, {'mean': 6}, operator, Bank(), tokenizer)
    assert again == row


def test_sol_and_luna_share_cap_but_attempts_are_fresh_and_resume_replays(tmp_path, monkeypatch):
    from verak.v3.eval import api as module
    monkeypatch.setattr(module.socket, 'getaddrinfo', lambda *_: [])
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path
    original = deepcopy(config['env'])
    sol = BoostAPI(config, 20000, kind='sol', adapter_factory=FakeAdapter)
    luna = BoostAPI(config, 20000, kind='luna', adapter_factory=FakeAdapter)
    FakeAdapter.requests = []
    messages = [{'role': 'user', 'content': '동일 입력'}]
    qc = sol.request(messages, stage='global_boost_qc', item_id='one', effort='high', max_output=8192)
    first = luna.request(messages, stage='global_boost_attempt_1', item_id='same')
    second = luna.request(messages, stage='global_boost_attempt_2', item_id='same')
    assert len(FakeAdapter.requests) == 3
    assert luna.request(messages, stage='global_boost_attempt_1', item_id='same')['replayed']
    assert first['phase_call'] != second['phase_call']
    assert sol.accounting() == luna.accounting()
    assert sol.model == 'gpt-6.1-sol' and luna.model == 'gpt-6-luna'
    # Reconcile another model's durable request through the shared ledger.
    with sol.db() as db:
        db.execute("UPDATE calls SET status='pending',confirmed=0,reserved=.2 WHERE id=?", (qc['phase_call'],))
    assert luna.settle_interrupted()[0]['status'] == 'completed'
    with sol.db() as db:
        db.execute('UPDATE calls SET reserved=0,confirmed=4')
    with pytest.raises(CallBudgetExceeded):
        luna.reserve('global_boost_attempt_2', 'new', 'new', .01)
    assert config['env'] == original
    assert all('seed' not in request for request in FakeAdapter.requests)
    sol.close()
    luna.close()


def test_deferred_reward_preserves_exact_v1_turns_and_korean_permissions(setup_env):
    from verak.v3.env import RevisionEnv
    from verak.v3.global_boost.teacher import DeferredRewardEnv
    base, episode, _ = setup_env
    delayed = DeferredRewardEnv(base.config, analysis=StubParagraphAnalysis())
    assert DeferredRewardEnv.step is RevisionEnv.step
    assert DeferredRewardEnv.observe is RevisionEnv.observe
    assert base.reset(episode) == delayed.reset(episode)
    for value in [action('MOVE', target='P2', position='before:P1'),
                  action('STOP', summary='순서 변경'),
                  action('MOVE', target='P1', position='before:P2'),
                  action('EDIT', target='S1:첫', new_text='처음'),
                  action('STOP', summary='문장 표현 정리')]:
        left, right = base.step(value), delayed.step(value)
        assert left[:2] == right[:2]
        for row in (left[2]['action'], right[2]['action']):
            row.pop('elapsed_s')
        assert left[2] == right[2]
    assert base.handoff == delayed.handoff and base.final_text() == delayed.final_text()
    assert delayed.reward() is None


def test_global_only_measurement_never_scores_or_fabricates_korean_final(tmp_path, monkeypatch, setup_env):
    from verak.v3.common import read_json
    from verak.v3.corrupt.operators import apply, Proposal
    from verak.v3.global_boost import measure, resources
    from verak.v3.global_boost.teacher import attempt_path
    from verak.v3.reward.total import rewards
    base, episode, _ = setup_env
    source = episode['document']
    class Bank:
        def tokens(self, text):
            return []
    corrupted, record = apply(source, Proposal('G_PARA_SWAP', [u.sid for u in source.units],
                                              {'paragraph_indices': [0, 1]}), Bank())
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path
    row = {'episode_id': 'case', 'source_id': 'source', 'question': '문항', 'genre': '논증',
        'records': [record], 'corrupted_layout': corrupted.snapshot(), 'preexisting_spell_spans': []}
    raw = {'completed': True, 'generation_completed': True, 'reward': None,
        'stage1_layout': source.snapshot(), 'final_layout': {'not_restored': 'KOREAN final is unnecessary'},
        'actions_by_role': {'global': [], 'korean': [{'some': 'KOREAN action'}]}}
    path = attempt_path(tmp_path, 1, 'case')
    write_json(path, raw)
    monkeypatch.setattr(measure, 'prepare_teacher', lambda _: ({'orders': {'1': ['case'], '2': ['case']}}, {'case': row}))
    class Resources:
        analysis = SimpleNamespace(suspended=lambda: False)
        def __init__(self, _):
            pass
        def source(self, _):
            return source.clone()
        def restore(self, layout):
            return Document.restore(layout, Bank())
    calls = []
    def score(_, question, text):
        calls.append(text)
        return {'mean': 6.0 if text == corrupted.text else 6.5}
    monkeypatch.setattr(resources, 'CPUResources', Resources)
    monkeypatch.setattr(resources, 'score_cpu', score)
    measured = measure.run(config, limit=1)
    result = read_json(tmp_path / 'attempt_1/measured' / path.name)
    expected = rewards(source, corrupted, source, [record], genre='논증', q_corrupted=6.,
        q_stage1=6.5, q_final=9., config=config['reward'], mode='two_stage', stage1=source,
        stage1_actions=[], stage2_actions=[{'valid': True, 'action': 'STOP', 'changed_sids': []}])
    assert calls == [corrupted.text, source.text]
    assert result['global_only_reward'] == expected['global']
    assert result['reward'] is None and result['completed'] and result['generation_completed']
    assert result['unmeasured_roles'] == ['korean', 'combined']
    assert not measured['errors']


def test_controller_restart_refuses_any_other_process(tmp_path, monkeypatch):
    import os
    from verak.v3.global_boost.service import restart
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path
    write_json(tmp_path / 'controller_worker.json', {'pid': os.getpid()})
    monkeypatch.setattr(os, 'kill', lambda *_: pytest.fail('Foreign process must never be signalled'))
    with pytest.raises(RuntimeError, match='does not own'):
        restart(config)


def test_prefetch_only_saved_global_states_and_reuses_validated_gpu_cache(tmp_path, monkeypatch):
    from verak.v3.common import read_json
    from verak.v3.global_boost import prefetch
    config = config_for()
    root = tmp_path / 'global'
    config['paths'][PHASE + '_output'] = root
    source = Document([Paragraph('P1', [Unit('S1', '복원한 글.', [])])], [''])
    row = {'question': '문항', 'corrupted_text': '손상한 글.'}
    monkeypatch.setattr(prefetch, 'corpus', lambda _: {'case': row})
    monkeypatch.setattr(prefetch, 'cached_gpu_score', lambda config, question, text:
        {'mean': 6, 'execution_device': 'saved_gpu_cache'} if text == '복원한 글.' else None)
    path = root / 'attempt_1/episodes/case.json'
    write_json(path, {'corpus_episode_id': 'case', 'stage1_layout': source.snapshot(),
        'final_layout': {'unneeded': 'final'}, 'generation_completed': False})
    result = prefetch.enqueue(config)
    assert result == {'episodes_seen': 1, 'added_this_scan': 1}
    requests = list((tmp_path / 'cpu_scorer/requests').glob('*.json'))
    assert len(requests) == 1 and read_json(requests[0])['text'] == '손상한 글.'
    assert prefetch.enqueue(config)['added_this_scan'] == 0
    write_json(path, {'changed': 'raw teacher files are immutable'})
    with pytest.raises(ValueError, match='episode changed'):
        prefetch.enqueue(config)


def test_expansion_batches_share_atomic_cap_and_never_settle_other_live_namespace(tmp_path, monkeypatch):
    from verak.v3.eval import api as module
    from verak.v3.global_boost.expansion import batch_config
    monkeypatch.setattr(module.socket, 'getaddrinfo', lambda *_: [])
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path / 'global'
    cfg = batch_config(config, tmp_path / 'global/batches/batch_002')
    original = BoostAPI(config, 20000, adapter_factory=FakeAdapter)
    expansion = BoostAPI(cfg, 20000, kind='sol', adapter_factory=FakeAdapter)
    assert original.path == expansion.path
    original.reserve('global_boost_attempt_1', 'teacher:two_stage:global_boost:original', 'live', .01)
    assert expansion.settle_interrupted() == []
    assert expansion.accounting()['pending'] == 1
    with original.db() as db:
        db.execute("UPDATE calls SET status='completed',confirmed=10,reserved=0")
    expansion.qc_hold_usd = 1.9
    with pytest.raises(CallBudgetExceeded, match='preserve outstanding Luna'):
        expansion.reserve('global_boost_qc', 'global_boost:batch_002:one', 'new', .2)
    assert original.accounting()['confirmed_usd'] == 10
    original.close()
    expansion.close()


def test_expansion_round_robin_reuses_sources_with_distinct_one_record_practices():
    from collections import Counter
    from verak.v3.global_boost.expansion import choose_variants
    original, audit = {}, {'per_source': {}}
    for number in (1, 2):
        source_id = f'valid:{number}'
        audit['per_source'][source_id] = {}
        for operator in ('G_PARA_SWAP', 'G_SENT_MOVE'):
            key = f'{source_id}:{operator}'
            original[key] = {'source_id': source_id, 'operator': operator, 'corrupted_hash': key+':old'}
            audit['per_source'][source_id][operator] = {'valid_proposals': [
                {'corrupted_hash': key + suffix} for suffix in (':old', ':new1', ':new2', ':new3')]}
    values = choose_variants(audit, original, original, {'G_PARA_SWAP': 4, 'G_SENT_MOVE': 3}, 97)
    assert len(values) == 7
    assert len({v['corrupted_hash'] for _, v, _ in values}) == 7
    assert not {v['corrupted_hash'] for _, v, _ in values} & {r['corrupted_hash'] for r in original.values()}
    assert Counter(r['source_id'] for r, _, _ in values if r['operator'] == 'G_PARA_SWAP') == {'valid:1': 2, 'valid:2': 2}
    assert all(ordinal in (2, 3) for _, _, ordinal in values)


def test_unapproved_fp32_measurements_are_not_reused_for_canonical_selection(tmp_path):
    from verak.v3.global_boost.aggregate import approved
    from verak.v3.global_boost.measure import measured_path
    config = config_for()
    config['paths'][PHASE + '_output'] = tmp_path / 'global'
    raw = tmp_path / 'global/attempt_1/episodes/example.json'
    old = measured_path(config, 1, raw)
    write_json(old, {'cpu_scores': {'stage1': {'scorer_fingerprint': 'old_fp32'}}, 'global_only_reward': {'R': .81}})
    approval = tmp_path / 'cpu_scorer/selection_approval.json'
    write_json(approval, {'canonical_for_selection': False, 'fingerprint': 'old_fp32'})
    assert approved(config) is None
    write_json(approval, {'canonical_for_selection': True, 'fingerprint': 'new_bf16_fingerprint'})
    verified = approved(config)
    config[PHASE]['score_fingerprint'] = verified['fingerprint']
    assert measured_path(config, 1, raw) != old
    assert not measured_path(config, 1, raw).exists() and old.exists()
