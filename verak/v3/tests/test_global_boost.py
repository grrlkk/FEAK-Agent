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
