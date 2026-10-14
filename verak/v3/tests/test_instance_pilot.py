"""Phase 3b operator restrictions, reversible edits, sampling and API boundaries."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from verak.src.schemas import Token
from verak.v3.common import load_config, read_json
from verak.v3.phase2 import restore_profile
from verak.v3.corrupt.api import QC_PROMPT
from verak.v3.corrupt.document import BareunBank, Document
from verak.v3.corrupt.instance_policy import (ACTIVE_LEVELS, POLICY, POLARITY_PAIRS,
                                            back_reference, candidates)
from verak.v3.corrupt.instance_pilot import (PilotAPI, filter_row, stratified_pilot, summarize, usage_cost)
from verak.v3.corrupt.operators import (apply, candidates as old_candidates,
    exact_restoration_satisfies, restore_record)
from verak.v3.tests.test_corruptions import sample as legacy_sample


@pytest.fixture
def sample(tmp_path):
    saved = read_json(Path(__file__).with_name('fixtures') / 'instance_bareun.json')
    config = load_config()
    bank = BareunBank(config, cache_dir=tmp_path)
    bank.memory = {text: [Token(**t) for t in tokens] for text, tokens in saved['tokens'].items()}
    def tokens(text):
        if text not in bank.memory:
            raise ValueError('No recorded synthetic observation')
        return deepcopy(bank.memory[text])
    bank.tokens = tokens
    return Document.from_profile(saved['text'], restore_profile(saved['profile'])), bank


@pytest.mark.parametrize('op', ['L_POLARITY', 'G_DELETE_SUPPORT'])
def test_changed_operator_apply_record_restore(sample, op):
    doc, bank = sample
    proposal = candidates(doc, op)[0]
    changed, record = apply(doc, proposal, bank)
    assert changed.text != doc.text
    restored = restore_record(changed, record, bank)
    assert restored.snapshot() == doc.snapshot()
    assert exact_restoration_satisfies(restored, record)
    assert not exact_restoration_satisfies(changed, record)


def test_polarity_only_requested_constructions(sample, legacy_sample):
    doc, _ = sample
    proposals = candidates(doc, 'L_POLARITY')
    assert len(proposals) == 2
    assert all(POLARITY_PAIRS[p.params['old']] == p.replacement for p in proposals)
    old, _, _ = legacy_sample
    assert old_candidates(old, 'L_POLARITY')  # Existing obligation fixture is retained.
    assert candidates(old, 'L_POLARITY') == []


def test_deletion_requires_next_back_reference(sample):
    doc, _ = sample
    values = candidates(doc, 'G_DELETE_SUPPORT')
    assert values and len(values) < len(old_candidates(doc, 'G_DELETE_SUPPORT'))
    for p in values:
        units = doc.units
        index = next(i for i, u in enumerate(units) if u.sid == p.sids[0])
        assert units[index+1].sid == p.params['next_sentence_id']
        assert back_reference(units[index+1].text)
        assert doc.locate(p.sids[0])[1] > 0


@pytest.mark.parametrize('text,expected', [('이처럼 해결한다.', '이처럼'), ('이러한 과정이다.', '이러한'),
    ('이와같이 본다.', '이와 같이'), ('이 때문에, 어렵다.', '이 때문에'), ('따라서 좋다.', '따라서'),
    ('그래서 좋다.', '그래서'), ('문장 안에서 따라서', None), ('따라서는 다른 말이다.', None),
    ('그러므로 좋다.', None)])
def test_back_reference_closed_set(text, expected):
    assert back_reference(text) == expected


@pytest.mark.parametrize('op', sorted(set(ACTIVE_LEVELS)-{'L_POLARITY','G_DELETE_SUPPORT'}))
def test_other_operators_unchanged(legacy_sample, op):
    doc, _, _ = legacy_sample
    assert candidates(doc, op) == old_candidates(doc, op)


@pytest.mark.parametrize('op', ['G_VAGUE', 'L_TRANSLATIONESE'])
def test_removed_operators_unavailable(sample, op):
    with pytest.raises(ValueError, match='removed'):
        candidates(sample[0], op)
    assert op not in POLICY.operators


def make_pool():
    return [{'episode_id': f'{split}:{op}:{level}:{i}', 'source_id': f'{split}:{op}:{i}',
        'split': split, 'level': level, 'records': [{'record_id': f'{op}:{i}', 'op': op, 'level': damage}]}
        for split in ('agent_train','agent_dev') for op, damage in ACTIVE_LEVELS.items()
        for level in (('L3','L4') if damage == 'GLOBAL' else ('L1','L2')) for i in range(6)]


def test_seeded_stratified_sampling_covers_operators_levels_splits():
    pool = make_pool()
    sample, plan = stratified_pilot(pool)
    again, again_plan = stratified_pilot(list(reversed(pool)))
    assert sample == again and plan == again_plan
    assert len(sample) == len({r['episode_id'] for r in sample}) == 100
    assert {rec['op'] for row in sample for rec in row['records']} == set(ACTIVE_LEVELS)
    assert {r['level'] for r in sample} == {'L1','L2','L3','L4'}
    assert {r['split'] for r in sample} == {'agent_train','agent_dev'}
    for h, values in plan['strata'].items():
        assert sum(r['sample_stratum'] == h for r in sample) == values['sample']
        assert values['weight']*values['sample'] == pytest.approx(values['population'])


def test_sampling_cannot_silently_omit_operator():
    with pytest.raises(ValueError, match='ten active'):
        stratified_pilot([r for r in make_pool() if r['records'][0]['op'] != 'L_POLARITY'])


def judgment_row(sid='x', values=((True,True),(True,True)), split='agent_train', stratum='train'):
    records = [{'record_id': f'{sid}:R{i}', 'op': op, 'level': ACTIVE_LEVELS[op]}
               for i, op in enumerate(('L_CONN','L_REGISTER')[:len(values)])]
    return {'episode_id': sid, 'source_id': sid, 'records': records, 'split': split, 'level':'L2',
        'sample_stratum': stratum, 'llm_judgment': {'judgments': [
            {'record_id': rec['record_id'], 'damage_real': fields[0], 'original_is_fix': fields[1], 'note':''}
            for rec, fields in zip(records, values)]}}


@pytest.mark.parametrize('fields', [(True,False),(False,True),(False,False)])
def test_one_failed_record_rejects_whole_essay(fields):
    row = filter_row(judgment_row(values=((True,True), fields)))
    assert list(row['record_kept'].values()) == [True,False]
    assert row['essay_kept'] is False
    assert filter_row(judgment_row())['essay_kept'] is True


def test_no_filtering_without_complete_response():
    row = judgment_row()
    row['llm_judgment']['judgments'].pop()
    with pytest.raises(ValueError, match='missing'):
        filter_row(row)
    row['llm_judgment'] = None
    with pytest.raises(ValueError, match='incomplete'):
        filter_row(row)


def test_cache_and_reasoning_not_double_counted():
    value = usage_cost({'input_tokens':1000, 'output_tokens':100,
        'input_tokens_details': {'cached_tokens':200, 'cache_write_tokens':600},
        'output_tokens_details': {'reasoning_tokens':80}})
    assert value['cost_usd_usage_estimate'] == pytest.approx(.00292)
    assert value['cost_usd_standard_input_output'] == pytest.approx(.003)
    with pytest.raises(ValueError, match='accounting'):
        usage_cost({'input_tokens':100, 'output_tokens':1, 'input_tokens_details':{'cache_write_tokens':101}})


def test_projection_uses_joint_essay_outcome_and_sampling_weights():
    sample = [judgment_row('a'), judgment_row('b', values=((True,True),(False,True))),
              judgment_row('c', split='agent_dev', stratum='dev'),
              judgment_row('d', split='agent_dev', stratum='dev')]
    plan = {'n':4, 'strata': {'train': {'population':20,'sample':2,'weight':10},
                             'dev': {'population':4,'sample':2,'weight':2}}}
    calls = [{'sample_id': r['episode_id'], 'usage': {'input_tokens':1000,'output_tokens':100}} for r in sample]
    result, filtered = summarize(sample, plan, calls)
    assert result['essay_yield']['overall']['yield'] == .75
    assert result['projections']['overall']['projected_total_kept'] == 14
    assert result['projections']['overall']['projected_WORD_records'] == 14  # No records from rejected essay.
    assert result['projections']['overall']['projected_TEXT_records'] == 14
    assert result['projections']['overall']['projected_remaining_cost_usd'] == pytest.approx(.06)
    assert result['record_yield']['operator']['L_CONN']['kept'] == 4
    assert result['record_yield']['operator']['L_CONN']['records_in_kept_essays'] == 3
    assert result['operator_gating'] is False
    assert result['api_usage']['cached_input_tokens'] == 0
    with pytest.raises(ValueError, match='exactly one'):
        summarize(sample, plan, calls+calls[:1])


def test_api_uses_same_prompt_and_shared_budget_before_request(tmp_path):
    config = load_config()
    config['paths']['phase3b_output'] = tmp_path
    config['paths']['phase3_output'] = tmp_path/'old_phase3'
    config['paths']['phase3_output'].mkdir()
    old_ledger = config['paths']['phase3_output']/'api_budget.json'
    old_ledger.write_text('{"phase":"v3_phase3_corruption","reserved_calls":84,"authorized_ceiling":100}')
    old_bytes = old_ledger.read_bytes()
    api = PilotAPI(config, 1)
    class Fake:
        def __init__(self, cfg, on_record):
            assert (cfg.model, cfg.reasoning_effort) == ('gpt-6.1-sol','high')
            self.callback = on_record
        def _load(self): pass
        def start_sample(self, sid): pass
        def close(self): pass
        def __call__(self, *, system, user, schema):
            assert system == QC_PROMPT and api.budget.used == 1
            self.callback({'usage':{'input_tokens':10,'output_tokens':10}})
            return {'judgments':[]}
    api.request('corruption_qc', 'one', {}, client_factory=Fake)
    assert PilotAPI(config, 100).budget.used == 1
    from feak_tc.runtime.openai import CallBudgetExceeded
    with pytest.raises(CallBudgetExceeded):
        api.request('corruption_qc', 'two', {}, client_factory=Fake)
    with pytest.raises(ValueError, match='Only pilot'):
        api.request('vague_generation', 'x', {})
    with pytest.raises(ValueError, match='ceiling'):
        PilotAPI(config, 101)
    assert old_ledger.read_bytes() == old_bytes
