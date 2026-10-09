from copy import deepcopy
from types import SimpleNamespace

import pytest

from verak.v3.rft1.rollout import request_settings
from verak.v3.rft1.selection import eligible, operator_evidence, rebalance


def episode():
    return {'completed': True, 'termination': {'global': 'STOP', 'korean': 'STOP'},
        'steps': {'global': 2, 'korean': 2},
        'reward': {role: {'R': .8, 'R_over': 0, 'per_record': [{'record_id': 'x', 'main': 1}]}
                   for role in ('global', 'korean')},
        'actions_by_role': {role: [{'action': 'STOP', 'valid': True}] for role in ('global', 'korean')}}


def test_explicit_round1_gate_rejects_step_limit_and_multiple_rejections():
    row, corpus = episode(), {'records': [{'record_id': 'x', 'level': 'GLOBAL', 'op': 'G_SENT_MOVE'}]}
    assert eligible(row, 'global', corpus)[0]
    row['termination']['global'] = 'step_limit'
    assert not eligible(row, 'global', corpus)[0]
    row['termination']['global'] = 'STOP'
    row['actions_by_role']['global'].insert(0, {'action': 'EDIT', 'valid': False})
    assert eligible(row, 'global', corpus)[0]
    row['actions_by_role']['global'].insert(0, {'action': 'EDIT', 'valid': False})
    assert not eligible(row, 'global', corpus)[0]
    row['reward']['korean']['R'] = .799999
    assert not eligible(row, 'korean', corpus)[0]


def test_partial_global_eligible_but_incomplete_korean_is_not():
    row, corpus = episode(), {'records': [{'record_id': 'x', 'level': 'GLOBAL', 'op': 'G_SENT_MOVE'}]}
    row['completed'] = False
    row['global_only_reward'] = row['reward']['global']
    row['reward'] = None
    assert eligible(row, 'global', corpus)[0]
    assert not eligible(row, 'korean', corpus)[0]


def test_confirmed_no_global_exception_keeps_stop_only_but_rejects_structural_attempts():
    row, corpus = episode(), {'records': [{'level': 'SENTENCE', 'op': 'L_CONJ', 'record_id': 'x'}]}
    row['reward']['global']['R'] = -.01
    assert eligible(row, 'global', corpus)[0]
    row['actions_by_role']['global'].insert(0, {'action': 'MOVE', 'valid': False})
    assert not eligible(row, 'global', corpus)[0]


def test_single_rejected_malformed_args_does_not_crash_post_run_selection():
    row, corpus = episode(), {'records': [{'level': 'SENTENCE', 'op': 'L_CONJ', 'record_id': 'x'}]}
    row['reward']['global']['R'] = -.02
    row['actions_by_role']['global'].insert(0, {'action': 'STOP', 'args': None, 'valid': False})
    assert eligible(row, 'global', corpus)[0]
    row['actions_by_role']['global'].pop(0)
    row['termination']['global'] = 'step_limit'
    assert not eligible(row, 'global', corpus)[0]


def test_stop_cap_is_applied_before_weight_two_and_does_not_compound():
    entries = [{'episode_id': str(i), 'STOP_only': i < 8, 'R': .9,
                'operators': {} if i < 8 else {'G_PARA_SWAP': {'all_fully_recovered': True},
                                              'G_SENT_MOVE': {'all_fully_recovered': True}}}
               for i in range(12)]
    before = deepcopy(entries)
    result, removed = rebalance(entries, 'global')
    assert entries == before
    assert len(result) == 6 and len(removed) == 6
    assert sum(e['STOP_only'] for e in result) / len(result) <= .35
    assert sum(e['weight'] for e in result) == 10
    assert all(e['weight'] == 2 for e in result if not e['STOP_only'])
    assert rebalance(entries, 'global') == (result, removed)


def test_operator_weight_requires_all_records_and_correct_role():
    row = episode()
    row['reward']['global']['per_record'].append({'record_id': 'y', 'main': .5})
    corpus = {'records': [{'record_id': i, 'level': 'GLOBAL', 'op': 'G_SENT_MOVE'} for i in ('x', 'y')]}
    ops = operator_evidence(row, 'global', corpus)
    assert ops['G_SENT_MOVE']['fully_recovered_records'] == 1
    assert not ops['G_SENT_MOVE']['all_fully_recovered']
    result, _ = rebalance([{'episode_id': 'x', 'STOP_only': False, 'R': .9, 'operators': ops}], 'global')
    assert result[0]['weight'] == 1
    result, _ = rebalance([{'episode_id': 'x', 'STOP_only': False, 'R': .9,
                           'operators': {'L_CONJ': {'all_fully_recovered': True}}}], 'korean')
    assert result[0]['weight'] == 2


def test_sampling_reproducible_and_distinct_for_four_requests():
    values = [request_settings('example:1', n) for n in range(1, 5)]
    assert len({v['seed'] for v in values}) == 4
    assert all(v['temperature'] == .7 and v['top_p'] == .95 for v in values)
    assert values == [request_settings('example:1', n) for n in range(1, 5)]
    assert request_settings('example:1', 1, evaluation=True)['temperature'] == 0


def test_two_rank_accumulated_gradient_equals_global_target_token_mean():
    import torch
    import torch.nn.functional as F
    from verak.v3.rft1.train import distributed_action_loss
    torch.manual_seed(7)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(6, 7, dtype=torch.float64))

        def forward(self, input_ids, attention_mask, logits_to_keep, use_cache):
            return SimpleNamespace(logits=self.logits[input_ids[0, logits_to_keep]].unsqueeze(0))

    reference = Toy()
    inputs = [dict(input_ids=torch.tensor([[0, 1, 2, 3, 4, 5]]),
                   attention_mask=torch.ones((1, 6), dtype=torch.long),
                   labels=torch.tensor([labels])) for labels in
              [[-100, -100, -100, 2, 3, 4], [-100, 1, -100, -100, -100, -100],
               [-100, -100, 2, 3, -100, -100], [-100, 2, 3, 4, 5, 6]]]
    total = sum(int((i['labels'] != -100).sum()) for i in inputs)
    full = sum(F.cross_entropy(reference.logits[i['input_ids'][0, :-1]], i['labels'][0, 1:],
                               ignore_index=-100, reduction='sum') for i in inputs) / total
    full.backward()
    rank_gradients = []
    for rank in range(2):
        model = deepcopy(reference)
        model.zero_grad()
        for i in inputs[rank::2]:
            loss, _ = distributed_action_loss(model, i, total, world_size=2)
            loss.backward()
        rank_gradients.append(model.logits.grad)
    assert torch.allclose(sum(rank_gradients) / 2, reference.logits.grad, rtol=1e-6, atol=1e-8)


def test_oneshot_filters_the_higher_combined_attempt_without_fallback():
    from verak.v3.rft1.oneshot import choose_target
    corpus = {'records': [{'record_id': 'x', 'op': 'G_SENT_MOVE', 'level': 'GLOBAL'}]}
    first, second = episode(), episode()
    for row, total in ((first, .95), (second, .9)):
        row['reward']['combined'] = {'R': total}
    first['reward']['korean']['R'] = .7
    chosen, reason = choose_target([(1, first), (2, second)], corpus)
    assert chosen[0] == 1
    assert reason == 'higher_combined_attempt_fails_KOREAN'
    first['reward']['korean']['R'] = .8
    assert choose_target([(1, first), (2, second)], corpus)[1] == 'eligible'


def test_oneshot_global_quality_gate_only_when_global_records_exist():
    from verak.v3.rft1.oneshot import choose_target
    row = episode()
    row['reward']['global']['R'] = -.01
    row['reward']['combined'] = {'R': .9}
    corpus = {'records': [{'record_id': 'x', 'op': 'L_CONJ', 'level': 'SENTENCE'}]}
    assert choose_target([(1, row)], corpus)[1] == 'eligible'
    corpus['records'][0].update(op='G_SENT_MOVE', level='GLOBAL')
    assert choose_target([(1, row)], corpus)[1] == 'higher_combined_attempt_fails_GLOBAL'


def test_oneshot_marker_case_uses_only_final_pair_and_reuses_judge_contract():
    from verak.v3.rft1.markers import oneshot_cases
    from verak.v3.observation.markers import contract
    def layout(ids):
        return {'paragraphs': [{'pid': 'P1', 'units': [{'sid': i, 'text': i + ' text'} for i in ids]}]}
    row = {'corpus_episode_id': 'real:1', 'initial_layout': layout(['S1', 'S2', 'S3']),
           'final_layout': layout(['S2', 'S1', 'S3']), 'completed': True}
    actions, cases = oneshot_cases(row)
    assert len(actions) == 1 and len(cases) == 3
    assert all(c['fields'] == ['conjunction', 'ending_style', 'subject_omission'] for c in cases)
    import json
    payload = json.loads(contract(next(c for c in cases if c['sid'] == 'S1'))[0][1]['content'])
    assert payload == {'previous_sentence': 'S2 text', 'affected_sentence': 'S1 text'}


@pytest.mark.skipif(__import__('os').environ.get('VERAK_DDP_CPU_TEST') != '1',
                    reason='Explicit CPU torchrun integration check needs localhost sockets')
def test_installed_trainer_two_rank_accumulation(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc-per-node=2', '-m', 'verak.v3.tests.ddp_target_loss_probe'],
        cwd=Path(__file__).resolve().parents[3], check=True, timeout=90,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'WANDB_DISABLED': 'true',
             'VERAK_DDP_PROBE_OUTPUT': str(tmp_path)})
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['same_update_as_global_token_mean'] and not result['cuda_used']
