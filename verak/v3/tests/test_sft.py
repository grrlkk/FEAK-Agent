"""Source holdouts and exact action loss, with no external services."""
import math

import pytest

from verak.v3.train.sft_data import source_split


def test_holdout_is_source_grouped_and_disjoint_across_roles():
    corpus, selections = {}, {'global': {}, 'korean': {}}
    for source in range(100):
        for variant in range(3):
            key = f'{source}:{variant}'
            corpus[key] = {'source_id': str(source)}
            for role, allowed in [('global', source < 80), ('korean', source >= 10)]:
                if allowed:
                    selections[role][key] = {}
    split = source_split(selections, corpus)
    assert split == source_split(selections, dict(reversed(list(corpus.items()))))
    train, validation = set(), set()
    for role, n in [('global', 80), ('korean', 90)]:
        data = split['roles'][role]
        assert len(data['validation_sources']) == math.ceil(.05 * n)
        assert len(data['validation_ids']) == 3 * len(data['validation_sources'])
        assert set(data['train_ids']) | set(data['validation_ids']) == set(selections[role])
        train.update(data['train_sources'])
        validation.update(data['validation_sources'])
    assert not train & validation


def test_sparse_loss_and_gradients_equal_full_masked_llama_loss():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from verak.v3.train.sft_train import action_loss
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0., attn_implementation='eager'))
    tokens = torch.tensor([[1, 3, 7, 9, 2, 15, 4, 6]])
    labels = tokens.clone()
    labels[:, :5] = -100
    batch = {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens), 'labels': labels}
    dense = model(**batch).loss
    dense.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    sparse, output = action_loss(model, batch)
    assert output.logits.shape == (1, 3, 37)
    torch.testing.assert_close(sparse, dense)
    sparse.backward()
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], atol=1e-6, rtol=1e-5)
    summed, _ = action_loss(model, batch, num_items_in_batch=6)
    torch.testing.assert_close(summed, dense / 2)


def test_sparse_loss_rejects_missing_targets_or_multi_sample_batch():
    import torch
    from verak.v3.train.sft_train import target_positions
    with pytest.raises(ValueError):
        target_positions(torch.full((1, 5), -100))
    with pytest.raises(ValueError):
        target_positions(torch.full((2, 5), -100))


def test_current_action_masks_three_turn_observations_notices_and_old_actions():
    from verak.v3.tests.test_phase7 import TinyTemplate
    from verak.v3.train.formatting import encode_labels
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'profile'},
                {'role': 'assistant', 'content': 'first action'}, {'role': 'user', 'content': 'tool output'},
                {'role': 'assistant', 'content': 'second action'}, {'role': 'user', 'content': 'final notice'},
                {'role': 'assistant', 'content': 'STOP'}]
    sample = encode_labels(TinyTemplate(), messages, last_assistant_only=True)
    assert ''.join(chr(x) for x in sample['labels'] if x != -100) == 'STOP[END]'


@pytest.mark.parametrize('suffix', ['luna', 'sol'])
def test_evaluation_api_budgets_are_separate_three_dollar_caps(tmp_path, suffix):
    from feak_tc.runtime.openai import CallBudgetExceeded
    from verak.v3.train.sft_eval import eval_config
    from verak.v3.train.teacher_bulk import BulkAPI
    config = eval_config()
    key = 'phase7_sft_' + suffix
    config['paths'][key + '_output'] = tmp_path / suffix
    api = BulkAPI(config, 10, phase=key)
    api.reserve('test', 'one', 'a', .1)
    with api.db() as db:
        db.execute("UPDATE calls SET status='completed',confirmed=2.95,reserved=0")
    with pytest.raises(CallBudgetExceeded):
        api.reserve('test', 'two', 'b', .051)
    assert api.accounting()['confirmed_usd'] == 2.95
    assert not (tmp_path / ('sol' if suffix == 'luna' else 'luna')).exists()


def test_reporting_keeps_malformed_and_rejected_actions_in_denominator():
    from verak.v3.train.sft_report import summarize
    row = {'completed': False, 'corpus_episode_id': 'test', 'reward': None,
           'steps': {'global': 2, 'korean': 0}, 'termination': {'global': 'errors'},
           'actions_by_role': {'global': [{'action': 'EDIT', 'args': {}, 'valid': False},
                                        {'action': 'STOP', 'args': {}, 'valid': True}]},
           'calls': [{'valid_json_action': False}, {'valid_json_action': True}],
           'runtime_error': {'type': 'Failure'}}
    data = summarize([row], intended=2)
    assert data['unattempted'] == 1 and data['completion_rate'] == 0
    assert data['valid_action_rate'] == .5 and data['protocol_valid_rate'] == .5
    assert data['reward']['combined']['R']['n'] == 0
    assert data['action_counts_attempted']['malformed_EDIT'] == 1
