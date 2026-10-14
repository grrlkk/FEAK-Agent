from pathlib import Path

import pytest

from verak.v3.common import file_sha, pair_key, read_json, write_json
from verak.v3.insertion_boost.cpu_score import assert_cpu_process, score_cpu
from verak.v3.insertion_boost.data import summary


def setup_data(tmp_path):
    config = {'paths': {'v2_ops_output': tmp_path}, 'v2_ops': {'operators': ['G_DEL_LINK']}}
    write_json(tmp_path / 'source_plan.json', {
        'plans': {'G_DEL_LINK': {'agent_train': [None]*200}},
        'prior_qc': {'passed': 96, 'judged': 270, 'source_yield': '96/380'},
        'prior_passing_train': [None]*78})
    return config


def test_new_gate_uses_judged_not_planned(tmp_path):
    result = summary(setup_data(tmp_path))
    assert result['decision'] == 'retain'
    assert result['combined']['qc_pass_rate'] == pytest.approx(96/270)
    assert result['new_train']['awaiting_construction'] == 200
    assert result['passing_train_records'] == 78


def test_missing_qc_remains_unknown_and_never_passes(tmp_path):
    config = setup_data(tmp_path)
    path = tmp_path / 'candidates/G_DEL_LINK/agent_train/a.json'
    write_json(path, {'episode_id': 'boost:a', 'operator': 'G_DEL_LINK'})
    result = summary(config)
    assert result['new_train']['unknown_qc'] == 1
    assert result['combined'] == {'passed': 96, 'judged': 270, 'qc_pass_rate': 96/270}
    write_json(tmp_path / 'qc/boost_a.json', {'candidate_sha256': file_sha(path),
        'passed': True, 'verdict': {'damage_real': True, 'original_is_fix': True, 'recoverable_from_essay': True}})
    assert summary(config)['passing_train_records'] == 79
    write_json(path, {'episode_id': 'boost:a', 'operator': 'G_DEL_LINK', 'mutated': True})
    with pytest.raises(ValueError, match='hash mismatch'):
        summary(config)


def test_cpu_client_uses_separate_idempotent_file_queue(tmp_path):
    config = {'paths': {'repo': tmp_path}}
    root = tmp_path / 'verak/v3/outputs/data_boost/cpu_scorer'
    key = pair_key('question', 'essay')
    write_json(root / 'responses' / (key+'.json'), {'result': {'mean': 7.2},
        'fingerprint': 'cpu-distinct', 'seconds': 1.0, 'gpu_used': False})
    result = score_cpu(config, 'question', 'essay', requester='test')
    assert result['execution_device'] == 'cpu'
    assert result['scorer_fingerprint'] == 'cpu-distinct'
    before = (root / 'requests' / (key+'.json')).read_bytes()
    score_cpu(config, 'question', 'essay', requester='another')
    assert (root / 'requests' / (key+'.json')).read_bytes() == before


def test_cpu_guard_refuses_any_visible_gpu_setting(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError, match='explicit CUDA_VISIBLE_DEVICES empty'):
        assert_cpu_process()


def test_sparse_cpu_logits_match_full_v1_scorer(tmp_path):
    from verak.v3.insertion_boost.cpu_score import CPUScorer
    from verak.v3.tests.test_score import Model, Tokenizer, make_scorer
    class SparseModel(Model):
        def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
            result = super().__call__(input_ids, attention_mask, use_cache)
            assert len(logits_to_keep) == 8
            result.logits = result.logits[:, logits_to_keep, :]
            return result
    reference, _ = make_scorer(tmp_path)
    config = dict(reference.config)
    config['paths'] = {**config['paths'], 'output': tmp_path / 'cpu'}
    cpu = CPUScorer(config, model=SparseModel(), tokenizer=Tokenizer())
    expected, observed = reference.score('question', 'essay'), cpu.score('question', 'essay')
    assert observed.to_dict() == expected.to_dict()
    reference.close()
    cpu.close()


def test_global_only_reward_keeps_sft_gate_without_korean_selection():
    from verak.v3.train.teacher_comparison import absolute_selection
    raw = {'completed': True, 'reward': None, 'global_only_reward': {'R': .81}}
    keep = absolute_selection(raw, {'records': [{'level': 'GLOBAL'}]})
    assert keep == {'global': True, 'korean': False, 'global_rule': 'GLOBAL_R_ge_0.80'}


@pytest.mark.parametrize('sent', [False, True])
def test_dns_reconciliation_requires_proof_of_no_send(tmp_path, sent):
    import sqlite3
    from verak.v3.insertion_boost.data import reconcile_unsent_dns_errors
    config = {'paths': {'v2_ops_output': tmp_path}}
    path = tmp_path / 'judge_batches/v2_retry_labels/a.json'
    write_json(path, {'status': 'error', 'error': 'gaierror: [Errno -3] Temporary failure in name resolution'})
    write_json(tmp_path / 'qc_plan.json', {'all_candidates': []})
    (tmp_path / 'api').mkdir()
    with sqlite3.connect(tmp_path / 'api/ledger.sqlite') as db:
        db.execute('CREATE TABLE calls (id INTEGER)')
        if sent:
            db.execute('INSERT INTO calls VALUES (1)')
    reconcile_unsent_dns_errors(config)
    assert path.exists() == sent
    if not sent:
        evidence = read_json(tmp_path / 'no_send_reconciliation/dns_preflight/evidence.json')
        assert evidence['ledger_calls'] == evidence['paid_calls_repeated'] == 0


def test_bf16_linear_emulation_preserves_input_and_output_rounding():
    import torch
    from torch.nn import functional as F
    from verak.v3.insertion_boost.bf16_emulation import make_linear
    torch.manual_seed(73)
    x = torch.randn(5, 32).bfloat16()
    weight = torch.randn(7, 32).bfloat16()
    bias = torch.randn(7).bfloat16()
    linear = make_linear(weight, bias)
    assert linear(x).dtype == torch.bfloat16
    assert torch.equal(linear(x), F.linear(x, weight, bias))
    assert torch.equal(linear.weight, weight)


def test_original_bnb_lora_wrapper_keeps_separate_bf16_residual_rounding():
    import bitsandbytes as bnb
    import torch
    from peft.tuners.lora.bnb import Linear4bit
    from verak.v3.insertion_boost.bf16_emulation import make_linear
    torch.manual_seed(79)
    base = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.bfloat16,
        compress_statistics=True, quant_type='nf4').to('cpu')
    adapter = Linear4bit(base, 'default', r=2, lora_alpha=4, lora_dropout=0)
    adapter.lora_B['default'].weight.data.normal_()
    x = torch.randn(3, 32).bfloat16()
    weight = bnb.functional.dequantize_4bit(base.weight.data, base.weight.quant_state).bfloat16()
    adapter.base_layer = make_linear(weight)
    residual = adapter.lora_B['default'](adapter.lora_A['default'](x.float()))*adapter.scaling['default']
    expected = adapter.base_layer(x)+residual.bfloat16()
    assert torch.equal(adapter(x), expected)
    assert isinstance(adapter, Linear4bit)


def test_superseded_cpu_fingerprint_cannot_select_training_data(tmp_path, monkeypatch):
    from verak.v3.insertion_boost import cpu_score
    config = {'paths': {'repo': tmp_path}}
    write_json(tmp_path / 'verak/v3/outputs/data_boost/cpu_scorer/selection_approval.json',
        {'canonical_for_selection': True, 'fingerprint': 'approved_bf16'})
    monkeypatch.setattr(cpu_score, 'cached_gpu_score', lambda *a, **k: None)
    monkeypatch.setattr(cpu_score, 'score_cpu', lambda *a, **k: {'scorer_fingerprint': 'superseded_fp32'})
    with pytest.raises(ValueError, match='superseded'):
        cpu_score.score_available(config, 'q', 'text', requester='test')
