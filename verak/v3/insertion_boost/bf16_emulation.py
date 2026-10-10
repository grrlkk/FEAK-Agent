"""CPU implementation preserving NF4/BF16 and bitsandbytes LoRA rounding.

Only dense GEMM accumulation changes to CPU FP32. Inputs, frozen dequantized
weights and outputs keep BF16 rounding. Original PEFT bitsandbytes wrappers stay
in place, including their BF16 rounding of the LoRA branch before residual add.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from ..common import file_sha, pair_key, read_json, write_json
from ..train.teacher_bulk import atomic_new
from .cpu_score import CPUScorer, assert_cpu_process, shared_root


def make_linear(weight, bias=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    if weight.dtype != torch.bfloat16:
        raise ValueError('Emulation requires the original BF16-rounded frozen weights')
    class BF16Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(weight.detach(), requires_grad=False)
            self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None
            self.in_features, self.out_features = weight.shape[1], weight.shape[0]
            self.register_buffer('weight_fp32', weight.detach().float(), persistent=False)
            self.register_buffer('bias_fp32', bias.detach().float() if bias is not None else None, persistent=False)
        def forward(self, x):
            dtype = x.dtype
            rounded = x.to(torch.bfloat16).float()
            return F.linear(rounded, self.weight_fp32, self.bias_fp32).to(torch.bfloat16).to(dtype)
    return BF16Linear()


def load(config):
    assert_cpu_process()
    import bitsandbytes as bnb
    import numpy as np
    import torch
    from peft import PeftModel
    from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    config = deepcopy(config)
    threads = config['insertion_boost']['cpu_threads']
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    np.random.seed(config['scorer']['seed'])
    torch.manual_seed(config['scorer']['seed'])
    torch.use_deterministic_algorithms(True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(str(config['paths']['policy_base']),
        local_files_only=True, quantization_config=quantization, device_map={'': 'cpu'},
        torch_dtype=torch.bfloat16, attn_implementation='eager')
    # Attach the adapter before replacing arithmetic, preserving its bnb wrapper.
    model = PeftModel.from_pretrained(base, str(config['paths']['scorer_adapter']),
        local_files_only=True, is_trainable=False)
    original_wrappers = sum(isinstance(m, PeftLinear4bit) for m in model.modules())
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, bnb.nn.Linear4bit):
            weight = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state)
            replacement = make_linear(weight, module.bias)
            replaced.append({'name': name, 'original': 'bnb.Linear4bit', 'shape': list(weight.shape)})
        elif isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.bfloat16:
            replacement = make_linear(module.weight, module.bias)
            replaced.append({'name': name, 'original': 'nn.Linear BF16', 'shape': list(module.weight.shape)})
        else:
            continue
        model.set_submodule(name, replacement)
    preserved_wrappers = sum(isinstance(m, PeftLinear4bit) for m in model.modules())
    if original_wrappers == 0 or original_wrappers != preserved_wrappers:
        raise ValueError('Original bitsandbytes PEFT rounding wrappers must be preserved')
    if any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()):
        raise ValueError('A slow CPU quantized dense layer remains')
    if any(p.device.type != 'cpu' for p in model.parameters()) or any(p.device.type != 'cpu' for p in model.buffers()):
        raise RuntimeError('CPU scorer parameter escaped CPU')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['scorer_adapter']),
        local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config['scorer'].update(gpu='cpu', execution_device='cpu',
        compute_dtype='bf16_inputs_weights_outputs_fp32_gemm', quantization='nf4_double_dequantized_bf16',
        peft_wrapper='original_bnb_4bit_bf16_residual_rounding',
        cpu_loader_version='data_boost_bf16_semantic_v3', cpu_threads=threads)
    config['paths']['output'] = shared_root(config) / 'cpu_scorer/bf16_semantic'
    scorer = CPUScorer(config, model=model, tokenizer=tokenizer)
    scorer.execution_contract = {'settings': config['scorer'], 'replaced_linears': replaced,
        'original_bnb_peft_wrappers': original_wrappers, 'preserved_bnb_peft_wrappers': preserved_wrappers,
        'embedding_dtype': str(model.get_input_embeddings().weight.dtype),
        'code_sha256': file_sha(Path(__file__)), 'gpu_used': False}
    return scorer


def prepare_probe(config):
    """Archive material FP32 drift; preserve all queued inputs and raw teachers."""
    root = shared_root(config) / 'cpu_scorer'
    archive = root / 'fp32_superseded'
    archive.mkdir(parents=True, exist_ok=True)
    old_calibration = root / 'calibration.json'
    calibration = read_json(old_calibration if old_calibration.exists() else archive / 'calibration.json')
    worst = max(calibration['comparisons'], key=lambda row: abs(row['mean_delta']))
    plan = read_json(root / 'calibration_plan.json')
    reference = next(row for row in plan['episodes'] if row['episode_id'] == worst['episode_id'])
    item = reference['states'][worst['state']]
    key = pair_key(reference['question'], item['text'])
    if not (archive / 'provenance.json').exists():
        for name in ('responses', 'calibration.json', 'calibration_progress.json', 'status.json'):
            path = root / name
            if path.exists():
                path.rename(archive / name)
        atomic_new(archive / 'provenance.json', {'reason': 'material FP32 score drift; not canonical for SFT selection',
            'calibration_sha256': file_sha(archive / 'calibration.json'), 'archived_at': time.time(),
            'old_fingerprint': worst['cpu']['scorer_fingerprint'], 'raw_teacher_data_untouched': True})
    write_json(root / 'selection_approval.json', {'canonical_for_selection': False,
        'reason': 'Validating BF16-semantic CPU replacement against saved GPU scores', 'terminal': False})
    write_json(root / 'work_policy.json', {'allowed_keys': [key], 'stage': 'worst_case_probe'})
    probe = {'episode_id': worst['episode_id'], 'state': worst['state'], 'request_key': key,
        'gpu_score': item['gpu_score'], 'fp32_score': worst['cpu'],
        'question': reference['question'], 'text': item['text']}
    write_json(root / 'bf16_probe_plan.json', probe)
    return {'request_key': key, 'episode_id': worst['episode_id'], 'state': worst['state'],
            'previous_max_abs_Q_delta': calibration['max_abs_mean_delta'], 'gpu_used': False}


def probe_result(config):
    root = shared_root(config) / 'cpu_scorer'
    plan = read_json(root / 'bf16_probe_plan.json')
    path = root / 'responses' / (plan['request_key']+'.json')
    if not path.exists():
        return {'status': 'pending'}
    response = read_json(path)
    if response.get('error'):
        result = {'status': 'error', 'error': response['error']}
    else:
        cpu, gpu = response['result'], plan['gpu_score']
        result = {'status': 'completed', 'episode_id': plan['episode_id'], 'state': plan['state'],
            'seconds': response['seconds'], 'gpu_mean': gpu['mean'], 'cpu_mean': cpu['mean'],
            'mean_delta': cpu['mean']-gpu['mean'], 'score_line_equal': cpu['score_line'] == gpu['score_line'],
            'gpu_line': gpu['score_line'], 'cpu_line': cpu['score_line'], 'fingerprint': response['fingerprint']}
    write_json(root / 'bf16_probe_result.json', result)
    return result
