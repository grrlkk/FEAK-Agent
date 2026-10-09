"""Single CPU NF4 scorer service for the two independent data boosts.

The frozen base/adapter, quantization, prompt, generation, digit extraction and
reward formulas are unchanged. Device-specific arithmetic is explicitly audited
against saved GPU results and has its own cache fingerprint, never the v1 cache.
Requests use local files so callers need no model import or service port.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import time

from ..common import pair_key, read_json, write_json
from ..train.teacher_bulk import atomic_new


def shared_root(config):
    return config['paths']['repo'] / 'verak/v3/outputs/data_boost'


def assert_cpu_process():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU scorer requires explicit CUDA_VISIBLE_DEVICES empty')
    import torch
    if torch.cuda.is_available():
        raise RuntimeError('CUDA must be invisible to data boost processes')


@contextmanager
def scorer_slot(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cpu_scorer.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def load_cpu_scorer(config):
    assert_cpu_process()
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from ..score.kanana import KananaScorer
    config = deepcopy(config)
    settings = config['scorer']
    threads = config.get('insertion_boost', {}).get('cpu_threads', 4)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    torch.use_deterministic_algorithms(True)
    dtype = torch.bfloat16
    if not settings['load_in_4bit']:
        raise ValueError('The frozen scorer recipe must remain NF4 double quantized')
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(config['paths']['policy_base']),
        local_files_only=True, quantization_config=quantization, device_map={'': 'cpu'},
        torch_dtype=dtype, attn_implementation='eager')
    model = PeftModel.from_pretrained(model, str(config['paths']['scorer_adapter']),
        local_files_only=True, is_trainable=False)
    if any(p.device.type != 'cpu' for p in model.parameters()) or any(
            p.device.type != 'cpu' for p in model.buffers()):
        raise RuntimeError('A scorer parameter or buffer escaped CPU')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['scorer_adapter']),
        local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config['scorer'].update(gpu='cpu', execution_device='cpu', compute_dtype='bfloat16',
        quantization='nf4_double', cpu_loader_version='data_boost_cpu_nf4_v1',
        cpu_threads=threads)
    config['paths']['output'] = shared_root(config) / 'cpu_scorer'
    scorer = KananaScorer(config, model=model, tokenizer=tokenizer)
    return scorer


def serve(config):
    assert_cpu_process()
    shared = shared_root(config)
    root = shared / 'cpu_scorer'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'server.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        write_json(root / 'status.json', {'stage': 'loading', 'pid': os.getpid(), 'at': started,
            'gpu_used': False, 'affinity': sorted(os.sched_getaffinity(0))})
        try:
            with scorer_slot(shared):
                scorer = load_cpu_scorer(config)
        except Exception as exc:
            write_json(root / 'status.json', {'stage': 'load_error', 'pid': os.getpid(),
                'at': time.time(), 'type': type(exc).__name__, 'message': str(exc), 'gpu_used': False})
            raise
        status = {'stage': 'ready', 'pid': os.getpid(), 'at': time.time(),
            'loaded_seconds': time.time()-started, 'fingerprint': scorer.fingerprint,
            'gpu_used': False, 'affinity': sorted(os.sched_getaffinity(0)), 'scored_requests': 0}
        write_json(root / 'status.json', status)
        try:
            while not (root / 'stop.json').exists():
                pending = [p for p in (root / 'requests').glob('*.json')
                           if not (root / 'responses' / p.name).exists()]
                if not pending:
                    time.sleep(1)
                    continue
                for path in sorted(pending, key=lambda p: p.stat().st_mtime_ns):
                    row = read_json(path)
                    if pair_key(row['question'], row['text']) != path.stem:
                        raise ValueError('CPU score request key mismatch')
                    before = time.time()
                    try:
                        with scorer_slot(shared):
                            result = scorer.score(row['question'], row['text']).to_dict()
                        response = {'result': result, 'fingerprint': scorer.fingerprint,
                            'seconds': time.time()-before, 'gpu_used': False}
                    except Exception as exc:
                        response = {'error': {'type': type(exc).__name__, 'message': str(exc)},
                            'fingerprint': scorer.fingerprint, 'seconds': time.time()-before, 'gpu_used': False}
                    atomic_new(root / 'responses' / path.name, response)
                    status.update(at=time.time(), scored_requests=status['scored_requests']+1,
                                  last_seconds=response['seconds'])
                    write_json(root / 'status.json', status)
        finally:
            scorer.close()
            write_json(root / 'status.json', {**status, 'stage': 'stopped', 'at': time.time()})


def score_cpu(config, question, text, *, requester, timeout=3600):
    """No paid calls or models in the caller; idempotent request/result artifacts."""
    root = shared_root(config) / 'cpu_scorer'
    key = pair_key(question, text)
    path = root / 'requests' / (key + '.json')
    if not path.exists():
        try:
            atomic_new(path, {'question': question, 'text': text, 'requester': requester,
                'requested_at': time.time(), 'gpu_used': False})
        except FileExistsError:
            pass
    response = root / 'responses' / path.name
    deadline = time.monotonic()+timeout
    while not response.exists():
        status = root / 'status.json'
        if status.exists() and read_json(status)['stage'] in {'load_error', 'stopped'}:
            raise RuntimeError('CPU scorer unavailable: ' + str(read_json(status)))
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU scorer request is preserved: ' + str(path))
        time.sleep(1)
    row = read_json(response)
    if row.get('error'):
        raise RuntimeError('CPU scorer: ' + str(row['error']))
    return {**row['result'], 'execution_device': 'cpu', 'scorer_fingerprint': row['fingerprint'],
            'cpu_seconds': row['seconds']}
