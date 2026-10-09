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
import json
import sqlite3
import time

from ..common import file_sha, pair_key, read_json, write_json
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
    threads = config.get('insertion_boost', {}).get('cpu_threads', 16)
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
    # Keep the exact frozen NF4/double-quantized weights. The AVX2 host has no
    # native BF16 matmul, so only their evaluation arithmetic becomes FP32.
    # Dequantization uses the original BF16 quantization state before conversion.
    model.dequantize()
    model = PeftModel.from_pretrained(model, str(config['paths']['scorer_adapter']),
        local_files_only=True, is_trainable=False)
    model = model.float()
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
    config['scorer'].update(gpu='cpu', execution_device='cpu', compute_dtype='float32',
        quantization='frozen_nf4_double_dequantized_from_bfloat16', cpu_loader_version='data_boost_cpu_nf4_fp32_v2',
        cpu_threads=threads)
    config['paths']['output'] = shared_root(config) / 'cpu_scorer/fp32_from_nf4'
    scorer = CPUScorer(config, model=model, tokenizer=tokenizer)
    return scorer


from ..score.kanana import KananaScorer


class CPUScorer(KananaScorer):
    """Same score contract; project vocabulary logits at only the eight targets."""
    def score(self, question, text, *, use_cache=True):
        from dataclasses import replace
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from ..common import sha_text
        from ..score.kanana import DIGIT_IDS, ScoreResult, expected_from_digit_logits, parse_first_line, teacher_positions
        if not question.strip() or not text.strip():
            raise ValueError('Question and essay must be nonblank')
        qhash = sha_text(question)
        if qhash not in self.genres:
            raise ValueError('Question requires a cached genre label before scoring')
        genre, key = self.genres[qhash]['genre'], pair_key(question, text)
        encoded = self.prepare_input(question, text)
        length = int(encoded['input_ids'].shape[1])
        if self.cache and use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return replace(hit, cache_hit=True, genre=genre)
        tokenizer = self.tokenizer
        class FirstNewline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.tensor(['\n' in tokenizer.decode(row[length:], skip_special_tokens=True)
                                     for row in input_ids], device=input_ids.device, dtype=torch.bool)
        with torch.inference_mode():
            generated = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1, num_return_sequences=1, temperature=None,
                top_p=None, top_k=None, stopping_criteria=StoppingCriteriaList([FirstNewline()]),
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            line = tokenizer.decode(generated[0, length:].tolist(), skip_special_tokens=True).split('\n', 1)[0].rstrip('\r')
            parse_first_line(line)
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            positions = teacher_positions(length, line_ids)
            ids = torch.cat([encoded['input_ids'], torch.tensor([line_ids], dtype=encoded['input_ids'].dtype)], dim=1)
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                logits_to_keep=torch.tensor(positions, dtype=torch.long))
            if output.logits.shape[1] != 8:
                raise ValueError('CPU scorer must retain exactly eight prediction positions')
            expected, integers = expected_from_digit_logits(output.logits[0, :, list(DIGIT_IDS)])
        result = ScoreResult(expected, integers, sum(expected)/8, genre, False, length, line, key)
        if self.cache:
            self.cache.put(result)
        return result


def serve(config):
    assert_cpu_process()
    shared = shared_root(config)
    root = shared / 'cpu_scorer'
    root.mkdir(parents=True, exist_ok=True)
    def stopping():
        marker = shared / 'report_complete.json'
        return (root / 'stop.json').exists() or (marker.exists() and read_json(marker).get('status') == 'complete')
    with (root / 'server.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        write_json(root / 'status.json', {'stage': 'loading', 'pid': os.getpid(), 'at': started,
            'gpu_used': False, 'affinity': sorted(os.sched_getaffinity(0))})
        try:
            with scorer_slot(shared):
                if config.get('insertion_boost', {}).get('cpu_backend') == 'bf16_semantic':
                    from .bf16_emulation import load
                    scorer = load(config)
                else:
                    scorer = load_cpu_scorer(config)
        except Exception as exc:
            write_json(root / 'status.json', {'stage': 'load_error', 'pid': os.getpid(),
                'at': time.time(), 'type': type(exc).__name__, 'message': str(exc), 'gpu_used': False})
            raise
        status = {'stage': 'ready', 'pid': os.getpid(), 'at': time.time(),
            'loaded_seconds': time.time()-started, 'fingerprint': scorer.fingerprint,
            'gpu_used': False, 'affinity': sorted(os.sched_getaffinity(0)), 'scored_requests': 0}
        write_json(root / 'status.json', status)
        if hasattr(scorer, 'execution_contract'):
            write_json(root / 'execution_contract.json', {**scorer.execution_contract, 'fingerprint': scorer.fingerprint})
        try:
            while not stopping():
                pending = [p for p in (root / 'requests').glob('*.json')
                           if not (root / 'responses' / p.name).exists()]
                policy = root / 'work_policy.json'
                if policy.exists() and 'allowed_keys' in read_json(policy):
                    allowed = set(read_json(policy)['allowed_keys'])
                    pending = [p for p in pending if p.stem in allowed]
                if not pending:
                    time.sleep(1)
                    continue
                # Re-read policy and the completion guard between every input;
                # a later higher-priority component must not wait for an old batch.
                for path in sorted(pending, key=lambda p: p.stat().st_mtime_ns)[:1]:
                    if stopping():
                        break
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


def queue_cpu(config, question, text, *, requester):
    """Nonblocking, idempotent CPU prefetch; never makes a model or API call."""
    root = shared_root(config) / 'cpu_scorer'
    key = pair_key(question, text)
    path = root / 'requests' / (key + '.json')
    if not path.exists():
        try:
            atomic_new(path, {'question': question, 'text': text, 'requester': requester,
                'requested_at': time.time(), 'gpu_used': False})
        except FileExistsError:
            pass
    return root / 'responses' / path.name


def score_cpu(config, question, text, *, requester, timeout=86400):
    """No paid calls or models in the caller; idempotent request/result artifacts."""
    root = shared_root(config) / 'cpu_scorer'
    response = queue_cpu(config, question, text, requester=requester)
    deadline = time.monotonic()+timeout
    while not response.exists():
        status = root / 'status.json'
        if status.exists() and read_json(status)['stage'] in {'load_error', 'stopped'}:
            raise RuntimeError('CPU scorer unavailable: ' + str(read_json(status)))
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU scorer request is preserved: ' + str(response))
        time.sleep(1)
    row = read_json(response)
    if row.get('error'):
        raise RuntimeError('CPU scorer: ' + str(row['error']))
    return {**row['result'], 'execution_device': 'cpu', 'scorer_fingerprint': row['fingerprint'],
            'cpu_seconds': row['seconds']}


_GPU_FINGERPRINT = {}


def gpu_reference_fingerprint(config):
    """Identity of the unchanged original GPU scorer recipe, without loading it."""
    from ..common import file_sha, sha_text
    from ..score.kanana import SCORER_VERSION, SYSTEM_PROMPT
    paths = config['paths']
    settings = {k: v for k, v in config['scorer'].items() if k not in ('average_k', 'average_seed')}
    identity = json.dumps({'settings': settings, 'base': str(paths['policy_base']),
        'adapter': str(paths['scorer_adapter'])}, sort_keys=True)
    if identity not in _GPU_FINGERPRINT:
        values = {'version': SCORER_VERSION, 'system': SYSTEM_PROMPT, 'settings': settings,
            'base': str(paths['policy_base']), 'adapter': str(paths['scorer_adapter'])}
        for name, path in [('adapter_weights', paths['scorer_adapter'] / 'adapter_model.safetensors'),
                           ('adapter_config', paths['scorer_adapter'] / 'adapter_config.json'),
                           ('base_config', paths['policy_base'] / 'config.json')]:
            if path.is_file():
                values[name] = file_sha(path)
        _GPU_FINGERPRINT[identity] = sha_text(json.dumps(values, sort_keys=True, ensure_ascii=False))
    return _GPU_FINGERPRINT[identity]


def cached_gpu_score(config, question, text):
    """Read existing GPU scores only when the full frozen fingerprint matches."""
    expected = gpu_reference_fingerprint(config)
    paths = config['paths']
    output = paths['repo'] / 'verak/v3/outputs'
    for relative in ('phase1', 'phase1b', 'phase3', 'phase3b/full/scorer'):
        path = output / relative / 'score_cache.sqlite'
        if not path.exists():
            continue
        try:
            with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True, timeout=.1) as db:
                fingerprint = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
                if fingerprint != (expected,):
                    continue
                saved = db.execute('SELECT value FROM scores WHERE key=?', (pair_key(question, text),)).fetchone()
        except sqlite3.OperationalError:
            continue  # Never make a live GPU-cache writer wait for this task.
        if saved:
            return {**json.loads(saved[0]), 'cache_hit': True, 'execution_device': 'saved_gpu_cache',
                'scorer_fingerprint': expected, 'cache_path': str(path), 'new_gpu_calls': 0}
    return None


def provisional_score(config, question, text, *, requester):
    """CPU approximation is diagnostic only; every final reward is re-scored on GPU."""
    hit = cached_gpu_score(config, question, text)
    if hit is not None:
        return {**hit, 'use': 'provisional_comparison', 'final_selection_authorized': False}
    result = score_cpu(config, question, text, requester=requester)
    contract = read_json(shared_root(config) / 'cpu_scorer/provisional_contract.json')
    expected = contract.get('fingerprint', contract.get('old_fingerprint'))
    if result['scorer_fingerprint'] != expected:
        raise ValueError('Provisional tables cannot mix FP32 and BF16 CPU fingerprints')
    return {**result, 'use': 'provisional_approximation', 'final_selection_authorized': False}


def gpu_reference_score(config, question, text):
    """Consume root-owned deferred GPU results; never load a GPU or call its service."""
    key = pair_key(question, text)
    path = shared_root(config) / 'gpu_rescore/responses' / (key+'.json')
    row = read_json(path)
    if row.get('execution_device') != 'gpu_reference' or row.get('error'):
        raise ValueError('GPU reference result is missing or invalid')
    if row['fingerprint'] != gpu_reference_fingerprint(config) or row['result']['cache_key'] != key:
        raise ValueError('GPU reference scorer or exact input identity changed')
    return {**row['result'], 'execution_device': 'gpu_reference', 'scorer_fingerprint': row['fingerprint'],
        'gpu_result_path': str(path), 'gpu_result_sha256': file_sha(path),
        'gpu_slot': row.get('slot'), 'new_gpu_calls_by_component': 0}


def score_available(config, question, text, *, requester):
    hit = cached_gpu_score(config, question, text)
    if hit:
        return hit
    root = shared_root(config) / 'cpu_scorer'
    path = root / 'selection_approval.json'
    while not path.exists() or not read_json(path).get('canonical_for_selection'):
        if path.exists() and read_json(path).get('terminal'):
            raise RuntimeError('CPU selection approval unavailable: '+str(read_json(path)))
        time.sleep(10)
    approval = read_json(path)
    result = score_cpu(config, question, text, requester=requester)
    if result['scorer_fingerprint'] != approval['fingerprint']:
        raise ValueError('A superseded CPU precision result cannot be used for SFT selection')
    return result


def stop_when_both_components_complete(config):
    root = shared_root(config)
    records = []
    for name in ('insertion', 'global'):
        path = root / name / 'complete.json'
        if not path.exists():
            return {'stopped': False, 'waiting_for': name}
        row = read_json(path)
        if row.get('status') not in {'complete', 'budget_stop'} or not all(
                row.get(key) for key in ('no_live_paid_calls', 'measurements_finished', 'stopped')):
            return {'stopped': False, 'waiting_for': name}
        records.append({'component': name, 'complete_path': str(path)})
    write_json(root / 'cpu_scorer/stop.json', {'reason': 'both data components finished',
        'components': records, 'at': time.time(), 'never_stop_rft': True})
    return {'stop_requested': True, 'components': records}
