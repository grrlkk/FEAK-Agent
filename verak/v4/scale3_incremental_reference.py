"""Root-dispatched GPU0 reference scoring of immutable, finished B3 attempts.

No collector imports this entry point. A fresh, exact 20-episode RFT1 replay is
mandatory before accepting the physical-GPU remap. Teacher files, rewards,
selection, prompts and training are never changed here. A watch invocation
scans at most once per 45 minutes, and exits after publishing the existing B3
GPU-complete contract. GPU1 is neither queried for idleness nor used.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from verak.v3.common import file_sha, pair_key, read_json, write_json
from verak.v3.train.teacher_bulk import atomic_new
from .scale2_projection import restore_document
from .scale3_reference import reference_lock, validate_manifest, validate_response

INTERVAL_SECONDS = 2700
CALIBRATION_EPISODES = 20
AUTHORIZATION = 'user_20261011_incremental_reference_and_v4_dryrun'
SCORE_FIELDS = ('expected', 'integers', 'mean', 'score_line', 'input_tokens')


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable incremental artifact changed: ' + str(path))
    else:
        atomic_new(path, value)
    return value


def snapshot(path, destination):
    """Copy a complete atomic teacher artifact without modifying its source."""
    path, destination = Path(path), Path(destination)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if file_sha(path) != digest:
        raise ValueError('Artifact changed during snapshot')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha(destination) != digest:
            raise ValueError('Previously snapshotted teacher artifact changed')
    else:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {'path': str(path), 'sha256': digest,
            'snapshot_path': str(destination), 'snapshot_sha256': digest}


def layout_text(layout):
    if len(layout['gaps']) != len(layout['paragraphs']):
        raise ValueError('Incomplete saved RFT1 layout')
    return ''.join(gap + ''.join(u['leading'] + u['text'] for u in p['units'])
                   for gap, p in zip(layout['gaps'], layout['paragraphs'])) + layout['tail']


def calibration_manifest(config, root):
    """Choose 20 distinct saved RFT1 sources, and all three actual endpoints."""
    repo = Path(config['paths']['repo'])
    rft = repo / 'verak/v3/outputs/phase8_rft1'
    design_path = rft / 'rollout_design.json'
    design = read_json(design_path)
    if file_sha(design['corpus']) != design['corpus_sha256']:
        raise ValueError('RFT1 calibration corpus changed')
    with Path(design['corpus']).open() as stream:
        corpus = {r['episode_id']: r for r in map(json.loads, stream)}
    sample = read_json(Path(root) / 'B3/sample.json')
    fingerprint = sample['reference_gpu_fingerprint']
    episodes, requests, sources = [], {}, set()
    for eid in design['order']:
        row = corpus[eid]
        if row['source_id'] in sources:
            continue
        path = rft / 'rollouts/sample_1/episodes' / (eid.replace(':', '_') + '.json')
        raw = read_json(path)
        if not raw['completed']:
            continue
        if (raw['source_id'] != row['source_id'] or raw['corpus_episode_id'] != eid or
                raw['rft1']['design_sha256'] != file_sha(design_path) or
                row['scorer_fingerprint'] != fingerprint):
            raise ValueError('RFT1 episode lacks the frozen GPU reference identity')
        expected = {}
        for call in raw['score_calls']:
            result = call['result']
            key = result['cache_key']
            if key in expected and any(expected[key][f] != result[f] for f in SCORE_FIELDS):
                raise ValueError('Saved RFT1 endpoint has conflicting GPU scores')
            expected[key] = result
        keys = {}
        for name, field in (('initial', 'initial_layout'), ('stage1', 'stage1_layout'), ('final', 'final_layout')):
            text = layout_text(raw[field])
            key = pair_key(row['question'], text)
            result = expected[key]
            validate_response({'fingerprint': fingerprint, 'execution_device': 'gpu_reference',
                               'result': result}, key, fingerprint)
            request = {'key': key, 'question': row['question'], 'text': text, 'expected': result}
            if key in requests and any(requests[key]['expected'][f] != result[f] for f in SCORE_FIELDS):
                raise ValueError('RFT1 calibration input has inconsistent scores')
            requests[key] = request
            keys[name] = key
        sources.add(row['source_id'])
        episodes.append({'episode_id': eid, 'source_id': row['source_id'], 'sample': 1,
                         'genre': row['genre'], 'level': row['level'], 'path': str(path),
                         'sha256': file_sha(path), 'quality_keys': keys})
        if len(episodes) == CALIBRATION_EPISODES:
            break
    if len(episodes) != CALIBRATION_EPISODES:
        raise ValueError('Need exactly 20 completed distinct-source RFT1 episodes')
    return {'schema_version': 1, 'reference_fingerprint': fingerprint,
            'selection': 'first completed distinct sources in frozen RFT1 design order, sample 1',
            'episode_count': len(episodes), 'distinct_sources': len(sources),
            'endpoint_references': sum(len(e['quality_keys']) for e in episodes),
            'episodes': episodes, 'requests': [requests[k] for k in sorted(requests)],
            'bindings': {str(design_path): file_sha(design_path), design['corpus']: design['corpus_sha256']},
            'levels': dict(Counter(e['level'] for e in episodes)),
            'genres': dict(Counter(e['genre'] for e in episodes)),
            'cache_enabled': False, 'score_use_cache': False, 'equality': 'exact, no tolerance'}


def preflight(config, root):
    """File-only contract/calibration freeze; this never imports a GPU backend."""
    root = Path(root)
    output = root / 'B3/gpu_rescore/incremental'
    auth_path = root / 'gpu_schedule_amendment.json'
    auth = read_json(auth_path)
    if auth.get('authorization') != AUTHORIZATION or auth.get('no_paid_calls') is not True:
        raise ValueError('The explicit GPU0 incremental authorization is absent')
    if config['scorer']['gpu'] != 1 or config['scorer'].get('average_k', 1) != 1:
        raise ValueError('Reference settings must remain the RFT1 single-score settings')
    sample_path, contract_path = root / 'B3/sample.json', root / 'B3/contract.json'
    sample, teacher_contract = read_json(sample_path), read_json(contract_path)
    if (sample['component'] != 'B3' or sample['collection_version'] != 'v4.3' or
            teacher_contract['sample_sha256'] != file_sha(sample_path) or
            file_sha(sample['corpus_path']) != sample['corpus_sha256']):
        raise ValueError('Frozen B3 sample/contract/corpus mismatch')
    calibration = calibration_manifest(config, root)
    freeze(output / 'calibration/manifest.json', calibration)
    repo = Path(config['paths']['repo'])
    oneshot_path = repo / 'verak/v3/outputs/oneshot_baseline/complete.json'
    oneshot = read_json(oneshot_path)
    if (oneshot.get('completed') is not True or oneshot['report'].get('completed') is not True or
            file_sha(oneshot['report']['report']) != oneshot['report']['report_sha256']):
        raise ValueError('Completed one-shot report provenance is required')
    files = [Path(__file__), Path(__file__).with_name('scale3_reference.py'),
             Path(__file__).with_name('scale2_projection.py')]
    v3 = Path(__file__).parents[1] / 'v3'
    files += [v3 / p for p in ('score/kanana.py', 'score/loading.py', 'config.yaml')]
    files += [sample_path, contract_path, auth_path, oneshot_path,
              Path(oneshot['report']['report']), output / 'calibration/manifest.json']
    value = {'schema_version': 1, 'version': 'v4.3-incremental-reference',
        'reference_fingerprint': sample['reference_gpu_fingerprint'],
        'reference_scorer_settings': deepcopy(config['scorer']),
        'execution_scorer_settings': {**config['scorer'], 'gpu': 0},
        'scorer_base': str(config['paths']['policy_base']),
        'scorer_adapter': str(config['paths']['scorer_adapter']),
        'physical_gpu': 0, 'local_gpu': 0, 'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'worker_threads': 8,
        'minimum_collection_scan_interval_seconds': INTERVAL_SECONDS,
        'incremental_scope': 'atomically saved completed B3 attempts only',
        'final_scope': 'all endpoints in the original collector-owned gpu_manifest, including completed stage1 of incomplete Korean attempts',
        'calibration_gate': '20 distinct RFT1 episodes, exact expected[8], integers[8], Q, score line and input tokens',
        'cache_enabled': False, 'score_use_cache': False, 'training': False, 'paid_calls': 0,
        'CPU_scoring_performed': False, 'collector_mutations': False,
        'bindings': {str(p): file_sha(p) for p in files}}
    freeze(output / 'contract.json', value)
    return {'status': 'prepared_no_GPU_calls', 'contract': str(output / 'contract.json'),
            'contract_sha256': file_sha(output / 'contract.json'),
            'calibration_episodes': calibration['episode_count'],
            'calibration_unique_inputs': len(calibration['requests'])}


def validate_contract(root):
    output = Path(root) / 'B3/gpu_rescore/incremental'
    contract = read_json(output / 'contract.json')
    for path, digest in contract['bindings'].items():
        if file_sha(path) != digest:
            raise ValueError('Incremental reference binding changed: ' + path)
    calibration = read_json(output / 'calibration/manifest.json')
    for path, digest in calibration['bindings'].items():
        if file_sha(path) != digest:
            raise ValueError('RFT1 reference input changed')
    for episode in calibration['episodes']:
        if file_sha(episode['path']) != episode['sha256']:
            raise ValueError('Saved RFT1 calibration episode changed')
    return contract


def ensure_gpu0_only():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or
            os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'):
        raise RuntimeError('Only physical GPU0, visible as local cuda:0, is authorized')
    # This deliberately excludes GPU1: another authorized job may occupy it.
    query = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    if query.stdout.strip():
        raise RuntimeError('Physical GPU0 is occupied; no process will be stopped')


class GPU0Scorer:
    """Actual gpu=0 fingerprint stays explicit alongside the legacy identity."""
    def __init__(self, config, contract):
        from verak.v3.score.kanana import KananaScorer
        import torch
        torch.set_num_threads(8)
        actual = deepcopy(config)
        actual['scorer']['gpu'] = 0
        if (config['scorer'] != contract['reference_scorer_settings'] or
                actual['scorer'] != contract['execution_scorer_settings']):
            raise ValueError('Loaded scorer settings differ from the approved contract')
        self.scorer = KananaScorer(actual, cache_enabled=False)
        # An injected-model constructor performs no second load or score. It
        # recomputes the original fingerprint from unchanged reference settings.
        reference = KananaScorer(config, model=self.scorer.model,
                                 tokenizer=self.scorer.tokenizer, cache_enabled=False)
        self.fingerprint = reference.fingerprint
        self.execution_fingerprint = self.scorer.fingerprint
        if self.fingerprint != contract['reference_fingerprint']:
            raise ValueError('Loaded model/config differs from the saved RFT1 scorer')
        if torch.cuda.device_count() != 1 or str(self.scorer.model.device) != 'cuda:0':
            raise RuntimeError('Scorer did not load exclusively on visible local GPU0')
        props = torch.cuda.get_device_properties(0)
        physical_uuid = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
            '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout.strip()
        actual_uuid = str(getattr(props, 'uuid', 'unavailable'))
        if actual_uuid == 'unavailable' or actual_uuid.removeprefix('GPU-') != physical_uuid.removeprefix('GPU-'):
            raise RuntimeError('Visible CUDA device UUID differs from physical GPU0')
        self.device = {'physical_gpu': 0, 'local_gpu': 0, 'name': props.name,
                       'uuid': actual_uuid, 'physical_gpu0_uuid': physical_uuid,
                       'CUDA_VISIBLE_DEVICES': '0', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}

    def score(self, question, text):
        return self.scorer.score(question, text, use_cache=False).to_dict()

    def close(self):
        import torch
        self.scorer.close()
        self.scorer = None
        torch.cuda.empty_cache()


def calibration_summary(manifest, responses):
    comparisons, errors = [], []
    for request in manifest['requests']:
        saved, value = request['expected'], responses[request['key']]
        if value.get('error'):
            errors.append({'key': request['key'], **value['error']})
            continue
        actual = value['result']
        validate_response(value, request['key'], manifest['reference_fingerprint'])
        if actual.get('cache_hit') is not False:
            raise ValueError('Calibration must be a fresh no-cache GPU call')
        differences = [abs(a-b) for a, b in zip(actual['expected'], saved['expected'])]
        comparisons.append({'key': request['key'], 'expected_abs_differences': differences,
            'Q_abs_difference': abs(actual['mean']-saved['mean']),
            'digit_agreement': [a == b for a, b in zip(actual['integers'], saved['integers'])],
            'exact_fields': {f: actual[f] == saved[f] for f in SCORE_FIELDS}})
    count = len(comparisons)
    return {'passed': not errors and count == len(manifest['requests']) and
                     all(all(c['exact_fields'].values()) for c in comparisons),
        'episode_count': manifest['episode_count'], 'distinct_sources': manifest['distinct_sources'],
        'endpoint_references': manifest['endpoint_references'], 'unique_inputs': len(manifest['requests']),
        'compared_inputs': count, 'errors': errors, 'comparisons': comparisons,
        'mean_abs_Q_difference': sum(c['Q_abs_difference'] for c in comparisons)/count if count else None,
        'max_abs_Q_difference': max((c['Q_abs_difference'] for c in comparisons), default=None),
        'max_abs_expected_difference': max((max(c['expected_abs_differences']) for c in comparisons), default=None),
        'per_rubric_digit_agreement': [sum(c['digit_agreement'][i] for c in comparisons)/count for i in range(8)] if count else None,
        'exact_equality_required': True, 'tolerance': 0}


def score_response(scorer, request, *, calibration_sha=None):
    before = time.time()
    value = {'fingerprint': scorer.fingerprint, 'execution_fingerprint': scorer.execution_fingerprint,
        'execution_device': 'gpu_reference', 'gpu_id': 0, 'physical_gpu': 0, 'local_gpu': 0,
        'device': scorer.device, 'slot': 'post_oneshot_evaluation', 'scheduled_by': 'root',
        'incremental': True, 'calibration_sha256': calibration_sha, 'cache_disabled': True}
    try:
        value['result'] = scorer.score(request['question'], request['text'])
        if value['result'].get('cache_hit') is not False:
            raise ValueError('Scorer unexpectedly returned a cached value')
        validate_response(value, request['key'], scorer.fingerprint)
    except Exception as exc:
        value.pop('result', None)
        value['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    value['seconds'] = time.time()-before
    return value


def calibrate(output, scorer, stop):
    path = output / 'calibration'
    manifest_path = path / 'manifest.json'
    manifest = read_json(manifest_path)
    result_path = path / 'complete.json'
    responses = {}
    for request in manifest['requests']:
        destination = path / 'responses' / (request['key'] + '.json')
        if not destination.exists():
            stop()
            atomic_new(destination, score_response(scorer, request))
        response = read_json(destination)
        if (response['execution_fingerprint'] != scorer.execution_fingerprint or
                response.get('physical_gpu') != 0 or response.get('local_gpu') != 0):
            raise ValueError('Calibration response belongs to another execution')
        responses[request['key']] = response
    summary = calibration_summary(manifest, responses)
    result = {**summary, 'manifest_sha256': file_sha(manifest_path),
        'reference_fingerprint': scorer.fingerprint, 'execution_fingerprint': scorer.execution_fingerprint,
        'device': scorer.device, 'fresh_output_root': str(path / 'responses'),
        'responses_sha256': {k: file_sha(path / 'responses' / (k + '.json')) for k in responses},
        'cache_enabled': False, 'score_use_cache': False, 'CPU_scoring_performed': False,
        'new_B3_scores_allowed': summary['passed']}
    freeze(result_path, result)
    if not summary['passed']:
        raise RuntimeError('Exact RFT1/GPU0 calibration failed; no B3 scoring is permitted')
    return file_sha(result_path)


def saved_calibration(output, contract):
    """Revalidate immutable fresh-call evidence without loading a model."""
    path = output / 'calibration/complete.json'
    if not path.exists():
        return None
    result = read_json(path)
    manifest_path = output / 'calibration/manifest.json'
    if (result['manifest_sha256'] != file_sha(manifest_path) or
            result['reference_fingerprint'] != contract['reference_fingerprint'] or
            result['device']['physical_gpu'] != 0 or result['device']['local_gpu'] != 0):
        raise ValueError('Saved calibration identity changed')
    responses = {}
    for key, digest in result['responses_sha256'].items():
        response_path = output / 'calibration/responses' / (key + '.json')
        if file_sha(response_path) != digest:
            raise ValueError('Saved fresh calibration response changed')
        value = read_json(response_path)
        if (value['execution_fingerprint'] != result['execution_fingerprint'] or
                value['physical_gpu'] != 0 or value['local_gpu'] != 0 or
                value.get('cache_disabled') is not True):
            raise ValueError('Saved calibration execution provenance changed')
        responses[key] = value
    summary = calibration_summary(read_json(manifest_path), responses)
    if any(result[k] != v for k, v in summary.items()):
        raise ValueError('Saved calibration summary differs from actual responses')
    if not summary['passed']:
        raise RuntimeError('Exact RFT1/GPU0 calibration failed; no B3 scoring is permitted')
    return result


def endpoint_plan(raw_path, root, *, final=False):
    """Save exact provenance and input endpoints of an atomically completed file."""
    raw_path, root = Path(raw_path), Path(root)
    raw_bytes = raw_path.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    if not final and (raw.get('completed') is not True or raw.get('status') != 'completed'):
        return None
    sample = read_json(root / 'B3/sample.json')
    teacher_contract = read_json(root / 'B3/contract.json')
    if (raw['corpus_episode_id'] not in sample['episode_ids'] or raw['attempt'] not in (1, 2) or
            raw['prompt_version'] != teacher_contract['prompt_version']):
        raise ValueError('Unexpected B3 completed-attempt identity')
    expected_name = raw['corpus_episode_id'].replace(':', '_') + '_a' + str(raw['attempt']) + '.json'
    if raw_path.name != expected_name:
        raise ValueError('B3 saved filename and episode/attempt identity differ')
    prepared_path = Path(raw['prepared_path'])
    if file_sha(prepared_path) != raw['prepared_sha256']:
        raise ValueError('B3 prepared evidence changed')
    prepared = read_json(prepared_path)
    if (prepared['corpus']['episode_id'] != raw['corpus_episode_id'] or
            prepared['corpus']['source_id'] != raw['source_id'] or
            prepared['corpus']['scorer_fingerprint'] != sample['reference_gpu_fingerprint']):
        raise ValueError('B3 preparation/episode identity mismatch')
    projected = raw['reward_projection']['states']
    needed = ['initial', 'stage1'] if 'stage1' in projected else []
    if needed and raw['completed'] and 'final' in projected:
        needed.append('final')
    requests, keys = {}, {}
    for name in needed:
        text = restore_document(raw['states'][name]['document']).text
        question = prepared['corpus']['question']
        key = pair_key(question, text)
        requests[key] = {'key': key, 'question': question, 'text': text}
        keys[name] = key
    directory = root / 'B3/gpu_rescore/incremental'
    raw_proof = snapshot(raw_path, directory / 'snapshots/raw' / raw_path.name)
    prepared_proof = snapshot(prepared_path, directory / 'snapshots/prepared' / prepared_path.name)
    if raw_proof['sha256'] != raw_sha or prepared_proof['sha256'] != raw['prepared_sha256']:
        raise ValueError('B3 evidence changed while endpoint plan was built')
    value = {'episode_id': raw['corpus_episode_id'], 'source_id': raw['source_id'], 'attempt': raw['attempt'],
        'completed': raw['completed'], 'saved_status': raw['status'],
        'termination': deepcopy(raw.get('termination', {})),
        'global_only_inputs': not raw['completed'] and 'stage1' in keys,
        'raw': raw_proof, 'prepared': prepared_proof,
        'quality_keys': keys, 'requests': [requests[k] for k in sorted(requests)],
        'projection_unknown': raw['reward_projection']['unknown']}
    freeze(directory / 'episodes' / raw_path.name, value)
    return value


def validate_incremental_response(value, key, contract, calibration_sha, execution_fingerprint):
    if (value.get('calibration_sha256') != calibration_sha or value.get('physical_gpu') != 0 or
            value.get('local_gpu') != 0 or value.get('execution_fingerprint') != execution_fingerprint or
            value.get('cache_disabled') is not True):
        raise ValueError('B3 response lacks the exact calibrated GPU0 provenance')
    return validate_response(value, key, contract['reference_fingerprint'])


def score_batch(config, root, stop=lambda: None):
    """One directory scan. Existing saved failures are preserved, never retried."""
    root = Path(root)
    output = root / 'B3/gpu_rescore'
    contract = validate_contract(root)
    calibrated = saved_calibration(output / 'incremental', contract)
    final = (root / 'B3/gpu_ready.json').exists()
    final_manifest, identity = validate_manifest(root, config['paths']['repo']) if final else (None, None)
    paths = ([Path(e['raw_path']) for e in final_manifest['episodes']] if final else
             sorted((root / 'B3/content/attempts').glob('*.json')))
    plans = [plan for path in paths if (plan := endpoint_plan(path, root, final=final)) is not None]
    requests = {r['key']: r for p in plans for r in p['requests']}
    if final:
        if requests != {r['key']: r for r in final_manifest['requests']}:
            raise ValueError('Incremental inputs differ from the final collector-owned manifest')
        by_episode = {(p['episode_id'], p['attempt']): p for p in plans}
        for e in final_manifest['episodes']:
            plan = by_episode[e['episode_id'], e['attempt']]
            if (plan['quality_keys'] != e['quality_keys'] or plan['raw']['sha256'] != e['raw_sha256'] or
                    plan['prepared']['sha256'] != e['prepared_sha256']):
                raise ValueError('Final collector endpoint identity differs from its snapshot')
    with reference_lock(config['paths']['repo']), (output / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop()
        if (output / 'complete.json').exists():
            previous = read_json(output / 'complete.json')
            if (not final or calibrated is None or any(previous.get(k) != v for k, v in identity.items()) or
                    previous['contract_sha256'] != file_sha(output / 'incremental/contract.json')):
                raise ValueError('Completed incremental scoring belongs to other inputs')
            calibration_sha = file_sha(output / 'incremental/calibration/complete.json')
            for key in requests:
                validate_incremental_response(read_json(output / 'responses' / (key + '.json')),
                    key, contract, calibration_sha, calibrated['execution_fingerprint'])
            return previous
        needs_model = calibrated is None or any(not (output / 'responses' / (key + '.json')).exists() for key in requests)
        scorer = None
        try:
            if needs_model:
                ensure_gpu0_only()
                scorer = GPU0Scorer(config, contract)
            if calibrated is None:
                calibrate(output / 'incremental', scorer, stop)
                calibrated = saved_calibration(output / 'incremental', contract)
            if scorer is not None and scorer.execution_fingerprint != calibrated['execution_fingerprint']:
                raise ValueError('Loaded scorer differs from the fresh calibration execution')
            calibration_sha = file_sha(output / 'incremental/calibration/complete.json')
            errors, new = [], 0
            for request in requests.values():
                stop()
                key = request['key']
                destination = output / 'responses' / (key + '.json')
                if not destination.exists():
                    atomic_new(destination, score_response(scorer, request, calibration_sha=calibration_sha))
                    new += 1
                response = read_json(destination)
                if not validate_incremental_response(response, key, contract, calibration_sha, calibrated['execution_fingerprint']):
                    errors.append({'key': key, **response['error']})
            validate_contract(root)
            for plan in plans:
                for name in ('raw', 'prepared'):
                    proof = plan[name]
                    if (file_sha(proof['path']) != proof['sha256'] or
                            file_sha(proof['snapshot_path']) != proof['snapshot_sha256']):
                        raise ValueError('B3 immutable snapshot changed during scoring')
            result = {'status': 'complete' if final else 'incremental_batch_complete',
                'fingerprint': contract['reference_fingerprint'], 'execution_fingerprint': calibrated['execution_fingerprint'],
                'physical_gpu': 0, 'local_gpu': 0, 'calibration_sha256': calibration_sha,
                'contract_sha256': file_sha(output / 'incremental/contract.json'),
                'request_count': len(requests), 'episode_count': len(plans), 'new': new,
                'completed_episode_count': sum(p['completed'] for p in plans),
                'saved_incomplete_episode_count': sum(not p['completed'] for p in plans),
                'global_only_input_episode_count': sum(p['global_only_inputs'] for p in plans),
                'reused': len(requests)-new, 'errors': errors, 'responses_root': str(output / 'responses'),
                'training': False, 'paid_calls': 0, 'CPU_scoring_performed': False,
                'reference_driver_sha256': file_sha(__file__)}
            if final:
                validate_manifest(root, config['paths']['repo'])
                result.update(identity)
                freeze(output / 'complete.json', result)
            else:
                batch_key = hashlib.sha256(json.dumps({p['raw']['path']: p['raw']['sha256'] for p in plans}, sort_keys=True).encode()).hexdigest()
                # Repeating the same snapshot set can only reuse saved outcomes.
                path = output / 'incremental/batches' / (batch_key + '.json')
                if not path.exists():
                    atomic_new(path, result)
            write_json(output / 'incremental/status.json', {**result, 'at': time.time()})
            return result
        finally:
            if scorer is not None:
                scorer.close()


@contextmanager
def stop_handler(root):
    stopped = threading.Event()
    def handler(signum, frame):
        stopped.set()
    previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    def check():
        if stopped.is_set() or (Path(root) / 'B3/gpu_rescore/incremental/stop_requested.json').exists():
            raise InterruptedError('Incremental scorer stopped between requests; saved results preserved')
    try:
        yield stopped, check
    finally:
        for sig, value in previous.items():
            signal.signal(sig, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preflight', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true', help='scan no more often than every 45 minutes')
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    from verak.v3.rft1.config import config_for
    config = config_for()
    if args.command == 'preflight':
        print(json.dumps(preflight(config, args.root)))
        return
    directory = args.root / 'B3/gpu_rescore/incremental'
    with (directory / 'dispatch.lock').open('a+') as lock, stop_handler(args.root) as (event, stop):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            stop()
            result = score_batch(config, args.root, stop)
            print(json.dumps(result), flush=True)
            if result['status'] == 'complete' or not args.watch:
                return
            # No live-file/progress polling during this wait. A signal wakes it.
            event.wait(INTERVAL_SECONDS)


if __name__ == '__main__':
    main()
