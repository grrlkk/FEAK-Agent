"""Explicit parent-owned GPU reference pass, only after one-shot finishes.

The API/CPU collectors never import or dispatch this entry point. It consumes
their immutable ready manifest, saves reference scores, and does no selection,
training, teacher calls, or mutation of v1 rewards.
"""
from copy import deepcopy
from contextlib import contextmanager
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from verak.v3.common import file_sha, pair_key, read_json, write_json


def validate_manifest(root, repo):
    root, repo = Path(root), Path(repo)
    ready = read_json(root / 'B3/gpu_ready.json')
    path = root / 'B3/gpu_manifest.json'
    if Path(ready['manifest_path']).resolve() != path.resolve() or ready['manifest_sha256'] != file_sha(path):
        raise ValueError('B3 ready manifest changed')
    manifest = read_json(path)
    if (manifest.get('component'), manifest.get('version')) != ('B3', 'v4.3'):
        raise ValueError('Only the frozen v4.3 B3 manifest may be scored')
    if not all(manifest.get(k) is True and ready.get(k) is True for k in
               ('teacher_collection_finished', 'no_live_paid_calls')):
        raise ValueError('Finish B3 collection before GPU scoring')
    if manifest['api']['pending'] or not manifest.get('reference_gpu_fingerprint'):
        raise ValueError('B3 has a pending call or no GPU reference identity')
    for kind in ('sample', 'contract'):
        if file_sha(manifest[kind + '_path']) != manifest[kind + '_sha256']:
            raise ValueError('Frozen B3 ' + kind + ' changed')
    keys = set()
    for request in manifest['requests']:
        key = request['key']
        if key in keys or pair_key(request['question'], request['text']) != key:
            raise ValueError('Duplicate or mismatched B3 scorer input')
        keys.add(key)
    for episode in manifest['episodes']:
        for kind in ('raw', 'prepared'):
            if file_sha(episode[kind + '_path']) != episode[kind + '_sha256']:
                raise ValueError('Frozen B3 episode changed')
        if not set(episode['quality_keys'].values()) <= keys:
            raise ValueError('B3 endpoint has no scorer request')
        from .scale2_projection import restore_document
        raw, prepared = read_json(episode['raw_path']), read_json(episode['prepared_path'])
        for name, key in episode['quality_keys'].items():
            text = restore_document(raw['states'][name]['document']).text
            if pair_key(prepared['corpus']['question'], text) != key:
                raise ValueError('B3 quality key differs from its saved question and endpoint')
    complete_path = repo / 'verak/v3/outputs/oneshot_baseline/complete.json'
    completed = read_json(complete_path)
    report = completed['report']
    if completed.get('completed') is not True or report.get('completed') is not True:
        raise ValueError('One-shot training, evaluation, and report must finish first')
    if file_sha(report['report']) != report['report_sha256']:
        raise ValueError('One-shot completed report changed')
    return manifest, {'manifest_sha256': file_sha(path), 'fingerprint': manifest['reference_gpu_fingerprint'],
                      'oneshot_complete_sha256': file_sha(complete_path),
                      'slot': 'post_oneshot_evaluation', 'scheduled_by': 'root'}


def validate_response(value, key, fingerprint):
    if value.get('fingerprint') != fingerprint or value.get('execution_device') != 'gpu_reference':
        raise ValueError('A non-reference score cannot be used for B3')
    if value.get('error'):
        return False
    result = value['result']
    if result['cache_key'] != key or not math.isfinite(result['mean']):
        raise ValueError('Reference scorer response input mismatch')
    if (len(result['integers']) != 8 or len(result['expected']) != 8 or
        any(type(x) is not int or not 1 <= x <= 9 for x in result['integers']) or
        any(isinstance(x, bool) or not isinstance(x, (int, float)) or
            not math.isfinite(x) or not 1 <= x <= 9 for x in result['expected'])):
        raise ValueError('Reference scorer requires all eight rubric scores')
    if not math.isclose(result['mean'], sum(result['expected']) / 8, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Reference scorer mean differs from its eight rubric expectations')
    return True


@contextmanager
def reference_lock(repo):
    # Share the existing scorer worker lock with any deferred data-boost audit.
    path = Path(repo) / 'verak/v3/outputs/data_boost/gpu_rescore/worker.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(config, root):
    root = Path(root)
    repo = config['paths']['repo']
    manifest, identity = validate_manifest(root, repo)
    output = root / 'B3/gpu_rescore'
    output.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or config['scorer']['gpu'] != 1:
        raise RuntimeError('Preserve the reference physical GPU1 mapping')
    from verak.v3.rft1.service import idle_gpus
    from verak.v3.score.kanana import KananaScorer
    with reference_lock(repo), (output / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed_path = output / 'complete.json'
        if completed_path.exists():
            previous = read_json(completed_path)
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError('Completed B3 reference pass belongs to other inputs')
            for request in manifest['requests']:
                validate_response(read_json(output / 'responses' / (request['key'] + '.json')),
                                  request['key'], identity['fingerprint'])
            return previous
        idle_gpus()
        cfg = deepcopy(config)
        cfg['paths']['output'] = output / 'reference_cache'
        # Only this existing directory has explicit per-response GPU provenance.
        prior = repo / 'verak/v3/outputs/data_boost/gpu_rescore/responses'
        errors, reused, new = [], 0, 0
        scorer = None
        started = time.time()
        try:
            for index, request in enumerate(manifest['requests'], 1):
                key = request['key']
                destination = output / 'responses' / (key + '.json')
                if destination.exists():
                    response = read_json(destination)
                    reused += 1
                elif (prior / destination.name).exists():
                    path = prior / destination.name
                    response = read_json(path)
                    # A previous failure does not stand in for a new B3 score.
                    if validate_response(response, key, identity['fingerprint']):
                        response = {**response, 'reused_from': str(path), 'reused_sha256': file_sha(path)}
                        write_json(destination, response)
                        reused += 1
                    else:
                        response = None
                else:
                    response = None
                if response is None:
                    if scorer is None:
                        scorer = KananaScorer(cfg)
                        if scorer.fingerprint != identity['fingerprint']:
                            raise ValueError('Loaded scorer differs from the GPU reference')
                    response = {'fingerprint': scorer.fingerprint, 'execution_device': 'gpu_reference',
                                'gpu_id': 1, 'slot': identity['slot'], 'scheduled_by': 'root'}
                    before = time.time()
                    try:
                        response['result'] = scorer.score(request['question'], request['text']).to_dict()
                    except Exception as exc:
                        response['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                    response['seconds'] = time.time() - before
                    write_json(destination, response)
                    new += 1
                if not validate_response(response, key, identity['fingerprint']):
                    errors.append({'key': key, **response['error']})
                write_json(output / 'status.json', {**identity, 'stage': 'scoring', 'at': time.time(),
                    'processed': index, 'requested': len(manifest['requests']), 'reused': reused,
                    'new': new, 'errors': len(errors), 'elapsed_s': time.time() - started})
        finally:
            if scorer is not None:
                scorer.close()
        result = {**identity, 'status': 'complete', 'request_count': len(manifest['requests']),
                  'episode_count': len(manifest['episodes']), 'responses_root': str(output / 'responses'),
                  'errors': errors, 'reused': reused, 'new': new, 'at': time.time(),
                  'elapsed_s': time.time() - started, 'training': False, 'paid_calls': 0,
                  'CPU_scoring_performed': False, 'reference_driver_sha256': file_sha(__file__)}
        write_json(completed_path, result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    from verak.v3.rft1.config import config_for
    import torch
    torch.set_num_threads(8)
    print(json.dumps(run(config_for(), args.root)))


if __name__ == '__main__':
    main()
