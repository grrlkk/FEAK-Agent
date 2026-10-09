"""GPU reference rescoring at explicit RFT boundaries; no teacher/API calls."""
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import file_sha, pair_key, read_json, write_json

COMPONENTS = ('global', 'insertion')
SLOTS = ('pre_rft_training', 'post_rft_evaluation')
WORKTREES = {name: Path('/home/chanwoo/FEAK-Agent-' + name + '-boost') for name in COMPONENTS}


def root_for(config):
    return config['paths']['repo'] / 'verak/v3/outputs/data_boost'


def enabled(config):
    repo = config['paths'].get('repo')
    if repo is None:
        return False
    path = root_for(config) / 'task_contract.json'
    return path.exists() and read_json(path).get('gpu_reference_required') is True


def ready(config, name):
    if name not in COMPONENTS:
        raise ValueError('Unknown boost component')
    directory = root_for(config) / name
    path = directory / 'cpu_ready.json'
    if not path.exists():
        return None
    marker = read_json(path)
    if marker.get('status') != 'ready' or not all(marker.get(k) is True for k in
            ('no_live_paid_calls', 'teacher_collection_finished')):
        return None
    manifest_path = Path(marker['manifest_path']).resolve()
    if not manifest_path.is_relative_to(directory.resolve()) or file_sha(manifest_path) != marker['manifest_sha256']:
        raise ValueError(name + ': changed or out-of-scope GPU manifest')
    manifest = read_json(manifest_path)
    raw_ready = (name == 'global' and manifest.get('schema_version') == 2
        and manifest.get('handoff_contract') == 'teacher_complete_gpu_reference_v2'
        and marker.get('handoff_contract') == 'teacher_complete_gpu_reference_v2'
        and manifest.get('version') == 'v1' and manifest.get('selected_role') == 'global')
    if not raw_ready and marker.get('cpu_measurements_finished') is not True:
        return None
    if manifest.get('component') != name or (manifest.get('schema_version') != 1 and not raw_ready):
        raise ValueError('Unexpected GPU manifest identity')
    if not manifest.get('reference_gpu_fingerprint'):
        raise ValueError('GPU reference fingerprint is required')
    seen = set()
    for request in manifest['requests']:
        key = request['key']
        if key in seen or key != pair_key(request['question'], request['text']):
            raise ValueError('Duplicate or mismatched scorer input')
        seen.add(key)
    if raw_ready:
        for episode in manifest['episodes']:
            for field in ('raw','candidate'):
                if file_sha(episode[field+'_path']) != episode[field+'_sha256']:
                    raise ValueError('Teacher-complete raw/candidate artifact changed')
            if episode.get('provisional_path') is not None:
                if file_sha(episode['provisional_path']) != episode['provisional_sha256']:
                    raise ValueError('Frozen available CPU observation changed')
            elif any(episode.get(k) is not None for k in
                     ('provisional_sha256','provisional_R','provisional_global_eligible')):
                raise ValueError('Missing CPU observation cannot supply an eligibility or reward')
            if any(item['key'] not in seen for item in episode['quality_inputs']):
                raise ValueError('A raw teacher quality input is absent from the GPU request list')
    return {'path': str(manifest_path), 'sha256': marker['manifest_sha256'], 'manifest': manifest}


def boundary_plan(config, slot):
    """Freeze readiness at rollout end; later data waits until evaluation ends."""
    if slot not in SLOTS:
        raise ValueError('Unknown GPU allocation boundary')
    if not enabled(config):
        return None
    root = root_for(config)
    path = root / 'gpu_rescore/slots' / (slot + '.json')
    if path.exists():
        return read_json(path)
    components, deferred = {}, []
    for name in COMPONENTS:
        if (root / 'gpu_rescore' / (name + '_complete.json')).exists():
            continue
        item = ready(config, name)
        if item is None:
            deferred.append(name)
        else:
            components[name] = {'manifest_path': item['path'], 'manifest_sha256': item['sha256']}
    result = {'slot': slot, 'at': time.time(), 'components': components, 'deferred': deferred,
              'training': False, 'paid_calls': 0}
    write_json(path, result)
    return result


def authorize_after_evaluation(config, name):
    """A deferred component can arrive after the RFT controller has exited."""
    root = root_for(config)
    done = config['paths']['phase8_rft1_output'] / 'a_complete.json'
    if not done.exists() or read_json(done).get('completed') is not True:
        return None
    status = config['paths']['phase8_rft1_output'] / 'status.json'
    if status.exists() and read_json(status).get('stage') not in {'a_complete', 'a_complete_c_on_hold', 'complete'}:
        return None
    item = ready(config, name)
    if item is None:
        return None
    path = root / 'gpu_rescore/post_evaluation_authorizations' / (name + '.json')
    result = {'component': name, 'slot': 'post_rft_evaluation', 'manifest_path': item['path'],
              'manifest_sha256': item['sha256'], 'rft_complete_sha256': file_sha(done)}
    if path.exists() and read_json(path) != result:
        raise ValueError('Deferred GPU authorization changed')
    write_json(path, result)
    return result


def assert_slot(config, name, slot, manifest_sha):
    root, rft = root_for(config), config['paths']['phase8_rft1_output']
    authorization = root / 'gpu_rescore/slots' / (slot + '.json')
    record = read_json(authorization).get('components', {}).get(name) if authorization.exists() else None
    if record is None and slot == 'post_rft_evaluation':
        path = root / 'gpu_rescore/post_evaluation_authorizations' / (name + '.json')
        record = read_json(path) if path.exists() else None
    if record is None or record['manifest_sha256'] != manifest_sha:
        raise ValueError('No matching explicit GPU slot authorization')
    if slot == 'pre_rft_training':
        state = read_json(rft / 'rollout_status.json')
        if not state.get('sampling_complete') or state.get('saved') != 5720 or state.get('errors'):
            raise RuntimeError('RFT rollouts still own the GPUs')
        if any((rft / 'adapters' / role / 'recipe.json').exists() for role in ('global', 'korean')):
            raise RuntimeError('The gap before RFT training has already closed')
    else:
        if not (rft / 'a_complete.json').exists() or not read_json(rft / 'a_complete.json').get('completed'):
            raise RuntimeError('RFT evaluation/report still owns the GPUs')


def score_component(config, name, slot):
    """Invoked only in its own GPU process, after both GPUs have been released."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise RuntimeError('Reference scorer requires the original physical GPU mapping')
    item = ready(config, name)
    if item is None:
        raise RuntimeError('Teacher component is not ready')
    assert_slot(config, name, slot, item['sha256'])
    root = root_for(config) / 'gpu_rescore'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # This check never terminates another process. RFT's controller releases
        # its owned policy service before scheduling this separate subprocess.
        from ..rft1.service import idle_gpus
        idle_gpus()
        from ..score.kanana import KananaScorer
        cfg = deepcopy(config)
        cfg['paths']['output'] = root / 'reference_cache'
        if cfg['scorer']['gpu'] != 1:
            raise ValueError('Original reference GPU1 recipe changed')
        scorer = KananaScorer(cfg)
        errors, reused = [], 0
        try:
            fingerprint = scorer.fingerprint
            if fingerprint != item['manifest']['reference_gpu_fingerprint']:
                raise ValueError('GPU scorer differs from the frozen SFT/evaluation reference')
            for request in item['manifest']['requests']:
                path = root / 'responses' / (request['key'] + '.json')
                if path.exists():
                    previous = read_json(path)
                    if previous.get('fingerprint') != fingerprint or previous.get('execution_device') != 'gpu_reference':
                        raise ValueError('A non-reference score was placed in the GPU output directory')
                    if previous.get('error'):
                        errors.append({'key': request['key'], **previous['error']})
                    elif previous['result']['cache_key'] != request['key']:
                        raise ValueError('GPU result input key changed')
                    reused += 1
                    continue
                started = time.time()
                response = {'fingerprint': fingerprint, 'execution_device': 'gpu_reference',
                    'gpu_id': 1, 'slot': slot, 'component_first_requested_by': name}
                try:
                    result = scorer.score(request['question'], request['text']).to_dict()
                    response.update(result=result, cache_hit=result['cache_hit'])
                except Exception as exc:
                    response['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                    errors.append({'key': request['key'], **response['error']})
                response['seconds'] = time.time() - started
                write_json(path, response)
                write_json(root / 'status.json', {'stage': 'scoring', 'component': name, 'slot': slot,
                    'last_key': request['key'], 'at': time.time(), 'errors': len(errors)})
        finally:
            scorer.close()
        result = {'status': 'complete', 'component': name, 'manifest_sha256': item['sha256'],
            'request_count': len(item['manifest']['requests']), 'episode_count': len(item['manifest']['episodes']),
            'fingerprint': fingerprint, 'slot': slot, 'responses_root': str(root / 'responses'),
            'errors': errors, 'reused': reused, 'at': time.time(), 'training': False, 'paid_calls': 0}
        write_json(root / (name + '_complete.json'), result)
        return result


def run_component(config, name, slot):
    """Child GPU process exits before the CPU-only reward/selection consumer."""
    root = root_for(config) / 'gpu_rescore'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = _run_component(config, name, slot)
        complete_reference_pass(config)
        return result


def complete_reference_pass(config):
    """Publish one immutable aggregate after both GPU-only consumers finish."""
    root = root_for(config) / 'gpu_rescore'
    files = [root / (name + suffix) for name in COMPONENTS
             for suffix in ('_complete.json', '_consumer_complete.json')]
    if not all(path.exists() for path in files):
        return None
    hashes = {path.name: file_sha(path) for path in files}
    fingerprints = {read_json(root / (name + '_complete.json'))['fingerprint'] for name in COMPONENTS}
    if len(fingerprints) != 1:
        raise ValueError('Components used different GPU reference scorers')
    path = root / 'complete.json'
    if path.exists():
        value = read_json(path)
        if value['component_artifact_sha256'] != hashes:
            raise ValueError('Completed reference-pass artifacts changed')
        return value
    value = {'status': 'complete', 'score_source': 'gpu_reference', 'fingerprint': fingerprints.pop(),
             'component_artifact_sha256': hashes, 'at': time.time(), 'paid_calls': 0, 'training': False}
    write_json(path, value)
    return value


def _run_component(config, name, slot):
    root = root_for(config) / 'gpu_rescore'
    completion = root / (name + '_complete.json')
    consumed = root / (name + '_consumer_complete.json')
    if consumed.exists():
        record = read_json(consumed)
        if record['gpu_complete_sha256'] != file_sha(completion) or file_sha(record['selection_path']) != record['selection_sha256']:
            raise ValueError('Completed GPU selection consumer changed')
        return record
    if not completion.exists():
        command = [sys.executable, '-m', 'verak.v3.cli.data_boost_report', 'rescore',
                   '--component', name, '--slot', slot, '--repo', str(config['paths']['repo'])]
        with (root / (name + '_gpu.log')).open('a') as stream:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[3], check=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1', 'OMP_NUM_THREADS': '8',
                     'MKL_NUM_THREADS': '8', 'HF_HUB_OFFLINE': '1'},
                stdout=stream, stderr=subprocess.STDOUT)
    command = [sys.executable, '-m', 'verak.v3.cli.' + name + '_boost', 'gpu-finalize']
    if name == 'insertion':
        command += ['--config', 'v2']
    with (root / (name + '_selection.log')).open('a') as stream:
        subprocess.run(command, cwd=WORKTREES[name], check=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            stdout=stream, stderr=subprocess.STDOUT)
    selection = root.parent / name / 'gpu_selection.json'
    record = {'gpu_complete_sha256': file_sha(completion), 'selection_path': str(selection),
              'selection_sha256': file_sha(selection), 'component': name, 'slot': slot, 'at': time.time()}
    write_json(consumed, record)
    return record


def at_boundary(config, slot):
    plan = boundary_plan(config, slot)
    if plan is None:
        return None
    for name in plan['components']:
        run_component(config, name, slot)
    if slot == 'pre_rft_training':
        root = root_for(config)
        selected = root / 'global/gpu_selection.json'
        # This manifest fixes what the imminent training run can use. Later
        # completion cannot modify an exported/trained RFT1 dataset.
        include = 'global' in plan['components'] and selected.exists()
        result = {'include_extra_global': include, 'version': 'v1', 'role': 'global',
            'operators': ['G_PARA_SWAP', 'G_SENT_MOVE'], 'v2_included': False,
            'reason': 'ready_and_GPU_rescored' if include else 'not_ready_at_rollout_boundary',
            'selection_path': str(selected) if include else None,
            'selection_sha256': file_sha(selected) if include else None,
            'slot_plan_sha256': file_sha(root / 'gpu_rescore/slots/pre_rft_training.json')}
        write_json(root / 'rft1_extra_merge.json', result)
    return plan


def service_deferred(config):
    """Called by the file watcher only after A has completed in full."""
    if not enabled(config):
        return
    for name in COMPONENTS:
        completion = root_for(config) / 'gpu_rescore' / (name + '_consumer_complete.json')
        if completion.exists():
            continue
        if authorize_after_evaluation(config, name):
            run_component(config, name, 'post_rft_evaluation')
    complete_reference_pass(config)
