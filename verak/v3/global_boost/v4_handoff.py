"""Freeze completed teachers for GPU scoring without waiting for CPU estimates."""
from collections import Counter, defaultdict
import time

from ..common import file_sha, pair_key, read_json
from ..corrupt.qc import snapshot_text
from ..train.teacher_bulk import atomic_new
from .config import PHASE
from .expansion import batch_configs, original_teacher_running, root_for
from .measure import measured_path
from .paid import BoostAPI
from .prepare import corpus
from .resources import reference_gpu_fingerprint
from .v4_selection import eligible

CONTRACT = 'teacher_complete_gpu_reference_v2'


def _publish_marker(root, manifest):
    path = root / 'gpu_rescore_manifest.json'
    marker_path = root / 'cpu_ready.json'
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker['manifest_sha256'] != file_sha(path):
            raise ValueError('Frozen GLOBAL manifest changed')
        return marker
    marker = {'status': 'ready', 'schema_version': 2, 'handoff_contract': CONTRACT,
        'manifest_path': str(path), 'manifest_sha256': file_sha(path),
        'teacher_collection_finished': True, 'no_live_paid_calls': True,
        'cpu_measurements_finished': manifest['cpu_measurements_finished'],
        'provisional_observation': manifest['provisional_observation'],
        'gpu_used': False, 'training': False, 'at': time.time()}
    atomic_new(marker_path, marker)
    return marker


def freeze_ready(config):
    root = root_for(config)
    expansion = read_json(root / 'expansion_status.json')
    if expansion.get('stage') != 'generation_finished' or original_teacher_running(config):
        raise RuntimeError('All GLOBAL teacher and Sol rescue generation must finish before readiness')
    api = BoostAPI(config, 0)
    account = api.accounting()
    api.close()
    if account['pending']:
        raise RuntimeError('Cannot freeze readiness while paid calls are live')
    manifest_path = root / 'gpu_rescore_manifest.json'
    if manifest_path.exists():
        # A crash between the immutable manifest and readiness marker must not
        # take a later CPU snapshot or alter the eventual GPU input identity.
        frozen = read_json(manifest_path)
        if (frozen.get('schema_version') != 2 or frozen.get('handoff_contract') != CONTRACT
                or frozen['diversity_plan_sha256'] != file_sha(root / 'v4/plan.json')
                or frozen['source_plan_parent_sha256'] != file_sha(root / 'source_plan.json')):
            raise ValueError('Frozen GLOBAL handoff contract changed')
        raw_paths = {str(p) for cfg in batch_configs(config)
            for p in cfg['paths'][PHASE + '_output'].glob('attempt_*/episodes/*.json')}
        if raw_paths != {e['raw_path'] for e in frozen['episodes']}:
            raise ValueError('Teacher collection changed after the immutable GPU handoff')
        for item in frozen['episodes']:
            for name in ('raw', 'candidate', 'provisional'):
                if item[name + '_path'] is not None and file_sha(item[name + '_path']) != item[name + '_sha256']:
                    raise ValueError('Frozen teacher, candidate, or CPU snapshot changed')
        return _publish_marker(root, frozen)
    fingerprint = read_json(root.parent / 'cpu_scorer/status.json')['fingerprint']
    episodes, requests, provenance, groups = [], {}, Counter(), defaultdict(list)
    for cfg in batch_configs(config):
        cfg[PHASE].update(measurement_source='cpu_provisional', score_fingerprint=fingerprint)
        batch = cfg['paths'][PHASE + '_output']
        rows = corpus(cfg)
        plan = read_json(batch / 'source_plan.json')
        files = {v['episode_id']: v for entries in plan['candidates'].values() for v in entries}
        for path in sorted(batch.glob('attempt_*/episodes/*.json')):
            attempt = int(path.parent.parent.name.split('_')[-1])
            raw = read_json(path)
            eid = raw['corpus_episode_id']
            candidate = rows[eid]
            candidate_file = files[eid]
            if file_sha(candidate_file['path']) != candidate_file['sha256']:
                raise ValueError('Frozen GLOBAL candidate changed')
            raw_sha = file_sha(path)
            measured = measured_path(cfg, attempt, path)
            value = read_json(measured) if measured.exists() else None
            if value is not None and (value.get('raw_generation_sha256') != raw_sha
                    or value.get('score_source') != 'cpu_provisional'):
                raise ValueError('Existing provisional measurement belongs to another trace or contract')
            reward = value.get('global_only_reward') if value is not None else None
            inputs = []
            if raw.get('stage1_layout'):
                for name, text in [('corrupted', candidate['corrupted_text']), ('stage1', snapshot_text(raw['stage1_layout']))]:
                    key = pair_key(candidate['question'], text)
                    requests[key] = {'key': key, 'question': candidate['question'], 'text': text}
                    inputs.append({'state': name, 'key': key})
            entry = {'episode_id': eid, 'attempt': attempt, 'raw_path': str(path), 'raw_sha256': raw_sha,
                'batch': str(batch), 'operator': candidate['operator'], 'source_id': candidate['source_id'],
                'candidate_path': candidate_file['path'], 'candidate_sha256': candidate_file['sha256'],
                'provisional_path': str(measured) if value is not None else None,
                'provisional_sha256': file_sha(measured) if value is not None else None,
                'provisional_global_eligible': eligible(value) if value is not None else None,
                'provisional_R': reward['R'] if reward else None, 'quality_inputs': inputs,
                'quality_identity': value.get('quality_identity') if value is not None else None}
            episodes.append(entry)
            groups[eid].append(entry)
            if value is not None:
                provenance.update((s.get('execution_device', 'unknown'), s.get('scorer_fingerprint', 'unknown'))
                    for s in value.get('quality_scores', value.get('cpu_scores', {})).values())
    complete_ids = {eid for eid, entries in groups.items() if all(e['provisional_path'] is not None for e in entries)}
    provisional_selected = {}
    for eid in complete_ids:
        qualifying = [e for e in groups[eid] if e['provisional_global_eligible']]
        if qualifying:
            best = max(qualifying, key=lambda e: (e['provisional_R'], -e['attempt']))
            provisional_selected[eid] = {'attempt': best['attempt'], 'R': best['provisional_R'], 'path': best['provisional_path']}
    observed = sum(e['provisional_path'] is not None for e in episodes)
    all_measured = observed == len(episodes)
    observations = {'attempts_total': len(episodes), 'attempts_observed': observed,
        'attempts_unobserved': len(episodes) - observed, 'practices_total': len(groups),
        'practices_fully_observed': len(complete_ids), 'practices_not_fully_observed': len(groups) - len(complete_ids)}
    manifest = {'schema_version': 2, 'handoff_contract': CONTRACT, 'component': 'global', 'version': 'v1',
        'task_version': 'v4_prep', 'selected_role': 'global', 'episodes': episodes,
        'requests': [requests[k] for k in sorted(requests)],
        'provisional_score_sources': [{'execution_device': device, 'fingerprint': fp, 'states': n}
            for (device, fp), n in sorted(provenance.items())],
        'cpu_fingerprint': fingerprint, 'reference_gpu_fingerprint': reference_gpu_fingerprint(config),
        'provisional_selected': provisional_selected, 'provisional_observation': observations,
        'canonical_for_selection': False, 'generation_stop_reason': expansion['stop_reason'],
        'source_plan_parent_sha256': file_sha(root / 'source_plan.json'), 'api': account,
        'diversity_plan_path': str(root / 'v4/plan.json'), 'diversity_plan_sha256': file_sha(root / 'v4/plan.json'),
        'teacher_collection_finished': True, 'no_live_paid_calls': True, 'cpu_measurements_finished': all_measured,
        'CPU_contract': 'Snapshot existing measurements only; null entries are unknown, never negative eligibility. CPU queue continues independently.'}
    atomic_new(manifest_path, manifest)
    return _publish_marker(root, manifest)
