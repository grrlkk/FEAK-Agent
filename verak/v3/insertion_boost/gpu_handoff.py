"""Immutable GPU work manifest and file-only reference-score consumer."""
from pathlib import Path
import fcntl
import time

from ..common import file_sha, pair_key, read_json, sha_text, write_json
from ..train.teacher_bulk import atomic_new
from ..train.teacher_comparison import absolute_selection
from ..v2_ops.config import PHASE
from ..v2_ops.report import accounting
from ..v2_ops.teacher import attempt_path
from .calibrate import layout_text
from .cpu_score import gpu_reference_fingerprint, gpu_reference_score, shared_root
from .evaluate import global_reward, measured_row
from .teacher import Resources, prepare


def publish_manifest(config, *, errors=()):
    root = config['paths'][PHASE + '_output']
    design, corpus = prepare(config)
    teacher_status = read_json(root / 'teacher_status.json')
    if teacher_status['saved_attempts'] != teacher_status['planned_attempts'] and not any(
            row['type'] == 'CallBudgetExceeded' for row in teacher_status['errors']):
        raise ValueError('Teacher collection has not finished or stopped cleanly at the cap')
    episodes, requests, sources = [], {}, set()
    for attempt in (1, 2):
        for eid in design['orders'][str(attempt)]:
            path = attempt_path(root, attempt, eid)
            if not path.exists():
                continue
            raw = read_json(path)
            if raw.get('stage1_layout') is None:
                continue
            row = corpus[eid]
            keys = {}
            for name, text in [('corrupted', row['corrupted_text']), ('stage1', layout_text(raw['stage1_layout']))]:
                key = pair_key(row['question'], text)
                requests[key] = {'key': key, 'question': row['question'], 'text': text, 'text_sha256': sha_text(text)}
                keys[name] = key
            measured = root / f'provisional_scored/attempt_{attempt}' / path.name
            item = {'episode_id': eid, 'source_id': row['source_id'], 'attempt': attempt,
                'genre': row['genre'], 'raw_path': str(path), 'raw_sha256': file_sha(path),
                'input_keys': keys, 'operator': 'G_DEL_LINK'}
            if measured.exists():
                values = read_json(measured)
                item.update(provisional_path=str(measured), provisional_sha256=file_sha(measured),
                    provisional_R=values['global_only_reward']['R'], identity_quality=values.get('identity_quality'))
                sources.update(v['scorer_fingerprint'] for v in values['quality_scores'])
            else:
                item['provisional_status'] = 'terminal_measurement_error'
            episodes.append(item)
    budget = accounting(root)
    if budget['pending']:
        raise ValueError('Cannot release GPU work while teacher API calls are live')
    value = {'schema_version': 1, 'component': 'insertion',
        'teacher_design_sha256': file_sha(root / 'teacher_design.json'),
        'reference_gpu_fingerprint': gpu_reference_fingerprint(config),
        'episodes': episodes, 'requests': [requests[k] for k in sorted(requests)],
        'provisional_score_sources': sorted(sources), 'terminal_cpu_measurement_errors': list(errors),
        'selection_policy': 'GPU values only; insertion remains separate from RFT1',
        'korean_selections': 0, 'training': False, 'gpu_calls_by_component': 0}
    path = root / 'gpu_rescore_manifest.json'
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable GPU hand-off manifest changed')
    else:
        atomic_new(path, value)
    ready = {'status': 'ready', 'manifest_path': str(path), 'manifest_sha256': file_sha(path),
        'no_live_paid_calls': True, 'teacher_collection_finished': True, 'cpu_measurements_finished': True,
        'episodes': len(episodes), 'requests': len(requests), 'at': time.time(), 'training': False}
    write_json(root / 'cpu_ready.json', ready)
    return ready


def finalize(config):
    root = config['paths'][PHASE + '_output']
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'gpu_finalize.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = root / 'gpu_selection.json'
        if saved.exists():
            result = read_json(saved)
            if (result['manifest_sha256'] != file_sha(root / 'gpu_rescore_manifest.json') or
                    result['gpu_completion_sha256'] != file_sha(result['gpu_completion_path'])):
                raise ValueError('Completed GPU selection provenance changed')
            for item in result['selected']['global'].values():
                if file_sha(item['path']) != item['sha256']:
                    raise ValueError('A selected GPU trajectory changed after finalization')
            return {k: v for k, v in result.items() if k not in {'selected', 'cpu_to_gpu_comparisons'}}
        return _finalize(config)


def _finalize(config):
    root = config['paths'][PHASE + '_output']
    shared = shared_root(config)
    manifest_path = root / 'gpu_rescore_manifest.json'
    manifest = read_json(manifest_path)
    completion_path = shared / 'gpu_rescore/insertion_complete.json'
    completed = read_json(completion_path)
    fingerprint = gpu_reference_fingerprint(config)
    if (completed.get('status') != 'complete' or completed['manifest_sha256'] != file_sha(manifest_path) or
            completed['fingerprint'] != fingerprint):
        raise ValueError('Root GPU completion does not match the frozen insertion work manifest')
    design, corpus = prepare(config)
    resources = Resources(config)
    selected, errors, comparisons = {}, [], []
    for item in manifest['episodes']:
        path = Path(item['raw_path'])
        if file_sha(path) != item['raw_sha256']:
            raise ValueError('Raw teacher trajectory changed after GPU hand-off')
        raw = read_json(path)
        destination = root / f"scored/attempt_{item['attempt']}" / path.name
        try:
            if destination.exists():
                enriched = read_json(destination)
                if enriched['raw_episode_sha256'] != file_sha(path) or enriched['score_source'] != 'gpu_reference':
                    raise ValueError('Final score provenance is not GPU-only')
            else:
                reward, values, identity = global_reward(config, corpus[item['episode_id']], raw, resources,
                    gpu_reference_score, identity_allowed=False)
                enriched = measured_row(raw, path, reward, values, source='gpu_reference')
                try:
                    atomic_new(destination, enriched)
                except FileExistsError:
                    if read_json(destination) != enriched:
                        raise ValueError('Concurrent GPU reward finalizers disagree')
            reward = enriched['global_only_reward']
            if len(enriched['quality_scores']) != 2 or any(v['execution_device'] != 'gpu_reference' or v['scorer_fingerprint'] != fingerprint
                   for v in enriched['quality_scores']):
                raise ValueError('CPU scores cannot enter final GLOBAL reward')
            keep = absolute_selection(enriched, corpus[item['episode_id']])
            if keep['korean']:
                raise ValueError('KOREAN selection is not authorized')
            if keep['global']:
                candidate = {'attempt': item['attempt'], 'path': str(destination), 'sha256': file_sha(destination),
                    'source_id': item['source_id'], 'R': reward['R'], 'operator': 'G_DEL_LINK'}
                old = selected.get(item['episode_id'])
                if old is None or (candidate['R'], -candidate['attempt']) > (old['R'], -old['attempt']):
                    selected[item['episode_id']] = candidate
            if 'provisional_R' in item:
                cpu_r, gpu_r = item['provisional_R'], reward['R']
                comparisons.append({'episode_id': item['episode_id'], 'attempt': item['attempt'],
                    'provisional_R': cpu_r, 'gpu_R': gpu_r, 'delta_R': gpu_r-cpu_r,
                    'eligibility_flip': (cpu_r >= .8) != (gpu_r >= .8),
                    'cpu_eligible': cpu_r >= .8, 'gpu_eligible': gpu_r >= .8})
        except Exception as exc:
            errors.append({'episode_id': item['episode_id'], 'attempt': item['attempt'],
                'type': type(exc).__name__, 'message': str(exc), 'selection': 'excluded_unmeasured_no_CPU_fallback'})
    value = {'schema_version': 1, 'component': 'insertion', 'score_source': 'gpu_reference',
        'fingerprint': fingerprint, 'manifest_sha256': file_sha(manifest_path),
        'gpu_completion_path': str(completion_path), 'gpu_completion_sha256': file_sha(completion_path),
        'selected': {'global': selected, 'korean': {}}, 'selected_counts': {'global': len(selected), 'korean': 0},
        'cpu_to_gpu_comparisons': comparisons, 'eligibility_flips': sum(r['eligibility_flip'] for r in comparisons),
        'cpu_only_eligible': sum(r['cpu_eligible'] and not r['gpu_eligible'] for r in comparisons),
        'gpu_only_eligible': sum(r['gpu_eligible'] and not r['cpu_eligible'] for r in comparisons),
        'unmeasured': errors, 'measured_episodes': len(manifest['episodes'])-len(errors),
        'gpu_used': True, 'gpu_calls_by_component': 0, 'training': False}
    atomic_new(root / 'gpu_selection.json', value)
    write_json(root / 'best_role_trajectories.json', value['selected'])
    return {k: v for k, v in value.items() if k not in {'selected', 'cpu_to_gpu_comparisons'}}


def publish_approval(config):
    """Publish once only after both GPU components and the >=200-source audit."""
    shared = shared_root(config)
    root = shared / 'cpu_scorer'
    path = root / 'selection_approval.json'
    current = read_json(path) if path.exists() else {}
    if current.get('canonical_for_selection'):
        return {'published': True, 'path': str(path)}
    audit_path = root / 'audit_200/calibration.json'
    complete_path = shared / 'gpu_rescore/complete.json'
    if not audit_path.exists() or not complete_path.exists():
        return {'published': False}
    audit = read_json(audit_path)
    provisional = read_json(root / 'provisional_contract.json')
    provisional_fingerprint = provisional.get('fingerprint', provisional.get('old_fingerprint'))
    if (audit.get('status') != 'complete' or audit.get('source_essays', 0) < 200 or
            audit['fingerprints'] != [provisional_fingerprint]):
        return {'published': False}
    # Older already-running audit supervisors may lack this redundant alias.
    if 'unique_source_essays' not in audit:
        plan = read_json(root / 'audit_200/plan.json')
        if len({e['source_id'] for e in plan['episodes']}) != audit['source_essays']:
            raise ValueError('Audit does not represent distinct source essays')
        audit['unique_source_essays'] = audit['source_essays']
        write_json(audit_path, audit)
    fingerprint = gpu_reference_fingerprint(config)
    markers = {}
    for name in ('insertion', 'global'):
        marker = shared / f'gpu_rescore/{name}_complete.json'
        if not marker.exists() or read_json(marker)['fingerprint'] != fingerprint:
            return {'published': False}
        markers[name] = {'path': str(marker), 'sha256': file_sha(marker)}
    approval = {'canonical_for_selection': True, 'score_source': 'gpu_reference', 'fingerprint': fingerprint,
        'calibration_complete': True, 'calibration_essays': audit['unique_source_essays'],
        'calibration_path': str(audit_path), 'calibration_sha256': file_sha(audit_path),
        'gpu_rescore_complete_sha256': file_sha(complete_path), 'component_completions': markers,
        'CPU_values_in_final_selections': False, 'published_at': time.time()}
    # Compatibility alias is the completed 200-essay audit, never the earlier probe.
    write_json(root / 'calibration.json', audit)
    write_json(path, approval)
    return {'published': True, 'path': str(path)}
