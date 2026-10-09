"""Final component artifacts across all immutable GLOBAL practice batches."""
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

from ..common import file_sha, pair_key, read_json, write_json
from ..corrupt.qc import snapshot_text
from ..train.teacher_comparison import absolute_selection
from ..train.teacher_bulk import atomic_new, collection_lock
from .config import OPERATORS, PHASE
from .expansion import batch_configs, original_teacher_running, root_for
from .measure import measured_path
from .paid import BoostAPI
from .prepare import corpus


def approved(config):
    path = root_for(config).parent / 'cpu_scorer/selection_approval.json'
    if not path.exists():
        return None
    value = read_json(path)
    if (not value.get('canonical_for_selection') or not value.get('fingerprint')
            or value.get('score_source') != 'gpu_reference' or value.get('calibration_essays', 0) < 200):
        return None
    component = root_for(config).parent / 'gpu_rescore/global_complete.json'
    if not component.exists() or read_json(component)['fingerprint'] != value['fingerprint']:
        return None
    calibration = Path(value.get('calibration_path', root_for(config).parent / 'cpu_scorer/audit_200/calibration.json'))
    if not calibration.exists() or file_sha(calibration) != value.get('calibration_sha256'):
        return None
    return {**value, 'calibration_path': str(calibration), 'path': str(path),
            'sha256': file_sha(path), 'gpu_component_sha256': file_sha(component)}


def weighted(rows, key, count='global_reward_measured'):
    total = sum(r[count] for r in rows if r[key] is not None)
    return sum(r[key]*r[count] for r in rows if r[key] is not None)/total if total else None


def report(config, approval):
    if (root_for(config)/'v4/plan.json').exists():
        from .v4_report import report as report_v4
        return report_v4(config,approval)
    from .report import report as batch_report
    root = root_for(config)
    batches, selected, inputs = [], {}, {}
    score_devices = Counter()
    for cfg in batch_configs(config):
        cfg[PHASE].update(measurement_source='gpu_reference', score_fingerprint=approval['fingerprint'],
                          scorer_approval_sha256=approval['gpu_component_sha256'])
        target = cfg['paths'][PHASE + '_output']
        batch_report(cfg)
        value = read_json(target / 'component_metrics.json')
        batches.append({'root': str(target), 'source_plan_sha256': file_sha(target / 'source_plan.json'),
                        'metrics': value})
        for key, item in value['selected'].items():
            if key in selected:
                raise ValueError('Duplicate immutable practice identity across batches')
            selected[key] = item
        inputs.update(corpus(cfg))
        score_devices.update(value['score_device_observations'])
    metrics = {}
    for operator in OPERATORS:
        rows = [b['metrics']['operators'][operator] for b in batches]
        qc = {k: sum(r['qc'][k] for r in rows) for k in ('generated', 'judged', 'passed', 'unknown')}
        qc['pass_rate_judged'] = qc['passed']/qc['judged'] if qc['judged'] else None
        counts = {k: sum(r[k] for r in rows) for k in ('generated', 'teacher_planned', 'teacher_saved',
            'generation_completed', 'global_reward_measured', 'fully_recovered_attempts', 'selected_global')}
        pairs = {k: sum(r['at_least_one_of_two'][k] for r in rows) for k in ('fully_observed_essays', 'recovered')}
        pairs['share'] = pairs['recovered']/pairs['fully_observed_essays'] if pairs['fully_observed_essays'] else None
        terminations = {}
        for role in ('global', 'korean'):
            end = Counter()
            for r in rows:
                end.update(r['termination'][role])
            terminations[role] = dict(end)
        unknown = [item for r in rows for item in r['unknown']]
        metrics[operator] = {'requested': 400, **counts, 'qc': qc,
            'not_generated': 400-counts['generated'], 'selected_korean': 0,
            'completion_rate': counts['generation_completed']/counts['teacher_saved'] if counts['teacher_saved'] else None,
            'teacher_main_recovery': weighted(rows, 'teacher_main_recovery'),
            'full_recovery_rate': counts['fully_recovered_attempts']/counts['global_reward_measured'] if counts['global_reward_measured'] else None,
            'R': weighted(rows, 'R'), 'R_over': weighted(rows, 'R_over'),
            'at_least_one_of_two': pairs,
            'observed_any_sample': {k: sum(r['observed_any_sample'][k] for r in rows) for k in ('essays', 'recovered')},
            'termination': terminations, 'unknown': unknown,
            'distinct_sources': len({r['source_id'] for r in inputs.values() if r['operator'] == operator}),
            'selected_distinct_sources': len({r['source_id'] for r in selected.values() if r['operator'] == operator})}
        hashes = [r['corrupted_hash'] for r in inputs.values() if r['operator'] == operator]
        if len(hashes) != len(set(hashes)) or counts['generated'] > 400:
            raise ValueError('Expansion exceeds its quota or repeats a corrupted practice')
    api = BoostAPI(config, 0)
    account = api.accounting()
    with api.db() as db:
        outcomes = Counter(dict(db.execute('SELECT status,COUNT(*) FROM calls GROUP BY status').fetchall()))
    api.close()
    if account['confirmed_usd']+account['reserved_usd'] > 12.+1e-8:
        raise ValueError('Shared GLOBAL boost cap exceeded')
    initial = read_json(root / 'source_plan.json')
    audit = read_json(root / 'variant_supply_validated.json')
    status = read_json(root / 'expansion_status.json')
    source_histogram = Counter(r['source_id'] for r in inputs.values())
    value = {'phase': PHASE, 'aggregate': True, 'operators': metrics,
        'selected_global': len(selected), 'selected_korean': 0, 'selected': selected,
        'api': account, 'api_terminal_status_counts': dict(outcomes),
        'distinct_new_sources': len(source_histogram), 'practices_per_source': dict(source_histogram),
        'qc_pass_distinct_sources': len({inputs[i]['source_id'] for b in batches for i in
            read_json(Path(b['root']) / 'teacher_design.json')['orders']['1']}),
        'selected_distinct_sources': len({r['source_id'] for r in selected.values()}),
        'source_inventory': initial['source_inventory']['counts'],
        'source_plan_parent_sha256': file_sha(root / 'source_plan.json'),
        'variant_supply': audit['overall'], 'variant_supply_sha256': file_sha(root / 'variant_supply_validated.json'),
        'batch_manifests': [{'root': b['root'], 'source_plan_sha256': b['source_plan_sha256']} for b in batches],
        'selection_rule': 'Existing SFT absolute_selection; best GLOBAL R per practice, not per shared source. No extra RFT-only STOP/rejection gate.',
        'source_grouping': 'These are new corruption practices on shared unused sources; any future training/validation partition must group by source.',
        'teacher_roles': ['global', 'korean'], 'selected_roles': ['global'],
        'unmeasured_reward_roles': ['korean', 'combined'],
        'score_device_observations': dict(score_devices),
        'canonical_for_selection': True, 'scorer_approval': approval,
        'score_source': 'gpu_reference', 'gpu_reference_rescoring_used': True,
        'selection_changes': read_json(root / 'gpu_selection.json')['selection_changes'],
        'provisional_score_sources': read_json(root / 'gpu_rescore_manifest.json')['provisional_score_sources'],
        'cpu_score_audit': read_json(approval['calibration_path']),
        'completion': {'teacher_requested': sum(r['teacher_planned'] for r in metrics.values()),
            'teacher_saved': sum(r['teacher_saved'] for r in metrics.values()),
            'rewards_measured': sum(r['global_reward_measured'] for r in metrics.values()),
            'stop_reason': status['stop_reason'], 'pending_api': account['pending']},
        'gpu_used': False, 'training': False}
    write_json(root / 'best_global_trajectories.json', {'global': selected, 'korean': {}})
    write_json(root / 'component_metrics.json', value)
    def number(v):
        return 'NA' if v is None else f'{v:.4f}'
    lines = ['# GLOBAL data boost', '',
        f"The preserved Phase3b source filter leaves {len(source_histogram)} unused source essays. "
        f"They supply {audit['overall']['G_PARA_SWAP']['valid_distinct']} distinct valid G_PARA_SWAP "
        f"and {audit['overall']['G_SENT_MOVE']['valid_distinct']} G_SENT_MOVE variants. "
        'Multiple distinct practices can share a source; each contains one corruption record. '
        'The original 84-candidate batch is preserved unchanged.', '',
        '|Operator|Practices|Distinct sources|QC pass/judged|Teacher saved/planned|Main recovery|R_over|Selected GLOBAL|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for op, r in metrics.items():
        lines.append(f"|{op}|{r['generated']}|{r['distinct_sources']}|{r['qc']['passed']}/{r['qc']['judged']}|"
            f"{r['teacher_saved']}/{r['teacher_planned']}|{number(r['teacher_main_recovery'])}|{number(r['R_over'])}|{r['selected_global']}|")
    changes = value['selection_changes']
    lines += ['', '|Mixed provisional / GPU reference comparison|Count|', '|---|---:|',
        f"|Provisional GLOBAL selections (CPU + exact saved GPU cache)|{changes['cpu_provisional_selected']}|",
        f"|Final GPU-reference GLOBAL selections|{changes['gpu_selected']}|",
        f"|Attempt eligibility flips at R >= 0.80|{changes['attempt_eligibility_flip_count']}|",
        f"|Best-attempt or membership changes|{changes['best_attempt_or_membership_change_count']}|"]
    lines += ['', f"Stopped: `{status['stop_reason']}`. API cost ${account['confirmed_usd']:.6f}; "
        f"retained unknown reservations ${account['reserved_usd']:.6f}; live pending {account['pending']}; shared cap $12.", '',
        value['selection_rule'], '', value['source_grouping'], '',
        'The unchanged two-stage v1 teacher runs both editors, but only GLOBAL reward states and GLOBAL selections are measured here. '
        'KOREAN/combined rewards are unmeasured. No KOREAN data was selected and this component did not train. '
        'Generation used API/CPU; the root-owned deferred GPU reference pass supplied every final quality reward.', '',
        'The provisional comparison combines CPU scores with exact-input, exact-fingerprint saved GPU cache hits; '
        'identical scorer inputs also have a proven zero quality delta without an absolute score. '
        'These explicitly mixed provisional values never supply the final reward table or final selections.', '',
        'Provisional scored-state provenance: ' + ', '.join(
            f"{s['execution_device']}={s['states']}" for s in value['provisional_score_sources']) + '.', '',
        f"Final selection and reported R use GPU-reference fingerprint `{approval['fingerprint']}` exclusively. "
        'CPU values and internal decisions are retained only for the >=200-essay compatibility audit and eligibility-flip comparison.', '',
        'Unattempted or unmeasured slots and terminal errors remain explicit in component_metrics.json; '
        'a missing result is never a successful recovery.', '']
    (root / 'component_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return value


def cpu_ready(config):
    """Freeze all raw episodes and provisional CPU results for root's GPU slot."""
    from .measure import run as measure
    from .resources import reference_gpu_fingerprint
    root = root_for(config)
    v4 = (root / 'v4/plan.json').exists()
    marker_path = root / 'cpu_ready.json'
    if marker_path.exists():
        marker = read_json(marker_path)
        if file_sha(marker['manifest_path']) != marker['manifest_sha256']:
            raise ValueError('Frozen GPU rescore manifest changed')
        return marker
    if v4:
        from .v4_handoff import freeze_ready
        return freeze_ready(config)
    expansion = read_json(root / 'expansion_status.json')
    if expansion.get('stage') != 'generation_finished' or original_teacher_running(config):
        raise RuntimeError('All GLOBAL teacher generation must finish before readiness')
    service = read_json(root.parent / 'cpu_scorer/status.json')
    fingerprint = service['fingerprint']
    episodes, requests, sources, provisional_selected = [], {}, Counter(), {}
    for cfg in batch_configs(config):
        cfg[PHASE].update(measurement_source='cpu_provisional', score_fingerprint=fingerprint)
        batch = cfg['paths'][PHASE + '_output']
        write_json(root / 'finalization_status.json', {'stage': 'cpu_provisional', 'batch': str(batch),
            'fingerprint': fingerprint, 'at': time.time(), 'gpu_used': False})
        measure(cfg)
        rows = corpus(cfg)
        plan = read_json(batch / 'source_plan.json')
        files = {v['episode_id']: v for entries in plan['candidates'].values() for v in entries}
        for path in sorted(batch.glob('attempt_*/episodes/*.json')):
            attempt = int(path.parent.parent.name.split('_')[-1])
            raw = read_json(path)
            episode_id = raw['corpus_episode_id']
            candidate = rows[episode_id]
            measured = measured_path(cfg, attempt, path)
            if not measured.exists():
                raise RuntimeError('Provisional CPU measurement missing: '+str(path))
            value = read_json(measured)
            if value['raw_generation_sha256'] != file_sha(path):
                raise ValueError('Provisional measurement raw trace changed')
            score_inputs = []
            if raw.get('stage1_layout'):
                for name, text in [('corrupted', candidate['corrupted_text']), ('stage1', snapshot_text(raw['stage1_layout']))]:
                    key = pair_key(candidate['question'], text)
                    requests[key] = {'key': key, 'question': candidate['question'], 'text': text}
                    score_inputs.append({'state': name, 'key': key})
            if v4:
                from .v4_selection import eligible
                keep = eligible(value)
            else:
                keep = absolute_selection(value, candidate)['global']
            reward = value.get('global_only_reward')
            entry = {'episode_id': episode_id, 'attempt': attempt, 'raw_path': str(path), 'raw_sha256': file_sha(path),
                'batch': str(batch), 'operator': candidate['operator'], 'source_id': candidate['source_id'],
                'candidate_path': files[episode_id]['path'], 'candidate_sha256': files[episode_id]['sha256'],
                'provisional_path': str(measured), 'provisional_sha256': file_sha(measured),
                'provisional_global_eligible': keep, 'provisional_R': reward['R'] if reward else None,
                'quality_inputs': score_inputs, 'quality_identity': value.get('quality_identity')}
            episodes.append(entry)
            if keep and (episode_id not in provisional_selected or
                    (reward['R'], -attempt) > (provisional_selected[episode_id]['R'], -provisional_selected[episode_id]['attempt'])):
                provisional_selected[episode_id] = {'attempt': attempt, 'R': reward['R'], 'path': str(measured)}
            sources.update((s.get('execution_device', 'unknown'), s.get('scorer_fingerprint', 'unknown'))
                           for s in value.get('quality_scores', value.get('cpu_scores', {})).values())
    api = BoostAPI(config, 0)
    account = api.accounting()
    api.close()
    if account['pending']:
        raise RuntimeError('Cannot freeze readiness while paid calls are live')
    manifest = {'schema_version': 1, 'component': 'global', 'version': 'v1', 'selected_role': 'global',
        'episodes': episodes, 'requests': [requests[k] for k in sorted(requests)],
        'provisional_score_sources': [{'execution_device': device, 'fingerprint': fp, 'states': n}
                                      for (device, fp), n in sources.items()],
        'cpu_fingerprint': fingerprint, 'reference_gpu_fingerprint': reference_gpu_fingerprint(config),
        'provisional_selected': provisional_selected,
        'canonical_for_selection': False, 'generation_stop_reason': expansion['stop_reason'],
        'source_plan_parent_sha256': file_sha(root / 'source_plan.json'), 'api': account}
    if v4:
        manifest.update(task_version='v4_prep', diversity_plan_path=str(root/'v4/plan.json'),
                        diversity_plan_sha256=file_sha(root/'v4/plan.json'))
    manifest_path = root / 'gpu_rescore_manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError('Interrupted readiness manifest differs; preserve it for review')
    else:
        atomic_new(manifest_path, manifest)
    marker = {'status': 'ready', 'manifest_path': str(manifest_path), 'manifest_sha256': file_sha(manifest_path),
        'no_live_paid_calls': True, 'teacher_collection_finished': True, 'cpu_measurements_finished': True,
        'gpu_used': False, 'training': False, 'at': time.time()}
    atomic_new(marker_path, marker)
    return marker


@contextmanager
def gpu_selection_lock(root):
    directory = root / 'gpu_selection'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'collection.lock').open('a+') as stream:
        # Both the RFT boundary controller and this background finalizer may
        # consume the same completed GPU pass. The second waits, then returns
        # the immutable selection instead of failing the RFT controller.
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_finalize(config):
    """Synchronous no-API consumer, independent of the insertion/audit schedule."""
    from .measure import run as measure
    root = root_for(config)
    v4 = (root/'v4/plan.json').exists()
    with gpu_selection_lock(root):
        ready = read_json(root / 'cpu_ready.json')
        manifest = read_json(ready['manifest_path'])
        if file_sha(ready['manifest_path']) != ready['manifest_sha256']:
            raise ValueError('Frozen GLOBAL manifest changed')
        complete_path = root.parent / 'gpu_rescore/global_complete.json'
        complete = read_json(complete_path)
        if complete['manifest_sha256'] != ready['manifest_sha256'] or complete.get('component') != 'global':
            raise ValueError('GPU completion belongs to a different component/manifest')
        if complete['request_count'] != len(manifest['requests']):
            raise ValueError('GLOBAL GPU reference pass is incomplete')
        if complete['fingerprint'] != manifest['reference_gpu_fingerprint']:
            raise ValueError('GPU reference fingerprint differs from the frozen scorer')
        old_path = root / 'gpu_selection.json'
        if old_path.exists():
            old = read_json(old_path)
            if (old['manifest_sha256'] != ready['manifest_sha256'] or old['fingerprint'] != complete['fingerprint']
                    or old['gpu_complete_sha256'] != file_sha(complete_path)):
                raise ValueError('Immutable GPU selection identity changed')
            return old
        configs = {str(c['paths'][PHASE + '_output']): c for c in batch_configs(config)}
        for cfg in configs.values():
            cfg[PHASE].update(measurement_source='gpu_reference', score_fingerprint=complete['fingerprint'],
                scorer_approval_sha256=file_sha(complete_path))
            measure(cfg)
        selected, flips, failed_episodes = {}, [], []
        raw_ready = manifest.get('schema_version') == 2 and manifest.get('handoff_contract') == 'teacher_complete_gpu_reference_v2'
        if manifest.get('schema_version', 1) != 1 and not raw_ready:
            raise ValueError('Unknown GLOBAL handoff contract')
        if raw_ready and not v4:
            raise ValueError('Teacher-complete handoff is authorized only for the GLOBAL v4 continuation')
        compared_attempts, gpu_observed = 0, set()
        score_errors = complete.get('errors', [])
        failed_keys = (set(score_errors) if isinstance(score_errors, dict) else
                       {e.get('key', e.get('cache_key')) for e in score_errors})
        for item in manifest['episodes']:
            for path_key, hash_key in [('raw_path', 'raw_sha256'), ('candidate_path', 'candidate_sha256'),
                                        ('provisional_path', 'provisional_sha256')]:
                if path_key == 'provisional_path' and item[path_key] is None and raw_ready:
                    if any(item.get(key) is not None for key in ('provisional_sha256', 'provisional_global_eligible', 'provisional_R')):
                        raise ValueError('Missing CPU observation must have explicit null provenance and decision')
                    continue
                if file_sha(item[path_key]) != item[hash_key]:
                    raise ValueError('Frozen generation, candidate, or CPU observation changed')
            candidate = read_json(item['candidate_path'])
            path = measured_path(configs[item['batch']], item['attempt'], Path(item['raw_path']))
            if not path.exists():
                failed_inputs = [v['key'] for v in item['quality_inputs'] if v['key'] in failed_keys]
                if failed_inputs:
                    failed_episodes.append({'episode_id': item['episode_id'], 'attempt': item['attempt'],
                        'failed_input_keys': failed_inputs, 'cpu_eligible': item['provisional_global_eligible']})
                    continue
                raise RuntimeError('GPU reward rebuild missing: '+str(path))
            value = read_json(path)
            gpu_observed.add((item['episode_id'], item['attempt']))
            scores = value.get('quality_scores', {})
            if any(s.get('execution_device') != 'gpu_reference' or s.get('scorer_fingerprint') != complete['fingerprint'] for s in scores.values()):
                raise ValueError('Final GLOBAL reward contains a non-reference score')
            if v4:
                from .v4_selection import eligible
                keep = eligible(value)
            else:
                keep = absolute_selection(value, candidate)['global']
            if item['provisional_global_eligible'] is not None:
                compared_attempts += 1
            if item['provisional_global_eligible'] is not None and keep != item['provisional_global_eligible']:
                flips.append({'episode_id': item['episode_id'], 'attempt': item['attempt'],
                    'cpu_eligible': item['provisional_global_eligible'], 'gpu_eligible': keep,
                    'cpu_R': item['provisional_R'], 'gpu_R': value.get('global_only_reward', {}).get('R')})
            if keep:
                reward = value['global_only_reward']
                entry = {'path': str(path), 'sha256': file_sha(path), 'attempt': item['attempt'],
                    'R': reward['R'], 'R_rec': reward['R_rec'], 'R_over': reward['R_over'],
                    'candidate': candidate, 'candidate_path': item['candidate_path'], 'candidate_sha256': item['candidate_sha256'],
                    'source_id': candidate['source_id'], 'operator': candidate['operator'],
                    'score_source': 'gpu_reference', 'source_tag': 'extra_teacher', 'selected_role': 'global'}
                previous = selected.get(item['episode_id'])
                if previous is None or (entry['R'], -entry['attempt']) > (previous['R'], -previous['attempt']):
                    selected[item['episode_id']] = entry
        prior = manifest['provisional_selected']
        practice_ids = {i['episode_id'] for i in manifest['episodes']}
        comparable_practices = practice_ids if not raw_ready else {eid for eid in practice_ids if all(
            i['provisional_global_eligible'] is not None and (eid, i['attempt']) in gpu_observed
            for i in manifest['episodes'] if i['episode_id'] == eid)}
        best_changes = [{'episode_id': i, 'cpu_attempt': prior.get(i, {}).get('attempt'),
                         'gpu_attempt': selected.get(i, {}).get('attempt')}
                        for i in sorted((set(prior) | set(selected)) & comparable_practices)
                        if prior.get(i, {}).get('attempt') != selected.get(i, {}).get('attempt')]
        gpu_comparable_selected = len(set(selected) & comparable_practices)
        cap_excluded=[]
        gpu_pre_cap_count=len(selected)
        if v4:
            from .v4_data import holdouts
            from .v4_selection import source_cap, rescue_action_targets
            held,_=holdouts(config)
            selected={i:e for i,e in selected.items() if e['source_id'] not in held}
            selected,cap_excluded=source_cap(selected)
            for eid,entry in selected.items():
                if entry['attempt']!=3:
                    continue
                original=read_json(entry['path'])
                exported=rescue_action_targets(original)
                exported['source_gpu_measured_trajectory']={'path':entry['path'],'sha256':entry['sha256']}
                target=root/'v4/action_only_exports'/(eid.replace(':','_')+'.json')
                if target.exists():
                    if read_json(target)!=exported:
                        raise ValueError('Immutable Sol action-only export changed')
                else:
                    atomic_new(target,exported)
                entry.update(source_gpu_measured_path=entry['path'],source_gpu_measured_sha256=entry['sha256'],
                             path=str(target),sha256=file_sha(target),
                             action_only_export=exported['action_only_export'])
        value = {'version': 'v1', 'role': 'global', 'score_source': 'gpu_reference',
            'fingerprint': complete['fingerprint'], 'manifest_sha256': ready['manifest_sha256'],
            'gpu_complete_sha256': file_sha(complete_path), 'slot': complete['slot'],
            'selected': selected, 'selected_global': len(selected), 'selected_korean': 0,
            'selection_changes': {'attempt_eligibility_flip_count': len(flips), 'attempt_flips': flips,
                'best_attempt_or_membership_change_count': len(best_changes), 'best_attempt_or_membership_changes': best_changes,
                'cpu_provisional_selected': len(set(prior) & comparable_practices), 'gpu_selected': len(selected),
                'attempts_compared': compared_attempts, 'attempts_unknown': len(manifest['episodes']) - compared_attempts,
                'attempts_missing_CPU_snapshot': sum(i['provisional_global_eligible'] is None for i in manifest['episodes']),
                'practices_compared': len(comparable_practices), 'practices_unknown': len(practice_ids) - len(comparable_practices),
                'gpu_selected_before_cap_in_comparable_practices': gpu_comparable_selected,
                'comparison_scope': 'Observed CPU/GPU attempts; best-attempt comparison only when every saved attempt of a practice is observed on both.',
                'CPU_comparison_internal_until_200_source_audit': True},
            'failed_gpu_episodes': failed_episodes, 'gpu_score_errors': score_errors,
            'api_calls': 0, 'gpu_calls_by_component': 0, 'training': False}
        if v4:
            value.update(task_version='v4_prep',diversity_plan_path=str(root/'v4/plan.json'),
                diversity_plan_sha256=file_sha(root/'v4/plan.json'), source_cap=4,
                gpu_eligible_before_source_cap=gpu_pre_cap_count, source_cap_exclusions=cap_excluded,
                new_source_rule='Active agent_train sources allowed; new structural positions; frozen SFT source holdouts excluded.')
        atomic_new(old_path, value)
        return value


def finalize(config):
    root = root_for(config)
    status_path = root / 'finalization_status.json'
    with collection_lock(root / 'finalization'):
        while True:
            expansion = read_json(root / 'expansion_status.json') if (root / 'expansion_status.json').exists() else {}
            if expansion.get('stage') == 'generation_finished' and not original_teacher_running(config):
                break
            write_json(status_path, {'stage': 'waiting_generation', 'at': time.time(),
                'generation_stage': expansion.get('stage'), 'gpu_used': False, 'training': False})
            time.sleep(30)
        if (root / 'v4/plan.json').exists():
            from .v4_pre_gpu import report as pre_gpu_report
            pre_gpu_report(config)
        cpu_ready(config)
        while not (root.parent / 'gpu_rescore/global_complete.json').exists():
            write_json(status_path, {'stage': 'waiting_root_gpu_reference_slot', 'at': time.time(), 'gpu_used': False})
            time.sleep(30)
        gpu_finalize(config)
        while (approval := approved(config)) is None:
            write_json(status_path, {'stage': 'gpu_selection_ready_waiting_200_essay_audit', 'at': time.time(), 'gpu_used': False})
            time.sleep(30)
        if approved(config) != approval:
            raise RuntimeError('Scorer approval changed while measuring')
        result = report(config, approval)
        if (result['completion']['teacher_saved'] < result['completion']['teacher_requested']
                and 'budget' not in result['completion']['stop_reason']
                and not result['completion'].get('legacy_unattempted_authorized_rescope')):
            raise RuntimeError('Unattempted authorized teacher slots remain without a budget stop')
        if result['api']['pending']:
            raise RuntimeError('Live paid calls remain; cannot close the GLOBAL boost')
        terminal_errors = read_json(root / 'gpu_selection.json').get('failed_gpu_episodes', [])
        result['terminal_measurement_errors'] = terminal_errors
        result['measurements_finished'] = True
        write_json(root / 'component_metrics.json', result)
        marker = {'status': 'budget_stop' if 'budget' in result['completion']['stop_reason'] else 'complete',
            'metrics_path': str(root / 'component_metrics.json'), 'metrics_sha256': file_sha(root / 'component_metrics.json'),
            'report_path': str(root / 'component_report.md'), 'report_sha256': file_sha(root / 'component_report.md'),
            'scorer_approval_sha256': approval['sha256'], 'no_live_paid_calls': True,
            'measurements_finished': True, 'stopped': True, 'gpu_used': False, 'training': False,
            'terminal_measurement_error_count': len(terminal_errors), 'at': time.time()}
        if (root / 'complete.json').exists():
            raise ValueError('GLOBAL boost already finalized; preserve its completion marker')
        atomic_new(root / 'complete.json', marker)
        write_json(status_path, {'stage': 'complete', **marker})
        return marker


def launch(config, *, restart=False):
    root = root_for(config)
    worker = root / 'finalizer_worker.json'
    if worker.exists():
        old = read_json(worker)
        process = Path(f'/proc/{old["pid"]}/cmdline')
        if process.exists() and b'verak.v3.cli.global_boost' in process.read_bytes() and b'finalize' in process.read_bytes():
            if not restart:
                return old
            # This is only a CPU/file-queue consumer: it owns no API reservation,
            # GPU model, or generation state. Measured files are atomic.
            os.kill(old['pid'], signal.SIGTERM)
            for _ in range(50):
                if not process.exists() or not process.read_bytes():
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Owned CPU finalizer did not stop')
    with (root / 'finalization.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'verak.v3.cli.global_boost', 'finalize'],
            cwd=str(Path(__file__).resolve().parents[3]), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': process.pid, 'at': time.time(), 'gpu_used': False, 'training': False}
    write_json(worker, value)
    return value
