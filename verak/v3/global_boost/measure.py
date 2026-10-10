"""Evaluate saved turns with unmodified v1 recovery/reward on the CPU scorer queue."""
import time

from ..common import file_sha, read_json, write_json
from ..reward.total import rewards, _breakdown
from ..reward.recovery import recover_records
from ..reward.overedit import overedit
from ..train.teacher_bulk import atomic_new
from .config import PHASE
from .teacher import attempt_path, prepare_teacher
from .identity import identity_quality


def measured_path(config, attempt, path):
    root = config['paths'][PHASE + '_output']
    fingerprint = config[PHASE].get('score_fingerprint')
    source = config[PHASE].get('measurement_source', 'measured')
    directory = source + '_' + fingerprint[:12] if fingerprint else 'measured'
    return root / f'attempt_{attempt}' / directory / path.name


def run(config, limit=None):
    from .resources import CPUResources, score_cpu, score_gpu_reference
    root = config['paths'][PHASE + '_output']
    design, rows = prepare_teacher(config)
    resources = CPUResources(config)
    saved, errors = [], []
    attempts = sorted(int(p.name.split('_')[-1]) for p in root.glob('attempt_*') if p.is_dir())
    tasks = [(attempt, episode_id) for attempt in attempts for episode_id in rows
             if attempt_path(root, attempt, episode_id).exists()
             and not measured_path(config, attempt, attempt_path(root, attempt, episode_id)).exists()]
    if limit is not None:
        tasks = tasks[:limit]
    for attempt, episode_id in tasks:
        path = attempt_path(root, attempt, episode_id)
        raw, candidate = read_json(path), rows[episode_id]
        target = measured_path(config, attempt, path)
        try:
            source = resources.source(candidate)
            corrupted = resources.restore(candidate['corrupted_layout'])
            stage1 = resources.restore(raw['stage1_layout']) if raw.get('stage1_layout') else None
            scores = {}
            result = dict(raw)
            if stage1 is not None:
                gpu_reference = config[PHASE].get('measurement_source') == 'gpu_reference'
                identity = None if gpu_reference else identity_quality(config, candidate['genre'], candidate['question'],
                    corrupted.text, stage1.text, scorer_contract={'fingerprint': config[PHASE].get('score_fingerprint'),
                        'scorer': config['scorer'], 'base': str(config['paths']['policy_base']),
                        'adapter': str(config['paths']['scorer_adapter'])})
                if identity:
                    records = recover_records(source, stage1, candidate['records'], corrupted=corrupted,
                        similarity=None, tau=config['similarity']['tau'], coupled_weight=config['reward'].get('dependents_weight', .3))
                    records = [r for r in records if r['level'] == 'GLOBAL']
                    result['global_only_reward'] = _breakdown([r['main'] for r in records], records, identity,
                        overedit(source, corrupted, stage1, candidate['records'], actions=raw['actions_by_role']['global'],
                            preexisting_spell_spans=candidate['preexisting_spell_spans']),
                        len(raw['actions_by_role']['global']), config['reward'])
                    result['quality_identity'] = identity['identity']
                else:
                    scorer = score_gpu_reference if gpu_reference else score_cpu
                    for name, document in [('corrupted', corrupted), ('stage1', stage1)]:
                        scores[name] = scorer(config, candidate['question'], document.text)
                        expected = config[PHASE].get('score_fingerprint')
                        if expected and scores[name].get('execution_device') != 'saved_gpu_cache' and scores[name].get('scorer_fingerprint') != expected:
                            raise RuntimeError('Scorer result does not match the configured fingerprint')
                    # GLOBAL depends on corrupted and stage 1, never stage 2.
                    value = rewards(source, corrupted, stage1,
                        candidate['records'], genre=candidate['genre'],
                        q_corrupted=scores['corrupted']['mean'], q_stage1=scores['stage1']['mean'],
                        q_final=scores['stage1']['mean'], config=config['reward'], mode='two_stage',
                        stage1=stage1, stage1_actions=raw['actions_by_role']['global'],
                        stage2_actions=[],
                        similarity=None, tau=config['similarity']['tau'],
                        preexisting_spell_spans=candidate['preexisting_spell_spans'])
                    result['global_only_reward'] = value['global']
            result.update(reward_status='global_only_' + config[PHASE].get('measurement_source', 'cpu_provisional'),
                          cpu_scores=scores, quality_scores=scores,
                          score_source=config[PHASE].get('measurement_source', 'cpu_provisional'),
                          unmeasured_roles=['korean', 'combined'],
                          raw_generation_sha256=file_sha(path), gpu_used=False)
            result['canonical_for_selection'] = config[PHASE].get('measurement_source') == 'gpu_reference'
            result['scorer_approval_sha256'] = config[PHASE].get('scorer_approval_sha256')
            atomic_new(target, result)
            saved.append([attempt, episode_id])
        except Exception as exc:
            error = {'attempt': attempt, 'episode_id': episode_id, 'type': type(exc).__name__,
                     'message': str(exc), 'raw_generation_sha256': file_sha(path)}
            write_json(root / f'attempt_{attempt}/measurement_errors' / path.name, error)
            errors.append(error)
            # Source profiles and model arithmetic must not be silently replaced.
            if resources.analysis.suspended() or isinstance(exc, TimeoutError) or 'CPU scorer unavailable' in str(exc):
                break
        write_json(root / 'measurement_progress.json', {'measured_this_run': len(saved),
            'errors': errors, 'at': time.time(), 'gpu_used': False})
        print(f'CPU measured {len(saved)}/{len(tasks)}; errors={len(errors)}', flush=True)
    result = {'measured_this_run': len(saved), 'errors': errors, 'gpu_used': False, 'training': False,
              'at': time.time()}
    write_json(root / 'measurement_status.json', result)
    return result
