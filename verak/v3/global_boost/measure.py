"""Evaluate saved turns with unmodified v1 recovery/reward on the CPU scorer queue."""
import time

from ..common import file_sha, read_json, write_json
from ..reward.total import rewards
from ..train.teacher_bulk import atomic_new
from .config import PHASE
from .teacher import attempt_path, prepare_teacher


def measured_path(config, attempt, path):
    root = config['paths'][PHASE + '_output']
    fingerprint = config[PHASE].get('score_fingerprint')
    directory = 'measured_' + fingerprint[:12] if fingerprint else 'measured'
    return root / f'attempt_{attempt}' / directory / path.name


def run(config, limit=None):
    from .resources import CPUResources, score_cpu
    root = config['paths'][PHASE + '_output']
    design, rows = prepare_teacher(config)
    resources = CPUResources(config)
    saved, errors = [], []
    tasks = [(a, episode_id) for a in (1, 2) for episode_id in design['orders'][str(a)]
             if attempt_path(root, a, episode_id).exists()
             and not measured_path(config, a, attempt_path(root, a, episode_id)).exists()]
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
                for name, document in [('corrupted', corrupted), ('stage1', stage1)]:
                    scores[name] = score_cpu(config, candidate['question'], document.text)
                    expected = config[PHASE].get('score_fingerprint')
                    if expected and scores[name].get('execution_device') != 'saved_gpu_cache' and scores[name].get('scorer_fingerprint') != expected:
                        raise RuntimeError('CPU scorer result does not match the approved fingerprint')
                # The user requested GLOBAL data only. The GLOBAL term depends
                # on the actual corrupted and stage-1 states, never stage 2.
                # Reuse the existing partial-GLOBAL formula and retain only it.
                value = rewards(source, corrupted, stage1,
                    candidate['records'], genre=candidate['genre'],
                    q_corrupted=scores['corrupted']['mean'], q_stage1=scores['stage1']['mean'],
                    q_final=scores['stage1']['mean'], config=config['reward'], mode='two_stage',
                    stage1=stage1, stage1_actions=raw['actions_by_role']['global'],
                    stage2_actions=[],
                    similarity=None, tau=config['similarity']['tau'],
                    preexisting_spell_spans=candidate['preexisting_spell_spans'])
                result['global_only_reward'] = value['global']
            result.update(reward_status='global_only_measured_cpu', cpu_scores=scores,
                          unmeasured_roles=['korean', 'combined'],
                          raw_generation_sha256=file_sha(path), gpu_used=False)
            result['canonical_for_selection'] = bool(config[PHASE].get('score_fingerprint'))
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
