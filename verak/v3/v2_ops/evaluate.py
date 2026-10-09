"""Deferred GPU1-only scoring; all semantic recovery judgments must already be cached."""
from copy import deepcopy

from ..common import file_sha, read_json, write_json
from ..train.teacher_bulk import atomic_new, collection_lock
from ..phase2 import write_jsonl
from .config import PHASE, require_scorer_slot
from .qc import candidate_paths, summary
from .local import load_environment
from .reward import rewards_v2
from .teacher import CPUResources, RecoveryJudge, prepare, attempt_path


def run(config):
    gpu = require_scorer_slot(config)  # Before importing/loading any GPU model.
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        design, corpus = prepare(config)
        resources = CPUResources(config)
        cfg = deepcopy(config)
        cfg['scorer']['gpu'] = gpu
        cfg['paths']['output'] = root / 'scorer'
        from ..score.kanana import KananaScorer
        scorer, errors, done = None, [], 0
        try:
            qc = summary(config)
            exported = {'agent_train': [], 'agent_dev': []}
            for candidate_path in candidate_paths(config):
                row = read_json(candidate_path)
                if qc['operators'][row['operator']]['decision'] != 'retain':
                    continue
                verdict_path = root / 'qc' / (row['episode_id'].replace(':', '_') + '.json')
                if not verdict_path.exists() or not read_json(verdict_path)['passed']:
                    continue
                score_path = root / 'corrupted_scores' / (row['episode_id'].replace(':', '_') + '.json')
                try:
                    if score_path.exists():
                        score = read_json(score_path)
                        if score['corrupted_hash'] != row['corrupted_hash']:
                            raise ValueError('Corrupted score provenance changed')
                    else:
                        if scorer is None:
                            scorer = KananaScorer(cfg)
                        score = {'corrupted_hash': row['corrupted_hash'],
                                 'score': scorer.score(row['question'], row['corrupted_text']).to_dict()}
                        atomic_new(score_path, score)
                    exported[row['split']].append({**row, 'corrupted_score': score['score'],
                                                   'q_corrupted': score['score']['mean']})
                except Exception as exc:
                    errors.append({'stage': 'corrupted_score', 'episode_id': row['episode_id'],
                                   'type': type(exc).__name__, 'message': str(exc)})
            for split, rows in exported.items():
                write_jsonl(config['paths']['v2_corpus'] / (split + '.jsonl'), rows)
            write_json(root / 'corpus_export.json', {'counts': {s: len(v) for s, v in exported.items()},
                'errors': list(errors), 'device': 'physical_GPU1', 'source_plan_sha256': file_sha(root / 'source_plan.json')})
            for attempt in (1, 2):
                for episode_id in design['orders'][str(attempt)]:
                    path = attempt_path(root, attempt, episode_id)
                    if not path.exists():
                        continue
                    out = root / f'scored/attempt_{attempt}' / path.name
                    if out.exists():
                        if read_json(out)['raw_episode_sha256'] != file_sha(path):
                            raise ValueError('Saved raw trajectory changed after scoring')
                        continue
                    result = read_json(path)
                    if result.get('stage1_layout') is None:
                        continue
                    candidate = corpus[episode_id]
                    source = resources.source(candidate)
                    try:
                        corrupted = resources.restore(candidate['corrupted_layout'])
                        stage1 = resources.restore(result['stage1_layout'])
                        final = resources.restore(result['final_layout']) if result['completed'] else stage1
                        judge = RecoveryJudge(config, candidate)  # Cannot make an API call in this process.
                        # Check reference judgments before allocating scorer time.
                        from .teacher import recover_saved
                        stages = recover_saved(config, candidate, result, resources)
                        write_json(root / f'attempt_{attempt}/recovery' / path.name,
                                   {'episode_sha256': file_sha(path), 'stages': stages})
                        if scorer is None:
                            scorer = KananaScorer(cfg)
                        values = [scorer.score(candidate['question'], d.text).to_dict()
                                  for d in (corrupted, stage1, final)]
                        actions = result['actions_by_role']
                        reward = rewards_v2(source, corrupted, final, candidate['records'], config=config,
                            genre=candidate['genre'], q_corrupted=values[0]['mean'], q_stage1=values[1]['mean'],
                            q_final=values[2]['mean'], stage1=stage1, stage1_actions=actions['global'],
                            stage2_actions=actions['korean'] if result['completed'] else [], judge=judge,
                            preexisting_spell_spans=candidate['preexisting_spell_spans'])
                        enriched = {**result, 'raw_episode_path': str(path), 'raw_episode_sha256': file_sha(path),
                            'quality_scores': values, 'score_device': 'physical_GPU1',
                            'reward': reward if result['completed'] else None,
                            'global_only_reward': reward['global'] if not result['completed'] else None}
                        atomic_new(out, enriched)
                        done += 1
                    except Exception as exc:
                        errors.append({'attempt': attempt, 'episode_id': episode_id,
                                       'type': type(exc).__name__, 'message': str(exc)})
                    write_json(root / 'scoring_status.json', {'scored_this_run': done, 'errors': errors,
                               'device': 'physical_GPU1', 'paid_calls': 0, 'training': False})
        finally:
            if scorer is not None:
                scorer.close()
        status = {'scored_this_run': done, 'errors': errors, 'device': 'physical_GPU1',
                  'model_loaded': scorer is not None, 'gpu_used': scorer is not None,
                  'exported_counts': {s: len(v) for s, v in exported.items()},
                  'paid_calls': 0, 'training': False}
        write_json(root / 'scoring_status.json', status)
        return status
