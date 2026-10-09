"""Frozen CPU/GPU consistency sample: 200 distinct source essays, no new GPU calls."""
from collections import Counter, defaultdict
import json
from math import floor
from pathlib import Path
from statistics import mean

from ..common import file_sha, pair_key, read_json, sha_text, write_json
from ..phase2 import read_jsonl
from ..reward.total import quality_reward
from ..train.teacher_bulk import atomic_new
from .calibrate import layout_text
from .cpu_score import queue_cpu, score_cpu, shared_root

SEED = 83
SAMPLE_SIZE = 200


def stable_rank(*values):
    return sha_text(json.dumps([SEED, *values], ensure_ascii=False))


def sample_sources(frame, count=SAMPLE_SIZE):
    """Genre-proportional, half near .80 when available; CPU outcomes never enter."""
    if len({row['source_id'] for row in frame}) != len(frame) or len(frame) < count:
        raise ValueError('At least 200 distinct source essays are required')
    groups = defaultdict(list)
    for row in frame:
        groups[row['genre']].append(row)
    shares = {g: count*len(rows)/len(frame) for g, rows in groups.items()}
    quotas = {g: floor(value) for g, value in shares.items()}
    for genre in sorted(groups, key=lambda g: (-(shares[g]-quotas[g]), g))[:count-sum(quotas.values())]:
        quotas[genre] += 1
    selected = []
    for genre, rows in sorted(groups.items()):
        near = sorted((r for r in rows if r['near_threshold']), key=lambda r: stable_rank(r['source_id']))
        far = sorted((r for r in rows if not r['near_threshold']), key=lambda r: stable_rank(r['source_id']))
        n = quotas[genre]
        take_near = min(len(near), (n+1)//2)
        take_far = min(len(far), n-take_near)
        take_near = min(len(near), n-take_far)
        selected.extend(near[:take_near]+far[:take_far])
    if len(selected) != count:
        raise AssertionError('The fixed sampling quotas did not fill')
    return sorted(selected, key=lambda r: stable_rank('order', r['source_id']))


def prepare(config):
    root = shared_root(config) / 'cpu_scorer/audit_200'
    destination = root / 'plan.json'
    if destination.exists():
        return read_json(destination)
    prior = config['paths']['repo'] / 'verak/v3/outputs/teacher_bulk_two_stage'
    design_path = prior / 'design.json'
    design = read_json(design_path)
    corpus_path = config['paths']['active_corrupt'] / 'agent_train.jsonl'
    corpus = {r['episode_id']: r for r in read_jsonl(corpus_path)}
    frame, exclusions = {}, Counter()
    for attempt in (1, 2):
        for eid in sorted(design['orders'][str(attempt)]):
            reused = design['reuse_attempt_1'].get(eid) if attempt == 1 else None
            path = (Path(reused['path']) if reused else
                    prior / f'attempt_{attempt}/episodes' / (eid.replace(':', '_')+'.json'))
            if not path.exists():
                exclusions['missing_saved_attempt'] += 1
                continue
            row = read_json(path)
            if not row['completed'] or not row.get('reward') or not all(
                    row.get(k) for k in ('initial_layout', 'stage1_layout', 'final_layout')):
                exclusions['incomplete_or_unscored_attempt'] += 1
                continue
            saved = {item['text_hash']: item['result'] for item in row['score_calls']}
            texts = [layout_text(row[k]) for k in ('initial_layout', 'stage1_layout', 'final_layout')]
            if any(sha_text(text) not in saved for text in texts):
                exclusions['missing_saved_score'] += 1
                continue
            sid = row['source_id']
            role_r = {role: row['reward'][role]['R'] for role in ('global', 'korean')}
            info = {'source_id': sid, 'episode_id': eid, 'attempt': attempt,
                'path': str(path), 'sha256': file_sha(path), 'genre': row['genre'],
                'role_R': role_r, 'near_threshold': min(abs(v-.8) for v in role_r.values()) <= .05,
                'has_global_records': any(r['level'] == 'GLOBAL' for r in corpus[eid]['records'])}
            old = frame.get(sid)
            if old is None or stable_rank(sid, eid, attempt) < stable_rank(sid, old['episode_id'], old['attempt']):
                frame[sid] = info
    frame_rows = sorted(frame.values(), key=lambda r: r['source_id'])
    selected = sample_sources(frame_rows)
    episodes = []
    unique = {}
    for info in selected:
        row = read_json(info['path'])
        question = corpus[info['episode_id']]['question']
        by_hash = {c['text_hash']: c['result'] for c in row['score_calls']}
        states = {}
        for name, key in [('initial', 'initial_layout'), ('middle', 'stage1_layout'), ('final', 'final_layout')]:
            text = layout_text(row[key])
            gpu = by_hash[sha_text(text)]
            request_key = pair_key(question, text)
            if gpu['cache_key'] != request_key:
                raise ValueError('GPU score does not match exact question/text input')
            states[name] = {'text': text, 'text_sha256': sha_text(text), 'request_key': request_key, 'gpu_score': gpu}
            unique[request_key] = gpu['input_tokens']
        episodes.append({**info, 'question': question, 'states': states, 'gpu_reward': row['reward']})
    plan = {'schema_version': 1, 'sample_seed': SEED, 'minimum_essays': SAMPLE_SIZE,
        'source_essays': len(episodes), 'state_comparisons': 3*len(episodes), 'unique_score_inputs': len(unique),
        'input_tokens': {'total_unique': sum(unique.values()), 'mean_unique': mean(unique.values()),
                         'max_unique': max(unique.values())},
        'selection': 'One completed two-stage attempt per source by seeded hash, independent of reward; '
            '200 sources by genre-proportional quotas, within genre up to half with either role R within .05 '
            'of .80 and the rest outside that band, seeded hash ordering. Sampling uses saved GPU outcomes only.',
        'interpretation': 'Stratified threshold-stress audit; not a random population accuracy estimate.',
        'frame_source_essays': len(frame_rows), 'frame_excluded_attempts': dict(exclusions),
        'frame_genres': dict(Counter(r['genre'] for r in frame_rows)),
        'frame_near_threshold': sum(r['near_threshold'] for r in frame_rows),
        'sample_genres': dict(Counter(r['genre'] for r in selected)),
        'sample_near_threshold': sum(r['near_threshold'] for r in selected),
        'frame_sha256': sha_text(json.dumps(frame_rows, sort_keys=True, ensure_ascii=False)),
        'bulk_design_path': str(design_path), 'bulk_design_sha256': file_sha(design_path),
        'corpus_path': str(corpus_path), 'corpus_sha256': file_sha(corpus_path),
        'gpu_inputs': 'Immutable saved bulk two-stage and reused pilot2 GPU-scored trajectories; zero new GPU calls.',
        'episodes': episodes, 'gpu_used': False, 'paid_calls': 0}
    atomic_new(root / 'frame.json', frame_rows)
    atomic_new(destination, plan)
    return plan


def compare_episode(config, episode, scores):
    comparisons, decisions = [], []
    for name, item in episode['states'].items():
        cpu, gpu = scores[name], item['gpu_score']
        if any(len(value) != 8 for value in (cpu['integers'], gpu['integers'],
                cpu['score_line'].split(), gpu['score_line'].split())):
            raise ValueError('Per-rubric agreement requires exactly eight score digits')
        comparisons.append({'source_id': episode['source_id'], 'episode_id': episode['episode_id'],
            'state': name, 'request_key': item['request_key'], 'gpu': gpu, 'cpu': cpu,
            'abs_mean_delta': abs(cpu['mean']-gpu['mean']),
            'digit_agreement': [a == b for a, b in zip(cpu['integers'], gpu['integers'])],
            'generated_digit_agreement': [a == b for a, b in zip(cpu['score_line'].split(), gpu['score_line'].split())],
            'score_line_equal': cpu['score_line'] == gpu['score_line']})
    for role, before, after in [('global', 'initial', 'middle'), ('korean', 'middle', 'final')]:
        old = episode['gpu_reward'][role]
        quality = quality_reward(scores[before]['mean'], scores[after]['mean'], episode['genre'], config['reward'])
        new_r = old['R']+config['reward'].get('w_q', .3)*(quality['value']-old['R_q'])
        decisions.append({'source_id': episode['source_id'], 'episode_id': episode['episode_id'], 'role': role,
            'has_global_records': episode['has_global_records'], 'gpu_R': old['R'], 'cpu_R': new_r,
            'gpu_R_q': old['R_q'], 'cpu_R_q': quality['value'],
            'dead_zone_changed': (old['R_q'] == 0) != (quality['value'] == 0),
            'threshold_changed': (old['R'] >= .8) != (new_r >= .8),
            'gpu_eligible_at_80': old['R'] >= .8, 'cpu_eligible_at_80': new_r >= .8})
    return {'comparisons': comparisons, 'decisions': decisions}


def summarize(plan, results, errors):
    comparisons = [r for episode in results for r in episode['comparisons']]
    decisions = [r for episode in results for r in episode['decisions']]
    unique = {r['request_key']: r for r in comparisons}
    fingerprints = sorted({r['cpu']['scorer_fingerprint'] for r in comparisons})
    return {'schema_version': 1, 'status': 'complete' if len(results) == SAMPLE_SIZE and not errors and len(fingerprints) == 1 else 'incomplete',
        'source_essays': len(results), 'unique_source_essays': len(results),
        'required_essays': SAMPLE_SIZE, 'state_comparisons': len(comparisons),
        'unique_score_inputs_completed': len(unique), 'sample_genres': plan['sample_genres'],
        'sample_near_threshold': plan['sample_near_threshold'], 'frame_source_essays': plan['frame_source_essays'],
        'mean_abs_Q_delta': mean(r['abs_mean_delta'] for r in comparisons) if comparisons else None,
        'max_abs_Q_delta': max((r['abs_mean_delta'] for r in comparisons), default=None),
        'unique_input_mean_abs_Q_delta': mean(r['abs_mean_delta'] for r in unique.values()) if unique else None,
        'argmax_digit_agreement_by_rubric': [sum(r['digit_agreement'][i] for r in comparisons)/len(comparisons)
            for i in range(8)] if comparisons else [],
        'generated_digit_agreement_by_rubric': [sum(r['generated_digit_agreement'][i] for r in comparisons)/len(comparisons)
            for i in range(8)] if comparisons else [],
        'score_line_changes': sum(not r['score_line_equal'] for r in comparisons),
        'selection_changes_by_role': {role: sum(r['threshold_changed'] for r in decisions if r['role'] == role)
            for role in ('global', 'korean')},
        'global_record_selection_changes': sum(r['threshold_changed'] for r in decisions
            if r['role'] == 'global' and r['has_global_records']),
        'max_abs_R_delta': max((abs(r['cpu_R']-r['gpu_R']) for r in decisions), default=None),
        'dead_zone_changes': sum(r['dead_zone_changed'] for r in decisions),
        'fingerprints': fingerprints,
        'unknown_essays': SAMPLE_SIZE-len(results), 'errors': errors,
        'selection_policy': 'CPU results are provisional only; all teacher selections require later GPU reference values.',
        'numeric_limit': 'Observed sample errors are not a universal tolerance or an error bound.',
        'gpu_used': False, 'paid_calls': 0}


def run(config):
    root = shared_root(config) / 'cpu_scorer/audit_200'
    plan = prepare(config)
    plan_sha = file_sha(root / 'plan.json')
    for episode in plan['episodes']:
        for state in episode['states'].values():
            queue_cpu(config, episode['question'], state['text'], requester='audit_200')
    results, errors = [], []
    for episode in plan['episodes']:
        path = root / 'essays' / (episode['source_id'].replace(':', '_')+'.json')
        try:
            if file_sha(episode['path']) != episode['sha256']:
                raise ValueError('GPU reference trajectory changed')
            if path.exists():
                result = read_json(path)
                if result['plan_sha256'] != plan_sha:
                    raise ValueError('Frozen CPU calibration plan changed')
            else:
                scores = {name: score_cpu(config, episode['question'], state['text'], requester='audit_200')
                          for name, state in episode['states'].items()}
                result = {**compare_episode(config, episode, scores), 'plan_sha256': plan_sha}
                atomic_new(path, result)
            results.append(result)
        except Exception as exc:
            errors.append({'source_id': episode['source_id'], 'type': type(exc).__name__, 'message': str(exc)})
        progress = {**summarize(plan, results, errors), 'plan_sha256': plan_sha}
        write_json(root / 'progress.json', progress)
    report = {**summarize(plan, results, errors), 'plan_sha256': plan_sha}
    write_json(root / 'calibration.json', report)
    return report
