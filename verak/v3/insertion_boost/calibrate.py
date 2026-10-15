"""Free CPU arithmetic audit against immutable saved GPU score calls."""
from ..common import file_sha, read_json, sha_text, write_json
from ..phase2 import read_jsonl
from ..reward.total import quality_reward
from ..train.teacher_bulk import atomic_new
from .cpu_score import queue_cpu, score_cpu, shared_root


def layout_text(layout):
    return ''.join(gap + ''.join(u['leading']+u['text'] for u in p['units'])
                   for gap, p in zip(layout['gaps'], layout['paragraphs'])) + layout['tail']


def prepare(config):
    root = shared_root(config) / 'cpu_scorer'
    prior = config['paths']['phase7_teacher_output']
    design = read_json(prior / 'design.json')
    corpus = {r['episode_id']: r for r in read_jsonl(config['paths']['active_corrupt'] / 'agent_train.jsonl')}
    episodes = []
    for eid in sorted(design['pilot_ids']):
        path = prior / 'luna_low/episodes' / (eid.replace(':', '_') + '.json')
        row = read_json(path)
        if row['completed'] and row['reward']:
            episodes.append((eid, path, row))
    chosen = {}
    for genre in sorted({r['genre'] for _, _, r in episodes}):
        item = next(item for item in episodes if item[2]['genre'] == genre)
        chosen[item[0]] = item
    for role in ('global', 'korean'):
        item = min(episodes, key=lambda item: (abs(item[2]['reward'][role]['R']-.8), item[0]))
        chosen[item[0]] = item
    rows = []
    for eid, path, row in sorted(chosen.values()):
        by_hash = {c['text_hash']: c['result'] for c in row['score_calls']}
        states = {}
        for name, key in [('initial', 'initial_layout'), ('middle', 'stage1_layout'), ('final', 'final_layout')]:
            text = layout_text(row[key])
            states[name] = {'text': text, 'text_sha256': sha_text(text), 'gpu_score': by_hash[sha_text(text)]}
        rows.append({'episode_id': eid, 'path': str(path), 'sha256': file_sha(path),
            'question': corpus[eid]['question'], 'genre': row['genre'], 'states': states,
            'gpu_reward': row['reward']})
    plan = {'selection': 'lexicographically first complete episode per genre plus nearest GLOBAL/KOREAN R=.80',
        'gpu_inputs': 'saved 92 pilot2 Luna low trajectories only; no new GPU calls', 'episodes': rows}
    destination = root / 'calibration_plan.json'
    if destination.exists() and read_json(destination) != plan:
        raise ValueError('CPU calibration selection changed')
    if not destination.exists():
        atomic_new(destination, plan)
    return plan


def run(config):
    root = shared_root(config) / 'cpu_scorer'
    plan = prepare(config)
    prefetch(config, plan=plan)
    comparisons, decisions = [], []
    for episode in plan['episodes']:
        scores = {}
        for name, item in episode['states'].items():
            cpu = score_cpu(config, episode['question'], item['text'], requester='calibration')
            scores[name] = cpu
            gpu = item['gpu_score']
            comparisons.append({'episode_id': episode['episode_id'], 'state': name,
                'text_sha256': item['text_sha256'], 'gpu': gpu, 'cpu': cpu,
                'mean_delta': cpu['mean']-gpu['mean'],
                'score_line_equal': cpu['score_line'] == gpu['score_line'],
                'max_expected_delta': max(abs(a-b) for a, b in zip(cpu['expected'], gpu['expected']))})
        for role, before, after in [('global', 'initial', 'middle'), ('korean', 'middle', 'final'),
                                    ('combined', 'initial', 'final')]:
            old = episode['gpu_reward'][role]
            quality = quality_reward(scores[before]['mean'], scores[after]['mean'], episode['genre'], config['reward'])
            new_r = old['R'] + config['reward'].get('w_q', .3)*(quality['value']-old['R_q'])
            decisions.append({'episode_id': episode['episode_id'], 'role': role, 'gpu_R': old['R'],
                'cpu_R': new_r, 'gpu_quality': old['quality'], 'cpu_quality': quality,
                'dead_zone_changed': (old['R_q'] == 0) != (quality['value'] == 0),
                'threshold_changed': (old['R'] >= .8) != (new_r >= .8)})
        write_json(root / 'calibration_progress.json', {'comparisons': comparisons, 'decisions': decisions})
    result = {'plan_sha256': file_sha(root / 'calibration_plan.json'), 'episodes': len(plan['episodes']),
        'comparisons': comparisons, 'decisions': decisions,
        'max_abs_mean_delta': max(abs(r['mean_delta']) for r in comparisons),
        'max_abs_R_delta': max(abs(r['cpu_R']-r['gpu_R']) for r in decisions),
        'dead_zone_changes': sum(r['dead_zone_changed'] for r in decisions),
        'threshold_changes': sum(r['threshold_changed'] for r in decisions if r['role'] != 'combined'),
        'score_line_changes': sum(not r['score_line_equal'] for r in comparisons),
        'gpu_used': False, 'paid_calls': 0,
        'scope': 'numeric spot check, not proof that every new threshold decision matches GPU arithmetic'}
    write_json(root / 'calibration.json', result)
    return {k: v for k, v in result.items() if k not in {'comparisons', 'decisions'}}


def prefetch(config, plan=None):
    plan = plan or prepare(config)
    requested = set()
    for episode in plan['episodes']:
        for item in episode['states'].values():
            path = queue_cpu(config, episode['question'], item['text'], requester='calibration')
            requested.add(str(path))
    return {'unique_reference_scores_queued': len(requested), 'paid_calls': 0, 'gpu_used': False}
