"""Original v1 policy observations/actions; only hidden reward computation is deferred."""
import json
import random
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from ..agent.runner import run_episode, system_prompt
from ..common import file_sha, read_json, write_json
from ..env import RevisionEnv
from ..train.teacher_bulk import atomic_new, BulkTeacher, collection_lock
from ..v2_ops.local import load_environment
from .config import PHASE
from .paid import BoostAPI
from .prepare import corpus, safe_id


class DeferredRewardEnv(RevisionEnv):
    """No CHECK exists, so postponing hidden post-episode scoring changes no turn."""
    def reward(self):
        if self.check_enabled:
            raise ValueError('Deferred rewards only support accepted no-CHECK v1')
        return None


def attempt_path(root, attempt, episode_id):
    return root / f'attempt_{attempt}/episodes' / (safe_id(episode_id) + '.json')


def prepare_teacher(config):
    root = config['paths'][PHASE + '_output']
    rows = {}
    unjudged = []
    for episode_id, row in corpus(config).items():
        path = root / 'qc' / (safe_id(episode_id) + '.json')
        failure = root / 'qc_errors' / path.name
        if not path.exists() and not failure.exists():
            if not config[PHASE].get('allow_unjudged_qc'):
                raise ValueError('Finish QC before freezing the teacher corpus')
            unjudged.append(episode_id)
            continue
        if path.exists() and read_json(path)['passed']:
            rows[episode_id] = row
    orders = {}
    for attempt, seed in enumerate(config[PHASE]['ordering_seeds'], 1):
        ids = sorted(rows)
        random.Random(seed).shuffle(ids)
        orders[str(attempt)] = ids
    design = {'phase': PHASE, 'method': 'two_stage_v1', 'reasoning': 'low',
        'model': config[PHASE]['model'], 'context_limit': 8192, 'output_limit': 1024,
        'attempts': 2, 'provider_sampling_seed': None,
        'ordering_seeds': config[PHASE]['ordering_seeds'], 'orders': orders,
        'source_plan_sha256': file_sha(root / 'source_plan.json'),
        'prompt': {role: system_prompt(role) for role in ('global', 'korean')},
        'reward': 'Exact v1 formulas after generation, frozen CPU NF4 scorer; no reward in the action observations.',
        'selected_roles': ['global'], 'teacher_roles': ['global', 'korean'],
        'teacher_workers': 1, 'gpu_used': False, 'training': False}
    if config[PHASE].get('allow_unjudged_qc'):
        design['unjudged_qc_excluded'] = unjudged
    path = root / 'teacher_design.json'
    if path.exists() and read_json(path) != design:
        raise ValueError('Frozen teacher corpus/contract changed')
    write_json(path, design)
    return design, rows


def run(config, *, max_api_calls=20000, limit=None):
    from .resources import CPUResources
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        design, rows = prepare_teacher(config)
        design_sha = file_sha(root / 'teacher_design.json')
        resources = CPUResources(config)
        api = BoostAPI(config, max_api_calls, kind='luna')
        api.settle_interrupted()
        tasks = [(a, design['orders'][str(a)][i]) for i in range(len(rows)) for a in (1, 2)]
        tasks = [(a, i) for a, i in tasks if not attempt_path(root, a, i).exists()]
        if limit is not None:
            tasks = tasks[:limit]
        completed, errors, stop_reason = [], [], 'all_slots_attempted'
        try:
            for attempt, episode_id in tasks:
                row = rows[episode_id]
                source = resources.source(row)
                document = resources.restore(row['corrupted_layout'])
                episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre',
                    'level', 'records', 'preexisting_spell_spans')}
                episode.update(source=source, document=document)
                env = DeferredRewardEnv(config, mode='two_stage', analysis=resources.analysis,
                                        tokenizer=resources.tokenizer)
                backend = BulkTeacher(api, attempt, resources.tokenizer)
                backend.condition = f'global_boost_attempt_{attempt}'
                result = run_episode(env, episode, backend,
                    event_path=root / f'attempt_{attempt}/events' / (safe_id(episode_id) + '.jsonl'))
                result['generation_completed'] = result['completed']
                result['reward_status'] = 'deferred_cpu_scoring'
                result['data_boost'] = {'attempt': attempt, 'design_sha256': design_sha,
                    'operator': row['operator'], 'provider_sampling_seed': None,
                    'ordering_seed': config[PHASE]['ordering_seeds'][attempt-1],
                    'selected_role': 'global', 'gpu_used': False}
                with api.db() as db:
                    requests = db.execute('SELECT path FROM calls WHERE stage=? AND item_id LIKE ?',
                        (backend.condition, result['episode_id'] + ':%')).fetchall()
                result['confirmed_episode_cost'] = sum(read_json(p[0]).get('cost', {}).get('confirmed_usd', 0)
                                                       for p in requests if p[0])
                atomic_new(attempt_path(root, attempt, episode_id), result)
                if config[PHASE].get('prefetch_saved'):
                    from .prefetch import enqueue
                    enqueue(config)
                completed.append([attempt, episode_id])
                if result['runtime_error']:
                    errors.append({'attempt': attempt, 'episode_id': episode_id, **result['runtime_error']})
                write_json(root / 'teacher_progress.json', {'finished_this_run': len(completed),
                    'requested_attempts': 2*len(rows), 'errors': errors, 'budget': api.accounting(), 'at': time.time()})
                print(json.dumps({'attempt': attempt, 'id': episode_id, 'completed': result['completed'],
                    'error': result['runtime_error'], 'budget': api.accounting()}), flush=True)
                if result['runtime_error'] and result['runtime_error']['type'] == 'CallBudgetExceeded':
                    stop_reason = 'budget_cap'
                    break
                if resources.analysis.suspended():
                    stop_reason = 'bareun_priority_suspended'
                    break
            if limit is not None and stop_reason == 'all_slots_attempted':
                stop_reason = 'requested_limit'
        finally:
            api.close()
            write_json(root / 'teacher_status.json', {'finished_this_run': len(completed),
                'requested_attempts': 2*len(rows), 'errors': errors, 'stop_reason': stop_reason,
                'budget': api.accounting(), 'gpu_used': False, 'training': False, 'at': time.time()})
        return read_json(root / 'teacher_status.json')
