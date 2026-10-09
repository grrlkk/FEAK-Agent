"""Two independent pinned Luna attempts; same v2 agents, GLOBAL selection only."""
from pathlib import Path
import json
import random
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, write_json
from ..train.teacher_bulk import BulkTeacher, atomic_new, collection_lock
from ..v2_ops.config import PHASE
from ..v2_ops.environment import V2RevisionEnv
from ..v2_ops.local import load_environment
from ..v2_ops.paid import V2API
from ..v2_ops.prompts import system_prompt
from ..v2_ops.qc import candidate_paths
from ..v2_ops.runner import run_episode
from ..v2_ops.teacher import CPUResources, RecoveryJudge, attempt_path, safe_id
from ..v2_ops.reward import overedit_v2, recovery
from .data import summary
from .resources import BoostBank, BoostParagraphs
from .cpu_score import cached_gpu_score, queue_cpu
from .calibrate import layout_text


class Resources(CPUResources):
    def worker(self):
        if not hasattr(self.local, 'analysis'):
            root = self.config['paths'][PHASE + '_output']
            self.local.analysis = BoostParagraphs(self.config, cache_dir=root / 'bareun_paragraphs')
            self.local.bank = BoostBank(self.config, cache_dir=root / 'bareun_units')
            self.local.tokenizer = self.tokenizer_class.from_pretrained(
                str(self.config['paths']['policy_base']), local_files_only=True)
        return self.local


def prepare(config):
    root = config['paths'][PHASE + '_output']
    qc = summary(config)
    if qc['new_train'].get('awaiting_construction', 0):
        raise ValueError('Finish source labeling/construction before freezing teacher inputs')
    if qc['decision'] != 'retain':
        raise RuntimeError('Combined QC yield below 30%; notify root before further teacher dispatch')
    corpus, provenance = {}, {}
    source_plan = read_json(root / 'source_plan.json')
    items = list(source_plan['prior_passing_train'])
    for path in candidate_paths(config):
        row = read_json(path)
        verdict = root / 'qc' / (safe_id(row['episode_id']) + '.json')
        if verdict.exists() and read_json(verdict)['passed']:
            items.append({'path': str(path), 'sha256': file_sha(path), 'qc_path': str(verdict),
                          'qc_sha256': file_sha(verdict), 'episode_id': row['episode_id'], 'source_id': row['source_id']})
    sources = set()
    for item in items:
        if file_sha(item['path']) != item['sha256'] or file_sha(item['qc_path']) != item['qc_sha256']:
            raise ValueError('Passing teacher input or QC changed')
        row = read_json(item['path'])
        verdict = read_json(item['qc_path'])
        if row['split'] != 'agent_train' or not verdict['passed'] or verdict['candidate_sha256'] != item['sha256']:
            raise ValueError('Teacher input is not an immutable QC-passing train record')
        if row['source_id'] in sources:
            raise ValueError('Only one deletion record per train source is permitted')
        sources.add(row['source_id'])
        corpus[row['episode_id']] = row
        provenance[row['episode_id']] = item
    orders = {}
    for attempt, seed in enumerate(config[PHASE]['ordering_seeds'], 1):
        ids = sorted(corpus)
        random.Random(seed).shuffle(ids)
        orders[str(attempt)] = ids
    directory = Path(__file__).parents[1] / 'v2_ops'
    runtime = {name: file_sha(directory / name) for name in
        ('actions.py', 'protocol.py', 'prompts.py', 'environment.py', 'runner.py', 'reward.py', 'operators.py', 'judges.py')}
    design = {'version': 'v2', 'experiment': 'GLOBAL insertion data boost',
        'model': config[PHASE]['model'], 'reasoning': 'low', 'context': 8192,
        'output_limit': 1024, 'attempts': 2, 'sampling_seed': None,
        'ordering_seeds': config[PHASE]['ordering_seeds'], 'orders': orders,
        'selected_roles': ['global'], 'korean_behavior': 'unchanged; runs after GLOBAL handoff',
        'reward_scope': 'GLOBAL only; KOREAN and combined rewards unmeasured',
        'source_plan_sha256': file_sha(root / 'source_plan.json'), 'runtime_sha256': runtime,
        'prompt': {r: system_prompt(r) for r in ('global', 'korean')},
        'corpus': provenance, 'teacher_workers': 1, 'gpu_used': False, 'training': False}
    path = root / 'teacher_design.json'
    if path.exists() and read_json(path) != design:
        raise ValueError('Frozen insertion teacher design changed')
    if not path.exists():
        atomic_new(path, design)
    return design, corpus


def recover_global(config, candidate, result, resources, api=None):
    if result.get('stage1_layout') is None:
        return {}
    source = resources.source(candidate)
    corrupted = resources.restore(candidate['corrupted_layout'])
    middle = resources.restore(result['stage1_layout'])
    rows, matched, origins = recovery(source, corrupted, middle, candidate['records'],
        result['actions_by_role'].get('global', []), RecoveryJudge(config, candidate, api),
        coupled_weight=config['reward'].get('dependents_weight', .3))
    over = overedit_v2(source, corrupted, middle, candidate['records'],
        actions=result['actions_by_role'].get('global', []), origins=origins, matched=matched,
        preexisting_spell_spans=candidate['preexisting_spell_spans'])
    return {'global': {'per_record': rows, 'matched_insertions': sorted(matched),
                      'insert_origins': origins, 'R_over': over['value'], 'overedit': over}}


def run(config, *, max_api_calls, paid_approved, limit=None):
    if not paid_approved:
        raise PermissionError('Explicit user authorization required')
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        design, corpus = prepare(config)
        design_sha = file_sha(root / 'teacher_design.json')
        resources = Resources(config)
        api = V2API(config, max_api_calls, kind='luna', paid_approved=True)
        tasks = [(attempt, design['orders'][str(attempt)][i]) for i in range(len(corpus)) for attempt in (1, 2)]
        errors, done, recovery_done = [], 0, 0
        try:
            api.client()
            api.settle_interrupted()
            for row in corpus.values():
                if cached_gpu_score(config, row['question'], row['corrupted_text']) is None:
                    queue_cpu(config, row['question'], row['corrupted_text'], requester='insertion:corrupted_prefetch')
            for attempt, episode_id in tasks:
                path = attempt_path(root, attempt, episode_id)
                row = corpus[episode_id]
                if path.exists():
                    result = read_json(path)
                    if result['v2']['design_sha256'] != design_sha:
                        raise ValueError('Saved teacher design provenance changed')
                else:
                    if limit is not None and done >= limit:
                        break
                    state = resources.worker()
                    started = time.time()
                    episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre',
                        'level', 'records', 'preexisting_spell_spans')}
                    source = resources.source(row)
                    document = resources.restore(row['corrupted_layout'])
                    episode.update(source=source, document=document)
                    preflight = root / 'bareun_preflight.json'
                    if not preflight.exists():
                        atomic_new(preflight, {'episode_id': episode_id,
                            'needed_initial_analysis_seconds': time.time()-started,
                            'new_paragraph_profiles': len(state.analysis.calls),
                            'paragraph_cache_hits': state.analysis.hits, 'gpu_used': False})
                    env = V2RevisionEnv(config, analysis=state.analysis, tokenizer=state.tokenizer)
                    backend = BulkTeacher(api, attempt, state.tokenizer)
                    backend.condition = f'insertion_boost_attempt_{attempt}'
                    result = run_episode(env, episode, backend,
                        event_path=root / f'attempt_{attempt}/events' / (safe_id(episode_id)+'.jsonl'))
                    result['v2'] = {'attempt': attempt, 'design_sha256': design_sha,
                        'operator': 'G_DEL_LINK', 'sampling_seed': None,
                        'ordering_seed': config[PHASE]['ordering_seeds'][attempt-1],
                        'experiment': 'insertion_boost'}
                    atomic_new(path, result)
                    done += 1
                if result.get('stage1_layout') is not None:
                    text = layout_text(result['stage1_layout'])
                    if cached_gpu_score(config, row['question'], text) is None:
                        queue_cpu(config, row['question'], text, requester=f'insertion:{attempt}:{episode_id}:stage1')
                recovery_path = root / f'attempt_{attempt}/recovery' / path.name
                if not recovery_path.exists():
                    try:
                        stages = recover_global(config, row, result, resources, api)
                        atomic_new(recovery_path, {'episode_sha256': file_sha(path), 'stages': stages})
                        recovery_done += 1
                    except Exception as exc:
                        write_json(root / f'attempt_{attempt}/recovery_errors' / path.name,
                            {'type': type(exc).__name__, 'message': str(exc)})
                        errors.append({'stage': 'recovery', 'attempt': attempt,
                            'episode_id': episode_id, 'type': type(exc).__name__, 'message': str(exc)})
                        if isinstance(exc, CallBudgetExceeded):
                            raise
                write_json(root / 'teacher_progress.json', {'finished_this_run': done,
                    'recovery_this_run': recovery_done, 'last': [attempt, episode_id],
                    'errors': errors, 'budget': api.accounting()})
                print(json.dumps({'attempt': attempt, 'episode_id': episode_id,
                    'completed': result['completed'], 'termination': result['termination'],
                    'confirmed_usd': api.accounting()['confirmed_usd']}), flush=True)
                if result['runtime_error'] and result['runtime_error']['type'] == 'CallBudgetExceeded':
                    raise CallBudgetExceeded(result['runtime_error']['message'])
        except Exception as exc:
            errors.append({'stage': 'collection', 'type': type(exc).__name__, 'message': str(exc)})
        finally:
            api.close()
            status = {'planned_attempts': len(tasks), 'finished_this_run': done,
                'saved_attempts': sum(attempt_path(root, a, eid).exists() for a, eid in tasks),
                'recovery_this_run': recovery_done, 'errors': errors, 'budget': api.accounting(),
                'gpu_used': False, 'training': False}
            write_json(root / 'teacher_status.json', status)
        return status
