"""Sol-low GLOBAL rescue with unchanged Luna KOREAN hand-off and policy context."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from threading import local
import time
from types import SimpleNamespace

from ..agent.runner import run_episode
from ..common import file_sha, read_json, write_json
from ..corrupt.document import Document, Paragraph, Unit
from ..env.protocol import parse_action
from ..reward.recovery import main_recovery
from ..train.teacher_bulk import BulkTeacher, atomic_new, collection_lock
from ..v2_ops.local import load_environment
from .config import PHASE
from .expansion import batch_configs, root_for
from .paid import BoostAPI
from .prepare import corpus, safe_id
from .teacher import DeferredRewardEnv, attempt_path


def structural_main(candidate, raw):
    """The frozen structural recovery formula needs positions only, no new calls."""
    if raw.get('stage1_layout') is None:
        return None
    def restore(layout):
        return Document([Paragraph(p['pid'], [Unit(u['sid'], u['text'], [], u['leading'])
            for u in p['units']]) for p in layout['paragraphs']], layout['gaps'], layout['tail'])
    source, corrupted, final = (restore(layout) for layout in
        (candidate['source_layout'], candidate['corrupted_layout'], raw['stage1_layout']))
    annotations = {u.sid: SimpleNamespace(paragraph=p.pid) for p in final.paragraphs for u in p.units}
    return main_recovery(source, final, candidate['records'][0], corrupted=corrupted, annotations=annotations)


def rescue_plan(config):
    root = root_for(config)
    path = root / 'v4/rescue_plan.json'
    if path.exists():
        return read_json(path)
    comparison_path = root / 'v4/sol_luna_same92.json'
    triggers = read_json(comparison_path)['triggers']
    tasks, excluded = [], {}
    for cfg in batch_configs(config):
        batch = cfg['paths'][PHASE+'_output']
        for eid, candidate in corpus(cfg).items():
            if not triggers[candidate['operator']]['sol_rescue_authorized']:
                continue
            paths = [attempt_path(batch, a, eid) for a in (1,2)]
            if not all(p.exists() for p in paths):
                excluded[eid] = 'both_Luna_attempts_not_saved'
                continue
            rows = [read_json(p) for p in paths]
            if not all(r.get('generation_completed', r.get('completed')) for r in rows):
                excluded[eid] = 'both_Luna_attempts_not_completed'
                continue
            values = [structural_main(candidate, r) for r in rows]
            if not all(v is not None and v < 1 for v in values):
                excluded[eid] = 'at_least_one_Luna_full_recovery'
                continue
            tasks.append({'episode_id': eid, 'batch': str(batch), 'source_id': candidate['source_id'],
                'operator': candidate['operator'], 'Luna_main': values,
                'Luna_paths': [str(p) for p in paths], 'Luna_sha256': [file_sha(p) for p in paths],
                'new_diverse_candidate': 'source_provenance' in candidate})
    # One rescue per distinct source before second positions; newest diverse data first.
    tasks.sort(key=lambda x: (not x['new_diverse_candidate'], x['episode_id']))
    counts = {}
    for task in tasks:
        task['source_round'] = counts.get(task['source_id'], 0)
        counts[task['source_id']] = task['source_round']+1
    tasks.sort(key=lambda x: (x['source_round'], not x['new_diverse_candidate'], x['episode_id']))
    value = {'model_global': config[PHASE]['sol_model'], 'model_korean': config[PHASE]['model'],
        'effort': 'low', 'teacher_output_global': 4096, 'teacher_output_korean': 1024,
        'policy_context_limit': 8192, 'policy_output_limit': 1024, 'tasks': tasks, 'excluded': excluded,
        'comparison_sha256': file_sha(comparison_path), 'sampling_seed': None,
        'selection': 'GLOBAL R>=.80, valid terminal STOP, rejected actions<=1; GPU only; max4/source/operator',
        'gpu_used': False, 'training': False}
    atomic_new(path, value)
    return value


class RescueTeacher:
    name = 'teacher'
    context_limit = 8192
    generation_reserve = 1024

    def __init__(self, sol, luna, tokenizer):
        self.global_backend = BulkTeacher(sol, 3, tokenizer)
        self.global_backend.condition = 'global_boost_sol_rescue_global'
        self.global_backend.max_output = 4096
        self.korean_backend = BulkTeacher(luna, 3, tokenizer)
        self.korean_backend.condition = 'global_boost_sol_rescue_korean'
        self.model = sol.model
        self.tokenizer = tokenizer

    def generate(self, messages, *, episode_id, role, turn):
        backend = self.global_backend if role == 'global' else self.korean_backend
        response = backend.generate(messages, episode_id=episode_id, role=role, turn=turn)
        if role == 'global':
            # API reasoning is retained in its raw response archive; the policy
            # sees and learns the one action JSON only, with its 1024-token limit.
            try:
                action = parse_action(response['raw'])
            except ValueError:
                return response
            canonical = json.dumps(action, ensure_ascii=False)
            target = self.tokenizer.apply_chat_template(messages+[{'role':'assistant','content':canonical}], tokenize=True)
            prefix = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if len(target)-len(prefix) > 1024 or len(target) > 8192:
                raise ValueError('Rescue action JSON exceeds unchanged policy output/context limit')
            response = {**response, 'teacher_raw': response['raw'], 'raw': canonical,
                'policy_target_tokens': len(target)-len(prefix), 'teacher_only_output_limit': 4096}
        return response


def run_rescue(config):
    from .resources import CPUResources
    from .prefetch import enqueue
    root = root_for(config)
    with collection_lock(root / 'v4/rescue'):
        load_environment(config)
        plan = rescue_plan(config)
        configurations = {str(c['paths'][PHASE+'_output']): c for c in batch_configs(config)}
        rows = {str(c['paths'][PHASE+'_output']): corpus(c) for c in configurations.values()}
        tasks = [t for t in plan['tasks'] if not attempt_path(Path(t['batch']), 3, t['episode_id']).exists()]
        sol, luna = BoostAPI(config,100000,kind='sol'), BoostAPI(config,100000,kind='luna')
        sol.settle_interrupted()
        state = local()
        def one(task):
            if not hasattr(state, 'resources'):
                state.resources = CPUResources(config)
            resources = state.resources
            for path, digest in zip(task['Luna_paths'], task['Luna_sha256']):
                if file_sha(path) != digest:
                    raise ValueError('Frozen Luna rescue trigger changed')
            candidate = rows[task['batch']][task['episode_id']]
            episode = {k:candidate[k] for k in ('episode_id','source_id','question','genre','level','records','preexisting_spell_spans')}
            episode.update(source=resources.source(candidate), document=resources.restore(candidate['corrupted_layout']))
            env = DeferredRewardEnv(config,mode='two_stage',analysis=resources.analysis,tokenizer=resources.tokenizer)
            backend = RescueTeacher(sol,luna,resources.tokenizer)
            result = run_episode(env,episode,backend,event_path=Path(task['batch'])/'attempt_3/events'/(safe_id(task['episode_id'])+'.jsonl'))
            result.update(generation_completed=result['completed'], reward_status='deferred_gpu_reference',
                data_boost={'attempt':3,'teacher':'Sol_GLOBAL_Luna_KOREAN','selected_role':'global',
                    'rescue_plan_sha256':file_sha(root/'v4/rescue_plan.json'),'gpu_used':False})
            atomic_new(attempt_path(Path(task['batch']),3,task['episode_id']),result)
            return task,result
        saved, stop = [], 'all_rescue_slots_attempted'
        try:
            # Dispatch at most two independent episodes. A cap stops new work;
            # already-started results remain durable and fully accounted.
            from concurrent.futures import FIRST_COMPLETED, wait
            with ThreadPoolExecutor(max_workers=2) as pool:
                iterator,active=iter(tasks),{}
                while True:
                    while len(active)<2 and stop=='all_rescue_slots_attempted':
                        task=next(iterator,None)
                        if task is None:break
                        active[pool.submit(one,task)]=task
                    if not active:break
                    done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                    for future in done:
                        active.pop(future)
                        task,result=future.result()
                        saved.append(task['episode_id'])
                        enqueue(configurations[task['batch']])
                        if (result.get('runtime_error') or {}).get('type')=='CallBudgetExceeded':
                            stop='budget_cap'
                    write_json(root/'v4/rescue_progress.json',{'saved_this_run':len(saved),'in_flight':list(active.values()),
                        'stop_reason':stop,'budget':sol.accounting(),'at':time.time()})
        finally:
            sol.close();luna.close()
        value={'saved_this_run':len(saved),'requested':len(plan['tasks']),'stop_reason':stop,'budget':sol.accounting(),'at':time.time()}
        write_json(root/'v4/rescue_status.json',value)
        return value
