"""Two independent Luna attempts, CPU-only generation and cached hidden recovery."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import fcntl
import json
import random
from threading import local

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, write_json
from ..corrupt.document import BareunBank, Document
from ..train.teacher_bulk import BulkTeacher, atomic_new, collection_lock
from .candidates import PriorityParagraphs
from .config import PHASE, OPERATORS, require_v2
from .data import source_pools
from .environment import V2RevisionEnv
from .judges import recovery_contract, contract_key, validate
from .local import load_environment
from .paid import V2API
from .prompts import system_prompt
from .qc import candidate_paths, summary
from .reward import recovery
from .runner import run_episode


def safe_id(value):
    return value.replace(':', '_')


def attempt_path(root, attempt, episode_id):
    return root / f'attempt_{attempt}/episodes' / (safe_id(episode_id) + '.json')


def prepare(config):
    require_v2(config)
    root = config['paths'][PHASE + '_output']
    qc = summary(config)
    corpus = {}
    for path in candidate_paths(config):
        row = read_json(path)
        verdict = root / 'qc' / (safe_id(row['episode_id']) + '.json')
        if row['split'] != 'agent_train' or qc['operators'][row['operator']]['decision'] != 'retain':
            continue
        if not verdict.exists() or not read_json(verdict)['passed']:
            continue
        if read_json(verdict)['candidate_sha256'] != file_sha(path):
            raise ValueError('Teacher candidate changed after QC')
        corpus[row['episode_id']] = row
    orders = {}
    for attempt, seed in enumerate(config[PHASE]['ordering_seeds'], 1):
        ids = sorted(corpus)
        random.Random(seed).shuffle(ids)
        orders[str(attempt)] = ids
    directory = __import__('pathlib').Path(__file__).parent
    runtime = {str(directory / name): file_sha(directory / name) for name in
               ('actions.py', 'protocol.py', 'prompts.py', 'environment.py', 'runner.py', 'reward.py', 'operators.py', 'judges.py')}
    design = {'version': 'v2', 'model': config[PHASE]['model'], 'reasoning': 'low',
        'context': 8192, 'output_limit': 1024, 'attempts': 2, 'sampling_seed': None,
        'ordering_seeds': config[PHASE]['ordering_seeds'], 'orders': orders,
        'source_plan_sha256': file_sha(root / 'source_plan.json'), 'runtime_sha256': runtime,
        'prompt': {r: system_prompt(r) for r in ('global', 'korean')},
        'corpus': {i: {'source_id': row['source_id'], 'operator': row['operator'],
                       'corrupted_hash': row['corrupted_hash']} for i, row in corpus.items()}}
    path = root / 'teacher_design.json'
    if path.exists() and read_json(path) != design:
        raise ValueError('Frozen v2 teacher design changed; do not silently regenerate')
    write_json(path, design)
    return design, corpus


class CPUResources:
    def __init__(self, config):
        from transformers import AutoTokenizer
        self.tokenizer_class = AutoTokenizer
        self.config = config
        pools, _ = source_pools(config)
        self.sources = {e.id: document for values in pools.values() for e, document, _ in values}
        self.local = local()

    def worker(self):
        if not hasattr(self.local, 'analysis'):
            root = self.config['paths'][PHASE + '_output']
            paragraphs, bank = PriorityParagraphs, BareunBank
            if self.config.get('v2_retry', {}).get('enabled'):
                from .retry_resources import RetryParagraphs, RetryBank
                paragraphs, bank = RetryParagraphs, RetryBank
            self.local.analysis = paragraphs(self.config, cache_dir=root / 'bareun_paragraphs')
            self.local.bank = bank(self.config, cache_dir=root / 'bareun_units')
            self.local.tokenizer = self.tokenizer_class.from_pretrained(str(self.config['paths']['policy_base']), local_files_only=True)
        return self.local

    def source(self, row):
        state = self.worker()
        source = self.sources[row['source_id']].clone()
        state.bank.seed(source)
        return source

    def restore(self, layout):
        state = self.worker()
        # Stable boundaries may include an original multi-unit source sentence.
        # Original units were seeded from the source; new actions enforce unit count.
        doc = Document.restore(layout, state.bank)
        state.analysis.refresh(doc, {p.pid for p in doc.paragraphs})
        return doc


class RecoveryJudge:
    def __init__(self, config, row, api=None):
        self.root = config['paths'][PHASE + '_output'] / 'recovery_judgments'
        self.row, self.api = row, api

    def __call__(self, record, unit):
        messages, schema = recovery_contract(self.row, record, unit.text)
        key = contract_key(messages, schema)
        path = self.root / (key + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._judge(path, key, messages, schema)

    def _judge(self, path, key, messages, schema):
        if path.exists():
            saved = read_json(path)
        elif self.api is None:
            raise ValueError('Missing cached G_DEL_LINK recovery verdict; never assume zero')
        else:
            response = self.api.request(messages, schema=schema, stage='v2_link_recovery', item_id=key,
                                        effort='low', max_output=1024)
            verdict = json.loads(response['raw'])
            validate(verdict, schema)
            saved = {'contract_sha256': key, 'phase_call': response['phase_call'], 'judgment': verdict}
            atomic_new(path, saved)
        validate(saved['judgment'], schema)
        return {'yes': 1., 'partial': .5, 'no': 0.}[saved['judgment']['verdict']]


def recover_saved(config, candidate, result, resources, api=None):
    source = resources.source(candidate)
    corrupted = resources.restore(candidate['corrupted_layout'])
    stages = {}
    actions = result['actions_by_role']
    judge = RecoveryJudge(config, candidate, api)
    for name, layout, steps in (
        ('global', result.get('stage1_layout'), actions.get('global', [])),
        ('combined', result.get('final_layout') if result['completed'] else None,
         actions.get('global', []) + actions.get('korean', []))):
        if layout is not None:
            final = resources.restore(layout)
            rows, matched, origins = recovery(source, corrupted, final, candidate['records'], steps, judge,
                coupled_weight=config['reward'].get('dependents_weight', .3))
            stages[name] = {'per_record': rows, 'matched_insertions': sorted(matched), 'insert_origins': origins}
    return stages


def run(config, *, max_api_calls, paid_approved=False, limit=None):
    if not paid_approved:
        raise PermissionError('Explicit Proceed is required before v2 teacher generation')
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        design, corpus = prepare(config)
        design_sha = file_sha(root / 'teacher_design.json')
        api = V2API(config, max_api_calls, kind='luna', paid_approved=True)
        resources = CPUResources(config)
        n = len(corpus)
        tasks = [(attempt, design['orders'][str(attempt)][i]) for i in range(n) for attempt in (1, 2)]
        tasks = [(a, i) for a, i in tasks if not attempt_path(root, a, i).exists()]
        if limit is not None:
            tasks = tasks[:limit]
        errors, finished = [], []

        def one(task):
            attempt, episode_id = task
            row = corpus[episode_id]
            state = resources.worker()
            episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre', 'level', 'records', 'preexisting_spell_spans')}
            episode.update(source=resources.source(row), document=resources.restore(row['corrupted_layout']))
            env = V2RevisionEnv(config, analysis=state.analysis, tokenizer=state.tokenizer)
            backend = BulkTeacher(api, attempt, state.tokenizer)
            backend.condition = f'v2_teacher_attempt_{attempt}'
            result = run_episode(env, episode, backend,
                event_path=root / f'attempt_{attempt}/events' / (safe_id(episode_id) + '.jsonl'))
            result['v2'] = {'attempt': attempt, 'design_sha256': design_sha, 'operator': row['operator'],
                'sampling_seed': None, 'ordering_seed': config[PHASE]['ordering_seeds'][attempt-1]}
            path = attempt_path(root, attempt, episode_id)
            atomic_new(path, result)  # Save raw generation even if recovery/remaining budget fails.
            try:
                stages = recover_saved(config, row, result, resources, api)
                write_json(root / f'attempt_{attempt}/recovery' / path.name,
                           {'episode_sha256': file_sha(path), 'stages': stages})
            except Exception as exc:
                write_json(root / f'attempt_{attempt}/recovery_errors' / path.name,
                           {'type': type(exc).__name__, 'message': str(exc)})
                if isinstance(exc, CallBudgetExceeded):
                    raise
            print(json.dumps({'attempt': attempt, 'episode': episode_id, 'completed': result['completed'],
                              'confirmed_usd': api.accounting()['confirmed_usd']}), flush=True)
            if result['runtime_error'] and result['runtime_error']['type'] == 'CallBudgetExceeded':
                raise CallBudgetExceeded(result['runtime_error']['message'])

        try:
            api.client()
            api.settle_interrupted()
            iterator = iter(tasks)
            with ThreadPoolExecutor(max_workers=config[PHASE]['max_concurrent_requests']) as pool:
                active = {}
                while True:
                    while len(active) < config[PHASE]['max_concurrent_requests'] and not errors:
                        task = next(iterator, None)
                        if task is None:
                            break
                        active[pool.submit(one, task)] = task
                    if not active:
                        break
                    done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = active.pop(future)
                        try:
                            future.result()
                            finished.append(task)
                        except Exception as exc:
                            errors.append({'task': task, 'type': type(exc).__name__, 'message': str(exc)})
                    write_json(root / 'teacher_progress.json', {'finished_this_run': len(finished),
                        'in_flight': list(active.values()), 'errors': errors, 'budget': api.accounting()})
        finally:
            api.close()
            status = {'requested_attempts': 2*n, 'finished_this_run': len(finished), 'errors': errors,
                'budget': api.accounting(), 'training': False, 'gpu_used': False}
            write_json(root / 'teacher_status.json', status)
        return status
