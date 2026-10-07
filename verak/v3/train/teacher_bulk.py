"""Authorized two-attempt Luna teacher collection, without training.

Each attempt has a separate API cache namespace. Existing trajectories, including
failures, are immutable; an uncertain interrupted request is never sent again.
"""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace
import uuid

from feak_tc.runtime.openai import CallBudgetExceeded, load_api_environment
from ..agent.runner import fit_history, run_episode, system_prompt
from ..common import file_sha, load_config, read_json, sha_text, write_json
from ..env import RevisionEnv
from ..eval.api import Phase6API, Phase6Teacher
from ..eval.resources import Resources
from ..phase2 import read_jsonl
from ..reconstruction_api import usage_cost
from ..view_data import load_episode_examples
from .pilot import safe_id, stratified_order

PHASE = 'teacher_bulk_two_stage'
ROLES = ('global', 'korean')


def config_for():
    config = load_config()
    pinned = read_json(config['paths']['phase4_output'] / 'models.json')['luna_model']
    config[PHASE] = {'model': pinned, 'max_cost_usd': 25.0, 'max_concurrent_requests': 16,
                     'phase_api_ceiling': 80000, 'seeds': [71, 72], 'attempts': 2}
    config['paths'][PHASE + '_output'] = config['paths']['repo'] / 'verak/v3/outputs' / PHASE
    return config


def atomic_new(path, value):
    """Commit a complete JSON file without ever replacing a saved attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def collection_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'collection.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another bulk collection process owns this output') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(config):
    root = config['paths'][PHASE + '_output']
    corpus_path = config['paths']['active_corrupt'] / 'agent_train.jsonl'
    rows = read_jsonl(corpus_path)
    corpus = {r['episode_id']: r for r in rows}
    if len(rows) != 1430 or len(corpus) != 1430 or any(r['split'] != 'agent_train' for r in rows):
        raise ValueError('Use all 1,430 unique active agent_train essays only')
    if any(x['op'] in {'G_DELETE_SUPPORT', 'L_CONJ_DROP'} for r in rows for x in r['records']):
        raise ValueError('Removed corruption operator in active corpus')
    if config['env']['mode'] != 'two_stage' or config['env']['enable_check']:
        raise ValueError('Bulk requires the accepted two_stage, no-CHECK environment')
    if (config['policy']['context_limit'], config['policy']['generation_reserve']) != (8192, 1024):
        raise ValueError('Keep the accepted 8192/1024 context')
    previous_root = config['paths']['phase7_teacher_output']
    previous = read_json(previous_root / 'design.json')
    pinned = read_json(config['paths']['phase4_output'] / 'models.json')['luna_model']
    if config[PHASE]['model'] != pinned or previous['model'] != pinned:
        raise ValueError('Use the exact Phase-4-pinned Luna ID')
    hashes = {p: file_sha(config['paths']['repo'] / 'verak/v3' / p) for p in previous['code_sha256']}
    prompts = {r: sha_text(system_prompt(r)) for r in ROLES}
    allowed_runner_transition = (
        '290f57f1a571277dc998fb65524c7e88a51afc093c90a396703fb0a9c2cb226f',
        'ee1715e3b6c62f4a2c61d94c330ad3ba4e60b2f19d36934888e43dd1f5f5bcf5')
    differing = {p for p, value in hashes.items() if value != previous['code_sha256'][p]}
    compatible_runner = differing == {'agent/runner.py'} and (
        previous['code_sha256']['agent/runner.py'], hashes['agent/runner.py']) == allowed_runner_transition
    if (differing and not compatible_runner) or prompts != previous['prompt_sha256']:
        raise ValueError('Two-stage environment, reward, context, or prompts changed')
    pilot_ids = previous['pilot_ids']
    if len(pilot_ids) != 92 or not set(pilot_ids) <= corpus.keys():
        raise ValueError('Exactly the existing 92 Luna-low trajectories must be reused')
    reused = {}
    for episode_id in pilot_ids:
        path = previous_root / 'luna_low/episodes' / (safe_id(episode_id) + '.json')
        row = read_json(path)
        if row['corpus_episode_id'] != episode_id or row['model'] != pinned or row['mode'] != 'two_stage':
            raise ValueError('Mismatched saved Luna-low trajectory')
        if row['initial_layout'] != corpus[episode_id]['corrupted_layout']:
            raise ValueError('Saved Luna-low initial layout differs from the active corpus')
        if any(c['reasoning_effort'] != 'low' or c['max_output_tokens'] != 1024 for c in row['calls']):
            raise ValueError('Saved trajectory violates the teacher contract')
        reused[episode_id] = {'path': str(path), 'sha256': file_sha(path), 'completed': row['completed']}
    orders = {str(attempt): stratified_order(rows, seed) for attempt, seed in enumerate((71, 72), 1)}
    design = {'phase': PHASE, 'corpus_path': str(corpus_path), 'corpus_sha256': file_sha(corpus_path),
        'essays': 1430, 'attempts_per_essay': 2, 'requested_slots': 2860, 'new_slots': 2768,
        'model': pinned, 'dated_snapshot_available_in_phase4': False, 'reasoning_effort': 'low',
        'mode': 'two_stage', 'enable_check': False, 'context_limit': 8192, 'output_limit': 1024,
        'budget_usd': 25, 'max_concurrent_requests': config[PHASE]['max_concurrent_requests'], 'phase_api_ceiling': 80000,
        'ordering_seeds': {'1': 71, '2': 72}, 'provider_sampling_seed': None,
        'sampling_note': 'Responses exposes no sampling-seed parameter. Attempt 2 is an independent '
            'request namespace with identical prompts/settings; 71/72 seed only collection ordering.',
        'ordering': 'shuffle within level and level round-robin; interleave attempt 1 and attempt 2 orders',
        'orders': orders, 'levels': dict(Counter(r['level'] for r in rows)), 'reuse_attempt_1': reused,
        'code_sha256': hashes, 'baseline_code_sha256': previous['code_sha256'], 'prompt_sha256': prompts,
        'runner_compatibility': 'Optional observation prompt_factory/generalized handoff parser; '
            'default two-stage context replay required on all92 saved trajectories' if compatible_runner else 'exact hash match',
        'contract_sha256': sha_text(json.dumps({k: config[k] for k in ('env', 'reward', 'policy', PHASE)}, sort_keys=True)),
        'failure_policy': 'Saved completed and failed attempts are immutable; never regenerate them. '
            'On interrupted collection replay completed requests, retain unknown reservations, and '
            'stop that attempt at a previously failed or uncertain request.'}
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'design.json'
    if target.exists():
        if read_json(target) != design:
            raise ValueError('Frozen bulk collection manifest changed')
    else:
        atomic_new(target, design)
    return design, corpus


def attempt_path(root, attempt, episode_id):
    return root / f'attempt_{attempt}/episodes' / (safe_id(episode_id) + '.json')


def tasks(design):
    for first, second in zip(design['orders']['1'], design['orders']['2']):
        if first not in design['reuse_attempt_1']:
            yield 1, first
        yield 2, second


class BulkAPI(Phase6API):
    """Use aggregate reservations and refuse to regenerate interrupted calls."""

    def _durable_outcome(self, call_id, stage, item_id, fingerprint, reserved):
        """Validate the record written before the parent's ledger update."""
        path = self.output / 'requests' / f'{call_id:06}.json'
        try:
            record = read_json(path)
            if not isinstance(record, dict) or type(record.get('phase_call')) is not int or any(record.get(key) != value for key, value in (
                ('phase_call', call_id), ('fingerprint', fingerprint), ('stage', stage),
                ('item_id', item_id), ('model', self.model))):
                raise ValueError('durable request identity mismatch')
            keys = ('stage', 'item_id', 'model', 'reasoning_effort', 'max_output_tokens', 'messages', 'schema')
            contract = {key: record[key] for key in keys}
            if sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True)) != fingerprint:
                raise ValueError('durable request payload fingerprint mismatch')
            status = record['status']
            if status not in {'completed', 'incomplete', 'error', 'failed', 'cancelled'}:
                raise ValueError('durable response has no terminal status')
            if status != 'error' and (not isinstance(record.get('raw'), str) or
                    not isinstance(record.get('response_id'), str) or not record['response_id']):
                raise ValueError('durable response body is incomplete')
            if status == 'error' and not isinstance(record.get('error_type'), str):
                raise ValueError('durable exception record is incomplete')
            confirmed = 0.0
            usage = record.get('usage')
            if usage is not None:
                if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                        for k in ('input_tokens', 'output_tokens')):
                    raise ValueError('durable response usage is invalid')
                for field, keys in (('input_tokens_details', ('cached_tokens', 'cache_write_tokens')),
                                    ('output_tokens_details', ('reasoning_tokens',))):
                    details = usage.get(field)
                    if details is not None and (not isinstance(details, dict) or any(
                            type(details.get(k, 0)) is not int or details.get(k, 0) < 0 for k in keys)):
                        raise ValueError('durable response usage details are invalid')
                expected = usage_cost(self.model, usage)
                if any(value < 0 or not math.isfinite(value) for value in expected.values()):
                    raise ValueError('durable response cost is invalid')
                if expected['reasoning'] > expected['output']:
                    raise ValueError('durable reasoning usage exceeds output usage')
                cost = record['cost']
                if not isinstance(cost, dict) or any(type(cost.get(k)) not in (int, float) or
                        not math.isfinite(cost[k]) or abs(cost[k] - value) > 1e-12 for k, value in expected.items()):
                    raise ValueError('durable response cost/usage mismatch')
                confirmed = expected['confirmed_usd']
            elif record.get('cost'):
                raise ValueError('durable cost lacks supporting usage')
            timeout = 'Timeout' in record.get('error_type', '') or 'timeout_reservation_usd' in record
            # Successful/incomplete responses without usage can be replayed/preserved,
            # but their billing remains unknown. Never release that reservation.
            retained = reserved if timeout or (usage is None and status != 'error') else 0.0
            return {'status': status, 'confirmed': confirmed, 'reserved': retained,
                'path': str(path), 'reconciliation': 'recovered_durable_record'}
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            return {'status': 'interrupted_unconfirmed', 'reserved': reserved,
                'path': None, 'reconciliation': 'reservation_retained', 'reason': type(exc).__name__ + ': ' + str(exc)}

    def settle_interrupted(self):
        reconciled = []
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT id,stage,item_id,fingerprint,reserved,confirmed FROM calls WHERE status='pending'").fetchall()
            for call_id, stage, item_id, fingerprint, reserved, confirmed in rows:
                outcome = self._durable_outcome(call_id, stage, item_id, fingerprint, reserved)
                outcome.setdefault('confirmed', confirmed)
                db.execute('UPDATE calls SET status=?,reserved=?,confirmed=?,path=?,finished=? WHERE id=?',
                    (outcome['status'], outcome['reserved'], outcome['confirmed'], outcome['path'], time.time(), call_id))
                reconciled.append({'phase_call': call_id, 'stage': stage, 'item_id': item_id,
                    'previous_reserved': reserved, **outcome})
        if reconciled:
            atomic_new(self.output / ('interrupted_' + uuid.uuid4().hex + '.json'),
                {'requests': reconciled, 'at': time.time()})
        return reconciled

    def reserve(self, stage, item_id, fingerprint, bound):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            n, pending, committed = db.execute("SELECT SUM(status!='blocked_before_send'), "
                "SUM(status='pending'), SUM(reserved+confirmed) FROM calls").fetchone()
            if (n or 0) >= self.limit:
                raise CallBudgetExceeded(f'{self.phase} --max-api-calls exhausted')
            if (pending or 0) >= self.settings['max_concurrent_requests']:
                return None
            if (committed or 0) + bound > self.settings['max_cost_usd']:
                raise CallBudgetExceeded(f"{self.phase} ${self.settings['max_cost_usd']} cap: insufficient unreserved budget")
            return db.execute('INSERT INTO calls(stage,item_id,fingerprint,status,reserved,created) VALUES(?,?,?,?,?,?)',
                (stage, item_id, fingerprint, 'pending', bound, time.time())).lastrowid

    def request(self, messages, *, stage, item_id, effort='low', max_output=1024, schema=None):
        contract = {'stage': stage, 'item_id': item_id, 'model': self.model,
            'reasoning_effort': effort, 'max_output_tokens': max_output, 'messages': messages, 'schema': schema}
        fingerprint = sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True))
        with self.db() as db:
            previous = db.execute('SELECT id,status FROM calls WHERE fingerprint=? ORDER BY id', (fingerprint,)).fetchall()
        if previous and not any(status == 'completed' for _, status in previous):
            raise RuntimeError(f'Preserved prior API outcome for this attempt: {previous}; no regeneration')
        return super().request(messages, stage=stage, item_id=item_id, effort=effort, max_output=max_output, schema=schema)


class BulkTeacher(Phase6Teacher):
    def __init__(self, api, attempt, tokenizer):
        super().__init__(api, f'bulk_attempt_{attempt}', max_output=1024, effort='low')
        self.tokenizer = tokenizer

    def generate(self, messages, *, episode_id, role, turn):
        prefix = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if prefix > 7168:
            raise ValueError('Policy input exceeds 8192 minus 1024 reserve')
        response = super().generate(messages, episode_id=episode_id, role=role, turn=turn)
        total = len(self.tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': response['raw']}], tokenize=True))
        if total > 8192:
            raise ValueError('Generated turn exceeds the 8192-token policy context; raw API response preserved')
        return {**response, 'policy_input_tokens': prefix, 'policy_total_tokens': total}


def load_environment(config):
    from dotenv import dotenv_values
    load_api_environment()
    values = dotenv_values(config['paths']['scorer_package'] / '.env')
    for name in ('OPENAI_API_KEY', 'BAREUN_API_KEY'):
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]
    if not os.environ.get('BAREUN_API_KEY'):
        raise RuntimeError('BAREUN_API_KEY unavailable; refuse paid dispatch')


def preflight(config, resources, corpus):
    load_environment(config)
    socket.getaddrinfo('api.openai.com', 443)
    state = resources.worker()
    replay = replay_baseline_contexts(config, state.tokenizer)
    profile = state.analysis.profile('글의 내용을 확인한다.')
    row = next(iter(corpus.values()))
    source, corrupted = resources.source(row['source_id']), resources.corrupted(row)
    score = resources.score(row['question'], corrupted.text)
    similarity = resources.similarity(source.units[0].text, source.units[0].text)
    result = {'passed': True, 'bareun_live_profile': bool(profile), 'scorer_mean': score['mean'],
        'scorer_input_tokens': score['input_tokens'], 'scorer_cache_hit': score['cache_hit'],
        'similarity_self': float(similarity), 'api_dns': True, 'paid_api_calls': 0,
        'baseline_context_replay': replay, 'at': time.time()}
    write_json(config['paths'][PHASE + '_output'] / 'preflight.json', result)
    return result


def replay_baseline_contexts(config, tokenizer):
    """Show the later optional observation hook preserves all accepted contexts."""
    root = config['paths'][PHASE + '_output']
    design = read_json(root / 'design.json')
    target = root / 'baseline_context_replay.json'
    design_sha = file_sha(root / 'design.json')
    if target.exists():
        saved = read_json(target)
        if saved.get('design_sha256') == design_sha and saved.get('passed'):
            return saved
    turns = 0
    for episode_id, source in design['reuse_attempt_1'].items():
        row = read_json(source['path'])
        for role, history in row['messages_by_role'].items():
            indices = [i for i, m in enumerate(history) if m['role'] == 'assistant']
            calls = [c for c in row['calls'] if c['role'] == role]
            if len(indices) != len(calls):
                raise ValueError('Baseline raw history/turn mismatch')
            for index, call in zip(indices, calls):
                t = int(call['turn'].split(':')[0])
                sent, compacted = fit_history(history[:index],
                    SimpleNamespace(context_limit=8192, generation_reserve=1024), tokenizer,
                    row['actions_by_role'][role][:t - 1])
                if sent != call['messages'] or compacted != call['history_compacted']:
                    raise ValueError(f'Default context changed for {episode_id}:{role}:{call["turn"]}')
                turns += 1
    result = {'passed': True, 'essays': 92, 'turns': turns, 'design_sha256': design_sha,
        'code_sha256': design['code_sha256'], 'baseline_code_sha256': design['baseline_code_sha256']}
    write_json(target, result)
    return result


def partial_global(config, resources, env, row, result):
    if result['completed'] or not result.get('stage1_layout') or 'global' not in result['termination']:
        return
    from ..reward.total import rewards
    q = resources.score(row['question'], env.stage1.text)
    result['global_only_reward'] = rewards(env.source, env.corrupted, env.stage1, row['records'],
        genre=row['genre'], q_corrupted=row['q_corrupted'], q_final=q['mean'], q_stage1=q['mean'],
        config=config['reward'], mode='two_stage', stage1=env.stage1,
        stage1_actions=result['actions_by_role']['global'], stage2_actions=[],
        similarity=resources.similarity, tau=config['similarity']['tau'],
        preexisting_spell_spans=row.get('preexisting_spell_spans', ()))['global']
    result['partial_global_score'] = q


def execute(config, api, *, limit=None):
    design, corpus = prepare(config)
    root = config['paths'][PHASE + '_output']
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    if any(r['source_id'] not in examples for r in corpus.values()):
        raise ValueError('Wrong split or view-ineligible active corruption source')
    if any(x['params']['donor']['source_id'] not in examples for r in corpus.values()
           for x in r['records'] if x['op'] == 'G_OFFTOPIC'):
        raise ValueError('Wrong-split off-topic donor')
    design_sha = file_sha(root / 'design.json')
    resources = Resources(config, examples, output_key=PHASE + '_output')
    finished, errors = [], []
    stop_reason = None
    def one(item):
        attempt, episode_id = item
        path = attempt_path(root, attempt, episode_id)
        if path.exists():
            result = read_json(path)
            if result['bulk']['design_sha256'] != design_sha or result['bulk']['attempt'] != attempt:
                raise ValueError('Saved attempt does not match the frozen design')
            return
        row = corpus[episode_id]
        state = resources.worker()
        episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre', 'level',
            'records', 'corrupted_score', 'preexisting_spell_spans')}
        episode.update(source=resources.source(row['source_id']), document=resources.corrupted(row))
        env = RevisionEnv(config, mode='two_stage', analysis=state.analysis, scorer=resources,
            similarity=resources.similarity, tokenizer=state.tokenizer)
        result = run_episode(env, episode, BulkTeacher(api, attempt, state.tokenizer),
            event_path=root / f'attempt_{attempt}/events' / (safe_id(episode_id) + '.jsonl'))
        try:
            partial_global(config, resources, env, row, result)
        except Exception as exc:
            result['partial_global_evaluation_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        with api.db() as db:
            paths = db.execute('SELECT path FROM calls WHERE stage=? AND item_id LIKE ?',
                (f'bulk_attempt_{attempt}', result['episode_id'] + ':%')).fetchall()
        requests = [read_json(p[0]) for p in paths if p[0]]
        result['confirmed_episode_cost'] = sum(r.get('cost', {}).get('confirmed_usd', 0) for r in requests)
        result['api_request_count'] = len(requests)
        result['api_noncompleted'] = [{k: r.get(k) for k in ('phase_call', 'status', 'error_type', 'http_status')}
            for r in requests if r['status'] != 'completed']
        result['bulk'] = {'attempt': attempt, 'ordering_seed': 70 + attempt, 'provider_sampling_seed': None,
            'design_sha256': design_sha, 'reused': False, 'stage': f'bulk_attempt_{attempt}'}
        atomic_new(path, result)
        print(json.dumps({'attempt': attempt, 'id': episode_id, 'completed': result['completed'],
            'termination': result['termination'], 'steps': result['steps'], 'error': result['runtime_error'],
            'cost': result['confirmed_episode_cost'], 'budget': api.accounting()}, ensure_ascii=False), flush=True)
        if result['runtime_error'] and result['runtime_error']['type'] == 'CallBudgetExceeded':
            raise CallBudgetExceeded(result['runtime_error']['message'])
    try:
        preflight(config, resources, corpus)
        api.client()
        interrupted = api.settle_interrupted()
        queue = [(a, i) for a, i in tasks(design) if not attempt_path(root, a, i).exists()]
        if limit is not None:
            queue = queue[:limit]
        workers = config[PHASE]['max_concurrent_requests']
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator, pending = iter(queue), {}
            def fill():
                while len(pending) < workers and not errors:
                    item = next(iterator, None)
                    if item is None:
                        break
                    pending[pool.submit(one, item)] = item
            fill()
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    item = pending.pop(future)
                    try:
                        future.result()
                        finished.append(item)
                    except Exception as exc:
                        errors.append({'attempt': item[0], 'id': item[1], 'type': type(exc).__name__, 'message': str(exc)})
                write_json(root / 'progress.json', {'finished_this_invocation': len(finished),
                    'new_slots_this_invocation': len(queue), 'in_flight': list(pending.values()),
                    'errors': errors, 'budget': api.accounting(), 'interrupted_reservations': interrupted})
                fill()
        stop_reason = 'budget_cap' if any(e['type'] == 'CallBudgetExceeded' for e in errors) else (
            'dispatch_error' if errors else 'requested_limit' if limit is not None else 'all_slots_attempted')
    finally:
        resources.close()
        write_json(root / 'run_status.json', {'requested_slots': 2860, 'reused_slots': 92,
            'finished_this_invocation': len(finished), 'errors': errors, 'stop_reason': stop_reason,
            'budget': api.accounting(), 'at': time.time()})
    return errors
