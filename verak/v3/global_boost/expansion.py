"""Balanced distinct Phase3b practices, frozen batches, and one shared $12 cap.

Source diversity and corruption diversity are different: the same unused source
may produce several practices, each with exactly one record. The original batch
is immutable. Additional batches contain at most 25 examples per operator and
reserve projected funds for both Luna attempts before paying for their QC.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import time

from ..common import file_sha, read_json, sha_text, write_json
from ..corrupt.builder import source_coupled_changes
from ..corrupt.document import BareunBank, source_document
from ..corrupt.instance_policy import candidates
from ..corrupt.operators import Proposal, apply, exact_restoration_satisfies, restore_record
from ..ko import render
from ..train.teacher_bulk import atomic_new, collection_lock
from ..view_data import load_episode_examples
from .config import OPERATORS, PHASE
from .paid import BoostAPI
from .prepare import corpus, safe_id


def root_for(config):
    return config['paths'].get('global_boost_shared_root', config['paths'][PHASE + '_output'])


def audit_variants(config):
    """Read cached source profiles only; exhaust the closed structural proposals."""
    from transformers import AutoTokenizer
    root = root_for(config)
    rows = corpus(batch_config(config, root))
    source_ids = {r['source_id'] for r in rows.values()}
    path = root / 'variant_supply_validated.json'
    if path.exists():
        saved = read_json(path)
        if saved['sources'] == len(source_ids):
            if set(saved['per_source']) != source_ids:
                raise ValueError('Variant audit source identity changed')
            return saved
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    bank = BareunBank(config, cache_dir=root / 'bareun_units')
    overall = {op: Counter() for op in OPERATORS}
    per_source = {}
    started = time.monotonic()
    for source_id in sorted(source_ids):
        source = source_document(config, examples[source_id], bank)
        per_source[source_id] = {}
        for op in OPERATORS:
            proposals = candidates(source, op, donors=(), vague_cache={}, question_hash=examples[source_id].question_hash)
            counts = Counter(raw=len(proposals))
            valid, seen = [], set()
            for proposal in proposals:
                try:
                    changed, record = apply(source, proposal, bank)
                    restored = restore_record(changed, record, bank)
                    if restored.text != source.text or not exact_restoration_satisfies(restored, record):
                        raise ValueError('inverse_failed')
                    view = render(changed.structure(), compact=True)
                    tokens = len(tokenizer.encode(view, add_special_tokens=False))
                    if tokens > 3000:
                        raise ValueError('view_exceeds_3000')
                    digest = sha_text(changed.text)
                    if digest in seen:
                        counts['duplicate'] += 1
                        continue
                    seen.add(digest)
                    valid.append({'sids': proposal.sids, 'params': proposal.params,
                                  'corrupted_hash': digest, 'compact_tokens': tokens})
                    counts['valid_distinct'] += 1
                except ValueError as exc:
                    counts[str(exc)] += 1
            per_source[source_id][op] = {'counts': dict(counts), 'valid_proposals': valid}
            overall[op].update(counts)
        value = {'sources': len(per_source), 'overall': {op: dict(v) for op, v in overall.items()},
            'per_source': per_source, 'seconds': time.monotonic()-started,
            'paid_calls': 0, 'gpu_used': False, 'new_bareun_calls': 0}
        write_json(path, value)
        print(json.dumps({k: v for k, v in value.items() if k != 'per_source'}), flush=True)
    return value


def batch_config(config, root):
    value = deepcopy(config)
    shared = root_for(config)
    value['paths']['global_boost_shared_root'] = shared
    value['paths'][PHASE + '_output'] = Path(root)
    if Path(root) != shared:
        value[PHASE].update(api_item_namespace=f'global_boost:{Path(root).name}:',
            allow_unjudged_qc=True, prefetch_saved=True)
    return value


def batch_configs(config):
    root = root_for(config)
    return [batch_config(config, root)] + [batch_config(config, p.parent)
        for p in sorted((root / 'batches').glob('batch_*/source_plan.json'))]


def generation_counts(config):
    counts = Counter()
    costs = []
    pending = 0
    for cfg in batch_configs(config):
        root = cfg['paths'][PHASE + '_output']
        for row in corpus(cfg).values():
            counts[row['operator']] += 1
            verdict = root / 'qc' / (safe_id(row['episode_id']) + '.json')
            if not verdict.exists() or not read_json(verdict)['passed']:
                continue
            for attempt in (1, 2):
                target = root / f'attempt_{attempt}/episodes' / (safe_id(row['episode_id']) + '.json')
                if target.exists():
                    result = read_json(target)
                    costs.append(result.get('confirmed_episode_cost', 0.))
                else:
                    pending += 1
    return counts, costs, pending


def projected_batch(config, *, max_per_operator=25):
    """Projection is a scheduling guard; each sent call also has a hard reservation.

    Plan for every new QC candidate passing, not just the observed pass rate.
    Conservative historical averages leave Luna headroom. Actual usage is reused
    for the next batch; the hard aggregate ledger remains the final authority.
    """
    counts, episode_costs, pending = generation_counts(config)
    api = BoostAPI(config, 0)
    account = api.accounting()
    with api.db() as db:
        qc_costs = [v[0] for v in db.execute("SELECT confirmed FROM calls WHERE "
            "stage='global_boost_qc' AND status='completed'").fetchall()]
    api.close()
    teacher = max(.007, sum(episode_costs)/len(episode_costs) if episode_costs else 0.) * 1.2
    qc = max(.010, sum(qc_costs)/len(qc_costs) if qc_costs else 0.) * 1.15
    available = max(0., 12. - account['confirmed_usd'] - account['reserved_usd'] - pending*teacher - .08)
    plan = {op: 0 for op in OPERATORS}
    # Alternate operators, including a final one-sided remainder only if the
    # other operator reached its quota. Do not bias towards an easier operator.
    for _ in range(max_per_operator):
        for op in OPERATORS:
            if counts[op] + plan[op] >= config[PHASE]['per_operator']:
                continue
            needed = qc + 2*teacher
            if available + 1e-12 < needed:
                continue
            plan[op] += 1
            available -= needed
    return {'counts_before': dict(counts), 'planned': plan,
        'outstanding_teacher_attempts': pending, 'teacher_cost_per_attempt_projected': teacher,
        'qc_cost_per_candidate_projected': qc, 'all_new_qc_assumed_pass': True,
        'qc_hold_usd': (pending + 2*sum(plan.values()))*teacher,
        'remaining_after_projection': available, 'accounting': account,
        'cap_usd': 12., 'max_per_operator_in_batch': max_per_operator}


def proposal_order(audit, source_id, operator, seed):
    values = list(audit['per_source'][source_id][operator]['valid_proposals'])
    random.Random(seed + int(source_id.split(':')[-1])*1000 + OPERATORS.index(operator)).shuffle(values)
    return values


def choose_variants(audit, original, existing, requested, seed):
    """Round-robin over the least-used sources; never repeat a corruption hash."""
    templates = {(r['source_id'], r['operator']): r for r in original.values()}
    used = defaultdict(set)
    for row in existing.values():
        used[(row['source_id'], row['operator'])].add(row['corrupted_hash'])
    selections = []
    for operator in OPERATORS:
        order = sorted(source for source, op in templates if op == operator)
        random.Random(seed + OPERATORS.index(operator)).shuffle(order)
        available = {source: [p for p in proposal_order(audit, source, operator, seed)
            if p['corrupted_hash'] not in used[(source, operator)]] for source in order}
        for _ in range(requested[operator]):
            possible = [source for source in order if available[source]]
            if not possible:
                break
            source = min(possible, key=lambda s: len(used[(s, operator)]))
            value = available[source].pop(0)
            used[(source, operator)].add(value['corrupted_hash'])
            selections.append((templates[(source, operator)], value, len(used[(source, operator)])))
    return selections


def make_variant(config, template, proposal_value, ordinal, batch_name, source, bank, tokenizer):
    operator = template['operator']
    proposal = Proposal(operator, proposal_value['sids'], deepcopy(proposal_value['params']))
    changed, record = apply(source, proposal, bank)
    restored = restore_record(changed, record, bank)
    if restored.text != source.text or not exact_restoration_satisfies(restored, record):
        raise ValueError('Expansion failed exact Phase3b restoration')
    view = render(changed.structure(), compact=True)
    tokens = len(tokenizer.encode(view, add_special_tokens=False))
    if tokens > 3000 or sha_text(changed.text) != proposal_value['corrupted_hash']:
        raise ValueError('Validated expansion proposal changed or exceeds the view limit')
    episode_id = f'global_boost:{batch_name}:{operator}:agent_train:{template["source_id"].split(":")[-1]}:v{ordinal}'
    seed = config[PHASE]['seed'] + int(template['source_id'].split(':')[-1])*1000 + ordinal
    record['record_id'] = episode_id + ':R1'
    record['params']['seed'] = seed
    record['coupled_changes'] = source_coupled_changes(record['coupled_changes'], source.structure())
    row = deepcopy(template)
    row.update(episode_id=episode_id, seed=seed, corrupted_text=changed.text,
        corrupted_hash=sha_text(changed.text), corrupted_layout=changed.snapshot(), records=[record],
        compact_view=view, compact_tokens=tokens, generation_rejections={},
        expansion={'batch': batch_name, 'source_variant': ordinal,
                   'original_practice_episode_id': template['episode_id']})
    return row


def prepare_batch(config, projection):
    from transformers import AutoTokenizer
    root = root_for(config)
    original = corpus(batch_config(config, root))
    audit = read_json(root / 'variant_supply_validated.json')
    if audit['sources'] != len({r['source_id'] for r in original.values()}):
        raise ValueError('Wait for the complete mechanical variant audit')
    existing = {}
    configs = batch_configs(config)
    for cfg in configs:
        existing.update(corpus(cfg))
    chosen = choose_variants(audit, original, existing, projection['planned'], config[PHASE]['seed'])
    if not chosen:
        return None
    batch_name = f'batch_{len(configs)+1:03}'
    output = root / 'batches' / batch_name
    if (output / 'source_plan.json').exists():
        raise ValueError('Never overwrite an existing frozen batch')
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    bank = BareunBank(config, cache_dir=root / 'bareun_units')
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    documents = {}
    entries = {op: [] for op in OPERATORS}
    for template, proposal, ordinal in chosen:
        source_id = template['source_id']
        if source_id not in documents:
            documents[source_id] = source_document(config, examples[source_id], bank)
        source = documents[source_id]
        if source.text != template['source_text'] or sha_text(source.text) != template['source_hash']:
            raise ValueError('Frozen source changed before expansion')
        row = make_variant(config, template, proposal, ordinal, batch_name, source, bank, tokenizer)
        path = output / 'candidates' / row['operator'] / (safe_id(row['episode_id']) + '.json')
        if path.exists():
            if read_json(path) != row:
                raise ValueError('Partial immutable expansion candidate changed')
        else:
            atomic_new(path, row)
        entries[row['operator']].append({'episode_id': row['episode_id'], 'source_id': source_id,
            'source_hash': row['source_hash'], 'path': str(path), 'sha256': file_sha(path)})
    parent = read_json(root / 'source_plan.json')
    plan = {'phase': PHASE, 'version': 'v1', 'batch': batch_name, 'budget_usd': 12,
        'shared_budget_root': str(root), 'source_plan_parent_sha256': file_sha(root / 'source_plan.json'),
        'variant_supply_sha256': file_sha(root / 'variant_supply_validated.json'),
        'prior_batches': {str(c['paths'][PHASE + '_output']): file_sha(c['paths'][PHASE + '_output'] / 'source_plan.json')
                         for c in configs},
        'source_policy': parent['source_policy'], 'source_inventory': parent['source_inventory'],
        'operators': list(OPERATORS), 'requested_per_operator': 400,
        'records_per_essay': 1, 'candidates': entries,
        'counts': {op: len(entries[op]) for op in OPERATORS},
        'source_overlap_across_operators': sorted({r['source_id'] for r in entries[OPERATORS[0]]}
            & {r['source_id'] for r in entries[OPERATORS[1]]}),
        'shortfall': {op: projection['planned'][op] - len(entries[op]) for op in OPERATORS},
        'scope': 'Distinct corruption practices may share the same eligible unused source.',
        'prompt_sha256': parent['prompt_sha256'], 'contract_sha256': parent['contract_sha256'],
        'budget_projection': projection, 'gpu_used': False, 'training': False, 'new_bareun_calls': 0}
    atomic_new(output / 'source_plan.json', plan)
    return batch_config(config, output)


def original_teacher_running(config):
    root = root_for(config)
    worker = root / 'controller_worker.json'
    status = root / 'status.json'
    if not worker.exists() or not status.exists():
        return False
    pid = read_json(worker)['pid']
    proc = Path(f'/proc/{pid}/cmdline')
    alive = proc.exists() and b'verak.v3.cli.global_boost' in proc.read_bytes()
    return alive and read_json(status).get('stage') in {'qc', 'teacher'}


def launch(config):
    root = root_for(config)
    path = root / 'expansion_worker.json'
    if path.exists():
        previous = read_json(path)
        process = Path(f'/proc/{previous["pid"]}/cmdline')
        if process.exists() and b'expand' in process.read_bytes() and b'verak.v3.cli.global_boost' in process.read_bytes():
            return previous
    with (root / 'expansion.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'verak.v3.cli.global_boost', 'expand'],
            cwd=str(Path(__file__).resolve().parents[3]), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': process.pid, 'at': time.time(), 'gpu_used': False, 'training': False}
    write_json(path, value)
    return value


def restart(config):
    """Only this expansion controller; preserve live original-teacher calls."""
    root = root_for(config)
    previous = read_json(root / 'expansion_worker.json')
    proc = Path(f'/proc/{previous["pid"]}')
    if not proc.exists():
        return launch(config)
    command = (proc / 'cmdline').read_bytes()
    if b'verak.v3.cli.global_boost' not in command or b'expand' not in command:
        raise RuntimeError('Refuse to stop a process outside the GLOBAL expansion')
    deadline = time.monotonic()+700
    while True:
        db = sqlite3.connect(root / 'api/ledger.sqlite', timeout=10)
        try:
            db.execute('BEGIN IMMEDIATE')
            pending = db.execute("SELECT COUNT(*) FROM calls WHERE status='pending' "
                "AND instr(item_id,'global_boost:batch_')>0").fetchone()[0]
            if not pending:
                os.kill(previous['pid'], signal.SIGTERM)
                for _ in range(50):
                    if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Owned expansion controller did not stop')
                write_json(root / f'expansion_restart_{int(time.time())}.json', {
                    'previous_pid': previous['pid'], 'no_pending_expansion_calls_under_lock': True,
                    'original_teacher_untouched': True, 'at': time.time()})
                break
        finally:
            db.rollback()
            db.close()
        if time.monotonic() >= deadline:
            raise TimeoutError('No safe GLOBAL expansion restart window')
        time.sleep(.2)
    return launch(config)


def run(config):
    from .qc import run as qc, summarize
    from .teacher import run as teacher
    from .prefetch import enqueue
    root = root_for(config)
    state = {'pid': os.getpid(), 'gpu_used': False, 'training': False}
    with collection_lock(root / 'expansion'):
        while True:
            unfinished = [cfg for cfg in batch_configs(config)[1:]
                          if not (cfg['paths'][PHASE + '_output'] / 'generation_finished.json').exists()]
            projection = projected_batch(config)
            if unfinished:
                cfg = unfinished[0]
            elif not any(projection['planned'].values()):
                reason = ('practice_quota' if all(projection['counts_before'].get(op, 0) >= 400 for op in OPERATORS)
                          else 'projected_budget_stop')
                break
            else:
                cfg = prepare_batch(config, projection)
                if cfg is None:
                    reason = 'mechanical_variant_supply'
                    break
            batch = cfg['paths'][PHASE + '_output']
            frozen = read_json(batch / 'source_plan.json')
            cfg[PHASE]['qc_hold_usd'] = frozen['budget_projection']['qc_hold_usd']
            write_json(root / 'expansion_status.json', {**state, 'stage': 'qc', 'batch': str(batch), 'at': time.time()})
            if not (batch / 'teacher_design.json').exists():
                qc(cfg)
            # The original 100 attempts keep their existing single teacher.
            # Do not introduce a second GLOBAL Bareun/editor worker.
            while original_teacher_running(config):
                write_json(root / 'expansion_status.json', {**state, 'stage': 'waiting_original_teacher',
                    'batch': str(batch), 'at': time.time()})
                time.sleep(30)
            # A transient Bareun priority pause is resumable, not a completed
            # initial corpus. Saved/failed slots remain immutable on resumption.
            initial_cfg = batch_config(config, root)
            initial_plan = read_json(root / 'teacher_design.json')
            initial_requested = 2*len(initial_plan['orders']['1'])
            if len(list(root.glob('attempt_*/episodes/*.json'))) < initial_requested:
                initial_status = teacher(initial_cfg)
                if initial_status['stop_reason'] == 'budget_cap':
                    reason = 'budget_cap'
                    break
                if initial_status['stop_reason'] == 'bareun_priority_suspended':
                    time.sleep(30)
                    continue
            write_json(root / 'expansion_status.json', {**state, 'stage': 'teacher', 'batch': str(batch), 'at': time.time()})
            result = teacher(cfg)
            enqueue(cfg)
            if result['stop_reason'] == 'bareun_priority_suspended':
                # This is an A-priority pause; do not close a partially collected
                # batch or treat waiting time as a budget stop.
                time.sleep(30)
                continue
            atomic_new(batch / 'generation_finished.json', {'teacher_status': result,
                'qc': summarize(cfg), 'at': time.time(), 'gpu_used': False})
            print(json.dumps({'batch': batch.name, 'teacher_status': result}), flush=True)
            if result['stop_reason'] == 'budget_cap':
                reason = result['stop_reason']
                break
        value = {**state, 'stage': 'generation_finished', 'stop_reason': reason,
            'projection': projected_batch(config), 'at': time.time(),
            'measurements_finished': False, 'canonical_for_selection': False}
        write_json(root / 'expansion_status.json', value)
        return value
