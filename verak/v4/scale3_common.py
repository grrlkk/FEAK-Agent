"""Gated v4.3 scale contracts; importing this module never starts work.

The smoke and scale ledgers are separate. B2 may open its ledger only after
all three smoke ledgers are settled and frozen, so their sum cannot exceed $50.
B3 has its own $30 ledger. Neither component owns a GPU or training process.
"""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import json
import os
import sqlite3
import time

from .common import REPO, atomic_new, file_sha, load_config, read_json, sha_text

ROOT = REPO / 'verak/v4/outputs/scale3'
CONTENT_CAP = 50.
CORRUPTION_CAP = 30.
ATTEMPTS = (1, 2)
MAX_WORKERS = {'B2': 6, 'B3': 4}


def trim_metrics(calls):
    """Count saved teacher-only transformations without interpreting old raw as actions."""
    diagnostics = [c.get('teacher_trim') for c in calls]
    return {'returned_responses': len(calls),
        'canonical_targets': sum(isinstance(c.get('canonical_action'), str) for c in calls),
        'raw_differs_from_canonical': sum(isinstance(c.get('canonical_action'), str) and
                                        c['canonical_action'] != c['raw'] for c in calls),
        'missing_diagnostics': sum(d is None for d in diagnostics),
        'trimmed': sum(bool(d and d.get('trimmed')) for d in diagnostics),
        'trimmed_first_action_valid': sum(bool(d and d.get('trimmed') and d.get('first_action_valid')) for d in diagnostics),
        'discarded_actions': sum((d or {}).get('discarded_action_count', 0) for d in diagnostics),
        'by_trim_kind': dict(Counter((d or {}).get('kind') or 'none' for d in diagnostics))}


def accounting(path):
    """Read-only ledger accounting, including unknown reservations in the cap."""
    path = Path(path)
    if not path.exists():
        return {'confirmed_usd': 0., 'reserved_usd': 0., 'pending': 0,
                'calls': 0, 'by_stage': {}, 'outcome_sha256': None}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT id,stage,item_id,status,reserved,confirmed FROM calls ORDER BY id').fetchall()
    return {'confirmed_usd': sum(r[5] for r in rows), 'reserved_usd': sum(r[4] for r in rows),
        'pending': sum(r[3] == 'pending' for r in rows),
        'calls': sum(r[3] != 'blocked_before_send' for r in rows),
        'by_stage': {stage: {'requests': sum(r[1] == stage and r[3] != 'blocked_before_send' for r in rows),
            'confirmed_usd': sum(r[5] for r in rows if r[1] == stage),
            'statuses': dict(Counter(r[3] for r in rows if r[1] == stage))}
            for stage in sorted({r[1] for r in rows})},
        'outcome_sha256': sha_text(json.dumps(rows, ensure_ascii=False))}


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen scale3 contract/artifact changed: ' + str(path))
    else:
        atomic_new(path, value)
    return value


def smoke_proof(root=ROOT):
    """B2/B3 fail closed until this exact v4.3 smoke passes and is settled."""
    from . import policy_prompts_v43 as prompts
    root = Path(root)
    gate_path, complete_path = root/'B1/gate.json', root/'B1/complete.json'
    gate, complete = read_json(gate_path), read_json(complete_path)
    if gate.get('passed') is not True or gate.get('stop_all_B') is not False:
        raise RuntimeError('v4.3 B1 gate has not passed; B2/B3 calls remain forbidden')
    if (gate.get('all20_started') is not True or gate.get('all20_saved') is not True or
        gate.get('minimum_valid_action_rate') != .9 or gate.get('returned_actions', 0) <= 0 or
        gate.get('valid_actions', 0)*10 < gate['returned_actions']*9):
        raise ValueError('B1 numeric evidence does not satisfy the complete same20 >=90% gate')
    if (complete.get('status') != 'passed' or complete.get('no_live_paid_calls') is not True or
        complete.get('gate_sha256') != file_sha(gate_path) or complete.get('gate') != gate or
        file_sha(complete['metrics_path']) != complete.get('metrics_sha256')):
        raise ValueError('B1 completion evidence does not prove the frozen passed gate')
    contract_path = root/'B1/contract.json'
    contract = read_json(contract_path)
    if contract.get('prompt_version') != prompts.VERSION:
        raise ValueError('B1 used a different frozen prompt version')
    prompts.verify_frozen()
    current_sample, prior_sample = root/'B1/sample.json', REPO/'verak/v4/outputs/scale2/B1/sample.json'
    actual_ids, previous_ids = read_json(current_sample)['source_ids'], read_json(prior_sample)['source_ids']
    if len(actual_ids) != 20 or len(set(actual_ids)) != 20 or set(actual_ids) != set(previous_ids):
        raise ValueError('B1 source identities differ from the authorized same20')
    smoke_ledgers = [REPO/'verak/v4/outputs/scale/B1/api/ledger.sqlite',
                     REPO/'verak/v4/outputs/scale2/B1/api/ledger.sqlite', root/'B1/api/ledger.sqlite']
    ledgers = []
    for path in smoke_ledgers:
        if not path.exists():
            raise FileNotFoundError('Required prior smoke ledger is absent: ' + str(path))
        account = accounting(path)
        if account['pending'] or account['reserved_usd']:
            raise RuntimeError('Smoke ledger still has live/unknown reservations')
        ledgers.append({'path': str(path), 'sha256': file_sha(path), **account})
    spent = sum(v['confirmed_usd'] for v in ledgers)
    if not 0 <= spent < CONTENT_CAP:
        raise RuntimeError('No authorized cumulative content budget remains')
    return {'gate_path': str(gate_path), 'gate_sha256': file_sha(gate_path),
        'complete_path': str(complete_path), 'complete_sha256': file_sha(complete_path),
        'contract_path': str(contract_path), 'contract_sha256': file_sha(contract_path),
        'sample_path': str(current_sample), 'sample_sha256': file_sha(current_sample),
        'previous_sample_path': str(prior_sample), 'previous_sample_sha256': file_sha(prior_sample),
        'prompt_version': prompts.VERSION, 'prompt_manifest_sha256': file_sha(prompts.FROZEN),
        'ledgers': ledgers, 'prior_confirmed_usd': spent,
        'B2_remaining_cap_usd': CONTENT_CAP-spent}


def component_contract(component, root=ROOT):
    if component not in {'B2', 'B3'}:
        raise ValueError('Only B2/B3 are authorized by this entry point')
    root = Path(root)
    proof = smoke_proof(root)
    sample_path = root/component/'sample.json'
    if not sample_path.exists():
        raise RuntimeError('Freeze the file-only sample before opening a paid component')
    if component == 'B2':
        # This manifest can be prepared before B1 exists. Verify now that the
        # actual smoke is still the authorized same20, already excluded by ID.
        smoke_ids = set(read_json(root/'B1/sample.json')['source_ids'])
        old_smoke = set(read_json(REPO/'verak/v4/outputs/scale2/B1/sample.json')['source_ids'])
        if smoke_ids != old_smoke or smoke_ids & set(read_json(sample_path)['source_ids']):
            raise ValueError('B2 sample and authorized same20 smoke are not disjoint')
        from .scale3_content import PLAN, judge_prompt
        teacher_contract = {'plan_sha256': sha_text(PLAN), 'judge_sha256': sha_text(judge_prompt()),
            'planner_max_items': 4, 'planner_max_INSERT_tasks': 1, 'Sol_pair_judge_calls_per_source': 1,
            'quality_gate': 'Dv3 six conditions plus cited support and paraphrase',
            'Orchestrator_targets': 'retain Sol raw plan and feedback-free structured payload; no token prefix invented'}
    else:
        teacher_contract = {'plan': 'record-derived tasks, no answer targets or search; Korean whole essay',
            'selection': 'R>=0.80, valid terminal STOP, <=1 rejection; inherited no-GLOBAL exception',
            'CPU_quality_scorer_calls': 0, 'GPU_reference_only_final_rewards': True}
    value = {'schema_version': 1, 'component': component, 'smoke': proof,
        'sample_path': str(sample_path), 'sample_sha256': file_sha(sample_path), 'teacher_contract': teacher_contract,
        'prompt_version': proof['prompt_version'], 'context_limit': 8192, 'output_limit': 1024,
        'attempts': 2, 'provider_sampling_seed': None,
        'sampling': 'two independent unseeded requests; no retry of a saved failed outcome',
        'revision_steps_per_delegation': 6, 'korean_steps': 14,
        'maximum_episode_workers': MAX_WORKERS[component], 'initial_episode_workers': 2,
        'ramp_after_started_attempts': 20, 'shared_Bareun_cache_misses_per_second': 1,
        'cap_usd': proof['B2_remaining_cap_usd'] if component == 'B2' else CORRUPTION_CAP,
        'cumulative_content_cap_usd': CONTENT_CAP if component == 'B2' else None,
        'GPU_used': False, 'training': False, 'automatic_GPU_dispatch': False,
        'reward_contract': reward_contract() if component == 'B3' else None,
        'teacher_response_trimming': 'teacher only; preserve raw and diagnostics, export canonical single action',
        'GPU_schedule_owner': 'root, after one-shot evaluation; explicit dispatch only'}
    return freeze(root/component/'contract.json', value)


def reward_contract():
    from verak.v3.reward import total, recovery, overedit
    return {'config': load_config()['reward'],
        'v1_code_sha256': {str(Path(m.__file__).relative_to(Path(__file__).parents[2])): file_sha(m.__file__)
                          for m in (total, recovery, overedit)}}


def api_for(component, kind='luna', *, root=ROOT):
    """Opening an API ledger is itself gated. No API object exists during prep."""
    from verak.v3.v2_ops.local import load_environment
    from .runtime_v43 import VersionedAPI
    if kind not in {'luna', 'sol'} or (component == 'B3' and kind != 'luna'):
        raise ValueError('Unexpected paid model for this scale component')
    contract = component_contract(component, root)
    config = load_config()
    load_environment(config)
    luna = read_json(config['paths']['phase4_output']/'models.json')['luna_model']
    phase = 'v43_scale_' + component.lower()
    config[phase] = {'model': luna if kind == 'luna' else 'gpt-6.1-sol',
        'max_cost_usd': contract['cap_usd'], 'max_concurrent_requests': MAX_WORKERS[component],
        'phase_api_ceiling': 100000}
    config['paths'][phase+'_output'] = Path(root)/component
    api = VersionedAPI(config, 100000, phase=phase)
    api.allowed_models = {luna, 'gpt-6.1-sol'} if component == 'B2' else {luna}
    return api


def component_accounting(component, root=ROOT):
    root = Path(root)
    account = accounting(root/component/'api/ledger.sqlite')
    contract = read_json(root/component/'contract.json')
    total = account['confirmed_usd'] + account['reserved_usd']
    if total > contract['cap_usd']+1e-9:
        raise ValueError('Component ledger exceeded its frozen cap')
    if component == 'B2':
        total += contract['smoke']['prior_confirmed_usd']
        if total > CONTENT_CAP+1e-9:
            raise ValueError('Cumulative content ledger cap exceeded')
    return {**account, 'component_cap_usd': contract['cap_usd'],
        'cumulative_confirmed_and_reserved_usd': total}


def low_priority_cpu(component=None):
    """No CUDA and one BLAS thread; distinct from the existing RFT/scorer workers."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    allowed = set(os.sched_getaffinity(0))
    chosen = allowed & set(range(108, 112) if component == 'B3' else range(104, 108))
    os.sched_setaffinity(0, chosen or sorted(allowed)[-2:])
    if os.nice(0) < 19:
        os.nice(19-os.nice(0))


def dispatch_limit(component, root, observed_attempts):
    """Parent may pause new episodes without interrupting an in-flight call.

    By default the first 20 attempts of this collection use two workers. Only
    healthy observed calls permit the authorized 6/4-worker ramp. This is the
    start of the full collection, not another paid pilot or sampling change.
    """
    root = Path(root)
    policy_path = root/'dispatch_policy.json'
    policy = read_json(policy_path) if policy_path.exists() else {}
    if policy.get('pause_new_episodes') or policy.get('paused_components', {}).get(component):
        return 0
    maximum = min(MAX_WORKERS[component], int(policy.get('max_workers', {}).get(component, MAX_WORKERS[component])))
    if maximum < 0:
        raise ValueError('Negative dispatch worker limit')
    if observed_attempts < 20 or policy.get('disable_ramp'):
        return min(2, maximum)
    ledger = root/component/'api/ledger.sqlite'
    if ledger.exists():
        with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as db:
            failed = db.execute("SELECT path FROM calls WHERE status NOT IN ('completed','pending','blocked_before_send')").fetchall()
        for (path,) in failed:
            value = read_json(path) if path and Path(path).exists() else {}
            detail = json.dumps({k: value.get(k) for k in ('error', 'error_type', 'http_status')}).lower()
            if not value or any(term in detail for term in ('429', 'timeout', 'timed out', 'rate_limit')):
                return min(2, maximum)
    return maximum


def bounded_episodes(component, jobs, worker, *, root=ROOT, observed_attempts=0, progress=None):
    """Resume-aware bounded episode dispatch; no API request is cancelled/retried."""
    root = Path(root)
    pending, results, iterator, exhausted, stopping = {}, [], iter(jobs), False, False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS[component]) as pool:
        while pending or not exhausted:
            limit = 0 if stopping else dispatch_limit(component, root, observed_attempts)
            while not exhausted and len(pending) < limit:
                try:
                    job = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(worker, job)] = job
            if not pending:
                if exhausted or stopping:
                    break
                # A pause only stops new episodes. The operator can adjust the
                # file while this process waits; no scorer/GPU status is inferred.
                time.sleep(5)
                continue
            finished, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                result = future.result()
                results.append(result)
                observed_attempts += result.get('new_started_attempts', 0)
                stopping |= result.get('budget_stop', False)
                if stopping:
                    exhausted = True
                if progress:
                    progress(len(results), result, observed_attempts)
    return results


def saved_cost_estimate(repo=REPO):
    metrics = read_json(Path(repo)/'verak/v4/outputs/prep3/content/metrics.json')
    cost = metrics['api']['confirmed_usd']
    count = metrics['planned_sources']
    return {'basis_path': str(Path(repo)/'verak/v4/outputs/prep3/content/metrics.json'),
        'basis_sources': count, 'basis_attempts': metrics['planned_attempts'],
        'basis_confirmed_usd': cost, 'B2_1000_source_extrapolation_usd': cost*1000/count,
        'uncertainty': 'v4.3 SEARCH and action counts can change cost; the ledger cap wins',
        'includes_planners_judges_and_two_teacher_attempts': True}
