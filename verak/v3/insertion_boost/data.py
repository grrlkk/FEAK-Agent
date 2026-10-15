"""Remaining dependency-feasible train sources; exact historical QC reuse."""
from collections import Counter
from copy import deepcopy
import socket
import sqlite3

from ..common import file_sha, read_json, write_json
from ..train.teacher_bulk import atomic_new, collection_lock
from ..v2_ops.config import PHASE
from ..v2_ops.data import source_pools
from ..v2_ops.dependency import candidates as dependency_candidates
from ..v2_ops.operators import label_targets
from ..v2_ops.paid import V2API
from ..v2_ops.local import load_environment
from ..v2_ops.qc import candidate_paths
from ..v2_ops import retry


def prior_passes(config):
    old = config['paths']['retry_output']
    selected = []
    for path in sorted((old / 'candidates/G_DEL_LINK/agent_train').glob('*.json')):
        row = read_json(path)
        qc = old / 'qc' / (row['episode_id'].replace(':', '_') + '.json')
        if not qc.exists():
            continue
        verdict = read_json(qc)
        if verdict['candidate_sha256'] != file_sha(path):
            raise ValueError('Archived retry candidate changed after QC')
        if verdict['passed']:
            selected.append({'path': str(path), 'sha256': file_sha(path), 'qc_path': str(qc),
                'qc_sha256': file_sha(qc), 'source_id': row['source_id'], 'episode_id': row['episode_id']})
    if len(selected) != config['insertion_boost']['prior_train_passes']:
        raise ValueError('Prior passing train records no longer match the approved 78')
    return selected


def prior_qc_counts(config):
    old = config['paths']['retry_output']
    judged, passed = 0, 0
    for path in sorted((old / 'candidates/G_DEL_LINK').glob('*/*.json')):
        row = read_json(path)
        qc = old / 'qc' / (row['episode_id'].replace(':', '_') + '.json')
        if not qc.exists():
            continue
        value = read_json(qc)
        if value['candidate_sha256'] != file_sha(path):
            raise ValueError('Archived retry QC changed')
        judged += 1
        passed += value['passed']
    if (passed, judged) != (96, 270):
        raise ValueError('Prior QC evidence differs from the accepted gate decision')
    return {'passed': passed, 'judged': judged, 'source_yield': '96/380'}


def prepare(config):
    root = config['paths'][PHASE + '_output']
    prior_path = config['paths']['retry_output'] / 'source_plan.json'
    prior = read_json(prior_path)
    used = {r['source_id'] for r in prior['plans']['G_DEL_LINK']['agent_train']}
    remaining = [sid for sid in prior['screened_sources']['agent_train']['feasible_source_ids'] if sid not in used]
    if len(remaining) != config['insertion_boost']['remaining_feasible_train_sources']:
        raise ValueError('Remaining feasible source population changed')
    pools, audit = source_pools(config)
    lookup = {e.id: (e, source, score) for e, source, score in pools['agent_train']}
    entries = []
    for sid in remaining:
        example, source, score = lookup[sid]
        proposals = dependency_candidates(source)
        if not proposals or sid in audit['excluded_sources']:
            raise ValueError('Dependency or exclusion contract changed')
        entries.append({'source_id': sid, 'source_hash': example.essay_hash,
            'question_hash': example.question_hash, 'genre': example.genre,
            'label_targets': label_targets(source), 'dependency_candidates': proposals,
            'source_score': score['mean']})
    old_passes = prior_passes(config)
    result = {'version': 'v2', 'experiment': 'GLOBAL insertion data boost',
        'operators': ['G_DEL_LINK'], 'records_per_essay': 1,
        'plans': {'G_DEL_LINK': {'agent_train': entries, 'agent_dev': []}},
        'requested_source_counts': {'agent_train': len(entries), 'agent_dev': 0},
        'sampling': {'agent_train': {'remaining_feasible': len(entries),
            'genres': dict(Counter(e['genre'] for e in entries))}},
        'prior_source_plan': {'path': str(prior_path), 'sha256': file_sha(prior_path)},
        'prior_passing_train': old_passes, 'excluded_retry_train_sources': sorted(used),
        'exclusion_audit': audit, 'qc_gate': 'QC passed / judged >= 0.30',
        'prior_qc': prior_qc_counts(config),
        'budget_usd': 6, 'gpu_used': False, 'training': False}
    destination = root / 'source_plan.json'
    if destination.exists():
        if read_json(destination) != result:
            raise ValueError('Frozen insertion boost source design changed')
    else:
        atomic_new(destination, result)
    return result


def reuse(config, function):
    for prior_root in (config['paths']['prior_v2_output'], config['paths']['retry_output']):
        cfg = deepcopy(config)
        cfg['paths']['prior_v2_output'] = prior_root
        function(cfg)


def build(config):
    result = retry.build(config)
    for path in candidate_paths(config):
        row = read_json(path)
        if row['episode_id'].startswith('v2retry:'):
            # Local deterministic construction only; freeze this namespace before QC.
            row['episode_id'] = row['episode_id'].replace('v2retry:', 'v2boost:', 1)
            row['records'][0]['record_id'] = row['episode_id'] + ':R1'
            row['experiment'] = 'insertion_boost'
            write_json(path, row)
    return result


def summary(config):
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')
    counts = Counter(planned=len(plan['plans']['G_DEL_LINK']['agent_train']))
    failures = Counter()
    for path in candidate_paths(config):
        row = read_json(path)
        counts['generated'] += 1
        qc = root / 'qc' / (row['episode_id'].replace(':', '_') + '.json')
        if not qc.exists():
            counts['unknown_qc'] += 1
            continue
        value = read_json(qc)
        if value['candidate_sha256'] != file_sha(path):
            raise ValueError('QC candidate hash mismatch')
        counts['judged'] += 1
        counts['passed' if value['passed'] else 'rejected'] += 1
        failures.update(k for k, v in value['verdict'].items() if v is False)
    counts['role_or_surface_failure'] = len(list((root / 'candidate_failures/G_DEL_LINK/agent_train').glob('*.json')))
    counts['awaiting_construction'] = counts['planned']-counts['generated']-counts['role_or_surface_failure']
    passed = plan['prior_qc']['passed']+counts['passed']
    judged = plan['prior_qc']['judged']+counts['judged']
    result = {'gate': 'QC pass / judged >= 0.30', 'prior': plan['prior_qc'],
        'new_train': dict(counts), 'new_false_criteria': dict(failures),
        'new_qc_pass_rate': counts['passed']/counts['judged'] if counts['judged'] else None,
        'combined': {'passed': passed, 'judged': judged, 'qc_pass_rate': passed/judged},
        'decision': 'retain' if passed/judged >= .30 else 'drop',
        'passing_train_records': len(plan['prior_passing_train'])+counts['passed'],
        'gpu_used': False, 'training': False}
    write_json(root / 'qc_summary.json', result)
    return result


def run_qc(config, *, max_api_calls, paid_approved):
    if not paid_approved:
        raise PermissionError('Explicit user authorization required')
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        prepare(config)
        reuse(config, retry.reuse_labels)
        api = V2API(config, max_api_calls, kind='sol', paid_approved=True)
        stage, error = 'labels', None
        try:
            api.client()
            api.settle_interrupted()
            # Fail before dispatching batch workers if this process has no network.
            socket.getaddrinfo('api.openai.com', 443)
            reconcile_unsent_dns_errors(config)
            retry.label(config, api)
            stage = 'construction'
            build(config)
            reuse(config, retry.reuse_qc)
            stage = 'qc'
            retry.judge(config, api)
            stage = 'completed'
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            api.close()
            result = {'stage': stage, 'error': error, 'budget': api.accounting(),
                      'qc': summary(config), 'gpu_used': False, 'training': False}
            write_json(root / 'qc_status.json', result)
        return result


def reconcile_unsent_dns_errors(config):
    """Archive preflight-only errors only with proof that no request was sent.

    Paid/unknown outcomes are never moved or resampled. The exact failed files and
    an empty provisional QC plan remain available in a reconciliation archive.
    """
    root = config['paths'][PHASE + '_output']
    errors = list((root / 'judge_batches/v2_retry_labels').glob('*.json'))
    if not errors:
        return
    if not all(read_json(p).get('error') == 'gaierror: [Errno -3] Temporary failure in name resolution'
               for p in errors):
        return
    with sqlite3.connect(root / 'api/ledger.sqlite') as db:
        calls = db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]
    if calls or list((root / 'api/requests').glob('*.json')) or list((root / 'labels').glob('*.json')):
        return
    plan = root / 'qc_plan.json'
    if plan.exists() and read_json(plan)['all_candidates']:
        return
    archive = root / 'no_send_reconciliation/dns_preflight'
    archive.mkdir(parents=True, exist_ok=True)
    evidence = {'reason': 'sandbox DNS failed before ledger reservation or API send',
        'ledger_calls': 0, 'api_request_files': 0,
        'preserved': [{'path': str(p), 'sha256': file_sha(p)} for p in errors],
        'paid_calls_repeated': 0}
    for path in errors:
        path.rename(archive / path.name)
    if plan.exists():
        evidence['provisional_empty_qc_plan_sha256'] = file_sha(plan)
        plan.rename(archive / 'empty_qc_plan.json')
    atomic_new(archive / 'evidence.json', evidence)
