"""Dependency-first G_DEL_LINK retry, isolated from both v1 and initial v2."""
from collections import Counter
import json
from pathlib import Path
import random

import yaml

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, write_json
from ..train.teacher_bulk import atomic_new, collection_lock
from .candidates import make_row
from .config import PHASE, config_for, require_v2
from .data import source_pools
from .dependency import candidates as dependency_candidates
from .judges import label_contract
from .judges import passes_qc, qc_contract, validate
from .local import load_environment
from .operators import delete_link, label_targets
from .paid import V2API
from .qc import batches, dispatch_batches, request, summary, candidate_paths


def config_for_retry():
    config = config_for(version='v2')
    overlay = Path(__file__).parents[1] / 'config_v2_retry.yaml'
    retry = yaml.safe_load(overlay.read_text(encoding='utf-8'))
    config.update(retry)
    config[PHASE]['operators'] = ['G_DEL_LINK']
    config[PHASE]['max_cost_usd'] = retry['v2_retry']['max_cost_usd']
    config[PHASE]['max_concurrent_requests'] = retry['v2_retry']['api_concurrency']
    config['paths']['v2_ops_output'] = config['paths']['repo'] / 'verak/v3/outputs/v2_ops_retry'
    config['paths']['v2_corpus'] = config['paths']['repo'] / 'verak/v3/data/corrupt_ops_v2_retry'
    config['paths']['prior_v2_output'] = config['paths']['repo'] / 'verak/v3/outputs/v2_ops'
    return config


def prepare(config):
    require_v2(config)
    root = config['paths'][PHASE + '_output']
    pools, audit = source_pools(config)
    plans, sampling, screen = {}, {}, {}
    for split_index, split in enumerate(('agent_train', 'agent_dev')):
        requested = config['v2_retry']['source_counts'][split]
        pool = list(pools[split])
        seed = config['v2_retry']['source_seed'] + 100 * split_index
        random.Random(seed).shuffle(pool)
        feasible, counts, rejected = [], Counter(), []
        for example, source, score in pool:
            proposals = dependency_candidates(source)
            counts['screened_sources'] += 1
            counts['eligible_source_sites'] += len(label_targets(source))
            counts['dependency_sites'] += len(proposals)
            if not proposals:
                counts['no_dependency_site'] += 1
                rejected.append(example.id)
                continue
            counts.update(h['kind'] for p in proposals for h in p['hints'])
            feasible.append({'source_id': example.id, 'source_hash': example.essay_hash,
                'question_hash': example.question_hash, 'genre': example.genre,
                'label_targets': label_targets(source), 'dependency_candidates': proposals,
                'source_score': score['mean']})
        selected = feasible[:requested]
        plans[split] = selected
        sampling[split] = {'requested': requested, 'selected': len(selected),
            'shortfall': max(0, requested-len(selected)), 'dependency_feasible_sources': len(feasible),
            'seed': seed, 'screening': dict(counts),
            'genres': dict(Counter(row['genre'] for row in selected))}
        screen[split] = {'feasible_source_ids': [r['source_id'] for r in feasible],
                        'excluded_no_dependency': rejected}
    result = {'version': 'v2', 'experiment': 'G_DEL_LINK dependency retry',
        'operators': ['G_DEL_LINK'], 'records_per_essay': 1,
        'requested_source_counts': config['v2_retry']['source_counts'],
        'plans': {'G_DEL_LINK': plans}, 'sampling': {'G_DEL_LINK': sampling},
        'source_policy': 'unchanged frozen valid-only genre-Q75 500–2500-character pool; dependency first, Sol role second',
        'exclusion_audit': audit, 'screened_sources': screen,
        'dependency_code_sha256': file_sha(Path(__file__).with_name('dependency.py')),
        'prior_l_fuse': {'acceptable_alternative_merges': 338, 'judged': 377, 'decision': 'dropped'},
        'budget_usd': 10, 'paid_calls': 0, 'gpu_used': False, 'training': False}
    train = {r['source_id'] for r in plans['agent_train']}
    dev = {r['source_id'] for r in plans['agent_dev']}
    train_q = {r['question_hash'] for r in plans['agent_train']}
    dev_q = {r['question_hash'] for r in plans['agent_dev']}
    if train & dev or train_q & dev_q or (train | dev) & audit['excluded_sources'].keys():
        raise ValueError('Retry source/question split or evaluation exclusion leakage')
    path = root / 'source_plan.json'
    if path.exists() and read_json(path) != result:
        raise ValueError('Frozen retry source design changed')
    atomic_new(path, result) if not path.exists() else None
    return result


def reuse_labels(config):
    root, old = config['paths'][PHASE + '_output'], config['paths']['prior_v2_output']
    plan = read_json(root / 'source_plan.json')['plans']['G_DEL_LINK']
    counts = Counter()
    for entries in plan.values():
        for row in entries:
            name = row['source_id'].replace(':', '_') + '.json'
            target, prior = root / 'labels' / name, old / 'labels' / name
            if target.exists():
                continue
            if not prior.exists():
                counts['requires_new_label'] += 1
                continue
            saved = read_json(prior)
            if saved['source_hash'] != row['source_hash'] or {x['sid'] for x in saved['labels']} != set(row['label_targets']):
                raise ValueError('Cached role labels do not match exact retry source/sites')
            if any(x['label'] not in {'topic', 'bridge', 'summary', 'none'} for x in saved['labels']):
                raise ValueError('Invalid prior Sol role label')
            atomic_new(target, {**saved, 'reused_from': str(prior), 'reused_sha256': file_sha(prior)})
            counts['reused_labels'] += 1
    counts['total_reused_labels'] = sum('reused_from' in read_json(p) for p in (root / 'labels').glob('*.json'))
    write_json(root / 'label_reuse.json', dict(counts))
    return dict(counts)


def label(config, api):
    root = config['paths'][PHASE + '_output']
    pools, _ = source_pools(config)
    plan = read_json(root / 'source_plan.json')['plans']['G_DEL_LINK']
    manifest = root / 'new_label_plan.json'
    if manifest.exists():
        frozen = read_json(manifest)
    else:
        frozen = [{'split': split, 'source_id': row['source_id']} for split, entries in plan.items() for row in entries
                  if not (root / 'labels' / (row['source_id'].replace(':', '_') + '.json')).exists()]
        atomic_new(manifest, frozen)
    entries = {row['source_id']: row for rows in plan.values() for row in rows}
    sources = {e.id: (e, source) for values in pools.values() for e, source, score in values}
    items = []
    for task in frozen:
        row = entries[task['source_id']]
        example, source = sources[row['source_id']]
        items.append(({**row, 'split': task['split'], 'question': example.question}, source))

    def one(batch):
        paths = [root / 'labels' / (row['source_id'].replace(':', '_') + '.json') for row, _ in batch]
        if all(p.exists() for p in paths):
            return
        messages, schema = label_contract(batch)
        try:
            result = request(api, root, messages, schema, 'v2_retry_labels', effort='low')
        except CallBudgetExceeded:
            raise
        except Exception:
            return
        for (row, _), path in zip(batch, paths):
            if path.exists():
                continue
            atomic_new(path, {'source_id': row['source_id'], 'source_hash': row['source_hash'],
                'contract_sha256': result['contract_sha256'], 'phase_call': result['phase_call'],
                'labels': [{'sid': sid, 'label': value} for sid, value in result['verdicts'][row['source_id']].items()]})
        print(json.dumps({'stage': 'retry_labels', 'budget': api.accounting()}), flush=True)
    dispatch_batches(batches(items, config[PHASE]['label_batch_size']), one, config[PHASE]['max_concurrent_requests'])


def build(config):
    root = config['paths'][PHASE + '_output']
    pools, _ = source_pools(config)
    plan = read_json(root / 'source_plan.json')['plans']['G_DEL_LINK']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    stats = {}
    for split, entries in plan.items():
        sources = {e.id: (e, doc, score) for e, doc, score in pools[split]}
        counts = Counter()
        for entry in entries:
            name = entry['source_id'].replace(':', '_') + '.json'
            path = root / 'candidates/G_DEL_LINK' / split / name
            if path.exists():
                if read_json(path)['source_hash'] != entry['source_hash']:
                    raise ValueError('Changed retry candidate')
                counts['reused'] += 1
                continue
            failure_path = root / 'candidate_failures/G_DEL_LINK' / split / name
            if failure_path.exists():
                if read_json(failure_path)['source_hash'] != entry['source_hash']:
                    raise ValueError('Changed retry failure source')
                counts['reused_role_or_surface_failure'] += 1
                continue
            labels_path = root / 'labels' / name
            if not labels_path.exists():
                counts['awaiting_label'] += 1
                continue
            labels = read_json(labels_path)
            if labels['source_hash'] != entry['source_hash']:
                raise ValueError('Changed retry role-label source')
            roles = {row['sid']: row['label'] for row in labels['labels']}
            proposals = [p for p in entry['dependency_candidates'] if roles[p['sid']] in {'topic', 'bridge', 'summary'}]
            example, source, score = sources[entry['source_id']]
            random.Random(config[PHASE]['seed'] + example.source_line).shuffle(proposals)
            result, failures = None, Counter()
            for dependency in proposals:
                try:
                    changed, record = delete_link(source, dependency['sid'], roles[dependency['sid']])
                    result = make_row(example, source, score, changed, record, 'G_DEL_LINK', split, tokenizer)
                    result['episode_id'] = result['episode_id'].replace('v2:', 'v2retry:', 1)
                    result['records'][0]['record_id'] = result['episode_id'] + ':R1'
                    result['dependency_screen'] = dependency
                    result['role_labels_sha256'] = file_sha(labels_path)
                    break
                except ValueError as exc:
                    failures[str(exc)] += 1
            if result is None:
                atomic_new(failure_path,
                    {'source_id': entry['source_id'], 'source_hash': entry['source_hash'],
                     'reason': 'no_dependency_site_with_eligible_role' if not proposals else 'no_valid_surface_candidate',
                     'failures': dict(failures)})
                counts['role_or_surface_failure'] += 1
            else:
                result['mechanical_rejections_before_selection'] = dict(failures)
                atomic_new(path, result)
                counts['created'] += 1
        stats[split] = dict(counts)
    result = {'counts': stats, 'new_bareun_calls': 0, 'new_api_calls': 0, 'gpu_used': False}
    write_json(root / 'local_build_G_DEL_LINK.json', result)
    return result


def qc_identity(row):
    record = row['records'][0]
    return {k: row[k] for k in ('operator', 'question', 'source_text', 'corrupted_text')} | {
        'changed_source_sentences': record['original_text'], 'operation': record['params']}


def reuse_qc(config):
    root, prior = config['paths'][PHASE + '_output'], config['paths']['prior_v2_output']
    counts = Counter()
    for path in candidate_paths(config):
        row = read_json(path)
        destination = root / 'qc' / (row['episode_id'].replace(':', '_') + '.json')
        if destination.exists():
            continue
        old_path = prior / 'candidates/G_DEL_LINK' / row['split'] / path.name
        if not old_path.exists():
            continue
        old = read_json(old_path)
        qc_path = prior / 'qc' / (old['episode_id'].replace(':', '_') + '.json')
        if qc_identity(old) != qc_identity(row) or not qc_path.exists():
            continue
        saved = read_json(qc_path)
        if saved['candidate_sha256'] != file_sha(old_path):
            raise ValueError('Prior QC candidate hash mismatch')
        _, schema = qc_contract([row])
        validate({row['episode_id']: saved['verdict']}, schema)
        if saved['passed'] != passes_qc(saved['verdict'], row['operator']):
            raise ValueError('Prior QC verdict/decision mismatch')
        atomic_new(destination, {**saved, 'episode_id': row['episode_id'], 'candidate_sha256': file_sha(path),
            'reused_from': str(qc_path), 'reused_sha256': file_sha(qc_path),
            'original_candidate_sha256': file_sha(old_path),
            'reuse_contract': 'identical question/operator/source/corruption/changed-sentences/operation; episode identifier only differs'})
        counts['reused'] += 1
        counts['reused_pass' if saved['passed'] else 'reused_reject'] += 1
    counts['total_reused'] = sum('reused_from' in read_json(p) for p in (root / 'qc').glob('*.json'))
    write_json(root / 'qc_reuse.json', dict(counts))
    return dict(counts)


def judge(config, api):
    root = config['paths'][PHASE + '_output']
    path = root / 'qc_plan.json'
    candidates = [{'path': str(p), 'sha256': file_sha(p)} for p in candidate_paths(config)]
    if path.exists():
        manifest = read_json(path)
        if manifest['all_candidates'] != candidates:
            raise ValueError('Frozen retry QC candidates changed')
    else:
        entries = []
        for item in candidates:
            row = read_json(item['path'])
            if not (root / 'qc' / (row['episode_id'].replace(':', '_') + '.json')).exists():
                entries.append(item)
        manifest = {'source_plan_sha256': file_sha(root / 'source_plan.json'),
                    'all_candidates': candidates, 'new_candidates': entries,
                    'batch_size': config[PHASE]['qc_batch_size']}
        atomic_new(path, manifest)

    def one(batch):
        rows = [read_json(item['path']) for item in batch]
        paths = [root / 'qc' / (r['episode_id'].replace(':', '_') + '.json') for r in rows]
        if all(p.exists() for p in paths):
            return
        messages, schema = qc_contract(rows)
        try:
            result = request(api, root, messages, schema, 'v2_retry_instance_qc', effort='high')
        except CallBudgetExceeded:
            raise
        except Exception:
            return
        for item, row, destination in zip(batch, rows, paths):
            if destination.exists():
                continue
            verdict = result['verdicts'][row['episode_id']]
            atomic_new(destination, {'episode_id': row['episode_id'], 'candidate_sha256': item['sha256'],
                'operator': 'G_DEL_LINK', 'split': row['split'], 'verdict': verdict,
                'passed': passes_qc(verdict, 'G_DEL_LINK'), 'contract_sha256': result['contract_sha256'],
                'phase_call': result['phase_call']})
        print(json.dumps({'stage': 'retry_instance_qc', 'budget': api.accounting()}), flush=True)
    dispatch_batches(batches(manifest['new_candidates'], manifest['batch_size']), one,
                     config[PHASE]['max_concurrent_requests'])


def run_qc(config, *, max_api_calls, paid_approved):
    if not paid_approved:
        raise PermissionError('Explicit user authorization required')
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        prepare(config)
        reuse_labels(config)
        api = V2API(config, max_api_calls, kind='sol', paid_approved=True)
        stage, error = 'labels', None
        try:
            api.client()
            api.settle_interrupted()
            label(config, api)
            stage = 'construction'
            build(config)
            reuse_qc(config)
            stage = 'qc'
            judge(config, api)
            stage = 'completed'
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            api.close()
            status = {'stage': stage, 'error': error, 'budget': api.accounting(), 'qc': summary(config),
                      'gpu_used': False, 'training': False}
            write_json(root / 'qc_status.json', status)
        return status
