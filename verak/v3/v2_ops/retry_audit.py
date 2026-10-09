"""Source, cache, API-ledger and isolation checks for the bounded retry."""
import json
from pathlib import Path
import sqlite3
import subprocess

from ..common import file_sha, read_json, sha_text, write_json
from ..reconstruction_api import usage_cost
from .config import PHASE
from .data import source_pools
from .dependency import candidates as dependency_candidates
from .operators import delete_link
from .qc import candidate_paths, summary
from .retry import qc_identity
from .report import accounting


def run(config):
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')
    checks = 0
    def check(value, message):
        nonlocal checks
        checks += 1
        if not value:
            raise AssertionError(message)
    pools, exclusion = source_pools(config)
    planned_ids, questions = {}, {}
    for split, entries in plan['plans']['G_DEL_LINK'].items():
        sources = {e.id: (e, source) for e, source, _ in pools[split]}
        planned_ids[split] = {r['source_id'] for r in entries}
        questions[split] = {r['question_hash'] for r in entries}
        check(len(entries) == config['v2_retry']['source_counts'][split], 'retry requested source count')
        check(len(planned_ids[split]) == len(entries), 'duplicate source')
        for entry in entries:
            example, source = sources[entry['source_id']]
            check(entry['source_hash'] == example.essay_hash == sha_text(source.text), 'source hash')
            check(entry['dependency_candidates'] == dependency_candidates(source), 'dependency screen changed')
            check(bool(entry['dependency_candidates']), 'no dependency hint')
            label_path = root / 'labels' / (entry['source_id'].replace(':', '_') + '.json')
            if label_path.exists():
                label = read_json(label_path)
                check(label['source_hash'] == entry['source_hash'], 'label source')
                check(set(x['sid'] for x in label['labels']) == set(entry['label_targets']), 'label sites')
                if 'reused_from' in label:
                    check(file_sha(label['reused_from']) == label['reused_sha256'], 'old label changed')
                    old = read_json(label['reused_from'])
                    check(label['labels'] == old['labels'], 'cached role label changed')
                else:
                    raw = read_json(root / 'api/requests' / f"{label['phase_call']:06}.json")
                    check(raw['model'] == config[PHASE]['sol_model'] and raw['reasoning_effort'] == 'low', 'new role label model/effort')
                    check({r['sid']: r['label'] for r in label['labels']} == json.loads(raw['raw'])[entry['source_id']], 'role label raw verdict')
            candidate_path = root / 'candidates/G_DEL_LINK' / split / label_path.name
            failure_path = root / 'candidate_failures/G_DEL_LINK' / split / label_path.name
            check(not (candidate_path.exists() and failure_path.exists()), 'candidate and failure coexist')
            if not candidate_path.exists():
                if failure_path.exists():
                    roles = {x['sid']: x['label'] for x in read_json(label_path)['labels']}
                    failure = read_json(failure_path)
                    if failure['reason'] == 'no_dependency_site_with_eligible_role':
                        check(not any(roles[p['sid']] != 'none' for p in entry['dependency_candidates']), 'false role-filter rejection')
                continue
            row = read_json(candidate_path)
            record = row['records'][0]
            roles = {x['sid']: x['label'] for x in read_json(label_path)['labels']}
            sid = record['sids'][0]
            check(len(row['records']) == len(record['sids']) == 1, 'not exactly one deletion')
            check(row['dependency_screen'] in entry['dependency_candidates'], 'deleted site not in dependency screen')
            check(row['dependency_screen']['sid'] == sid, 'wrong dependency sentence')
            check(record['params']['deleted_label'] == roles[sid] != 'none', 'role gate')
            changed, expected = delete_link(source, sid, roles[sid])
            check(row['corrupted_text'] == changed.text and row['corrupted_layout'] == changed.snapshot(), 'deletion output')
            check(record['inverse'] == source.snapshot(), 'inverse/source changed')
            check(row['source_text'] == source.text and row['corrupted_hash'] == sha_text(changed.text), 'text hash')
            check(len(changed.units) == len(source.units)-1, 'deletion count')
            check(row['q_corrupted'] is None, 'unexpected pre-QC GPU score')
            verdict_path = root / 'qc' / (row['episode_id'].replace(':', '_') + '.json')
            if verdict_path.exists():
                qc = read_json(verdict_path)
                check(qc['candidate_sha256'] == file_sha(candidate_path), 'QC candidate changed')
                check(qc['passed'] == all(qc['verdict'][key] for key in ('damage_real', 'original_is_fix', 'recoverable_from_essay')), 'QC aggregation')
                if 'reused_from' in qc:
                    check(file_sha(qc['reused_from']) == qc['reused_sha256'], 'old QC changed')
                    old_path = config['paths']['prior_v2_output'] / 'candidates/G_DEL_LINK' / split / candidate_path.name
                    check(file_sha(old_path) == qc['original_candidate_sha256'], 'old candidate changed')
                    check(qc_identity(row) == qc_identity(read_json(old_path)), 'nonidentical QC reuse')
                    check(qc['verdict'] == read_json(qc['reused_from'])['verdict'], 'reused verdict changed')
                else:
                    raw = read_json(root / 'api/requests' / f"{qc['phase_call']:06}.json")
                    check(raw['model'] == config[PHASE]['sol_model'] and raw['reasoning_effort'] == 'high', 'new QC model/effort')
                    check(json.loads(raw['raw'])[row['episode_id']] == qc['verdict'], 'QC raw verdict mismatch')
    check(not (planned_ids['agent_train'] & planned_ids['agent_dev']), 'source split leak')
    check(not (questions['agent_train'] & questions['agent_dev']), 'question split leak')
    check(not (planned_ids['agent_train'] | planned_ids['agent_dev']) & exclusion['excluded_sources'].keys(), 'evaluation leak')
    budget = accounting(root)
    check(budget['pending'] == 0, 'pending requests at final audit')
    check(budget['confirmed_usd'] + budget['reserved_usd'] <= 10+1e-9, 'cost cap')
    events, raw_hashes = [], {}
    path = root / 'api/ledger.sqlite'
    with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as db:
        records = db.execute('SELECT id,stage,item_id,fingerprint,status,reserved,confirmed,path,created,finished FROM calls ORDER BY id').fetchall()
    fingerprints = set()
    for call, stage, item_id, fingerprint, status, reserved, confirmed, raw_path, created, finished in records:
        check(fingerprint not in fingerprints, 'duplicate sent request')
        fingerprints.add(fingerprint)
        path = root / 'api/requests' / f'{call:06}.json'
        raw = read_json(path)
        raw_hashes[str(path)] = file_sha(path)
        check(raw['fingerprint'] == fingerprint and raw['phase_call'] == call, 'request identity')
        check(raw['stage'] == stage and raw['item_id'] == item_id, 'request routing')
        contract = {k: raw[k] for k in ('stage', 'item_id', 'model', 'reasoning_effort', 'max_output_tokens', 'messages', 'schema')}
        check(sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True)) == fingerprint, 'request fingerprint')
        if raw.get('usage'):
            recomputed = usage_cost(raw['model'], raw['usage'])['confirmed_usd']
            check(abs(recomputed-confirmed) < 1e-10, 'ledger usage/cost')
            check(abs(recomputed-raw['cost']['confirmed_usd']) < 1e-10, 'raw usage/cost')
        bound = raw['bound_usd']
        check(confirmed <= bound+1e-9, 'actual cost above reservation')
        events.append((created, 1, bound, 0, 1))
        if finished is not None:
            events.append((finished, 0, reserved-bound, confirmed, -1))
    committed, active, max_committed, max_active = 0., 0, 0., 0
    for at, order, reserve_change, charge, active_change in sorted(events):
        committed += reserve_change + charge
        active += active_change
        max_committed, max_active = max(max_committed, committed), max(max_active, active)
        check(committed <= 10+1e-9, 'historical cost cap')
        check(active <= 2, 'historical request concurrency')
    check(active == 0, 'live request count')
    check(abs(committed-budget['confirmed_usd']-budget['reserved_usd']) < 1e-8, 'timeline accounting')
    repo = Path(__file__).parents[3]
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=repo, text=True).splitlines()
    check(not any(p.startswith(('verak/v3/agent/', 'verak/v3/env/', 'verak/v3/reward/', 'verak/v3/score/', 'verak/v3/ko/'))
                  or p == 'verak/v3/config.yaml' for p in changed), 'v1 code changed')
    replay = read_json(root / 'v1_replay.json')
    check(replay['passed'] and replay['trajectories'] == 92 and replay['teacher_turns_replayed'] == 1165, 'v1 replay failed')
    result = {'passed': True, 'checks': checks, 'qc': summary(config), 'budget': budget,
        'max_historical_commitment_usd': max_committed, 'max_historical_api_concurrency': max_active,
        'api_raw_sha256': raw_hashes, 'source_plan_sha256': file_sha(root / 'source_plan.json'),
        'candidate_sha256': {str(p): file_sha(p) for p in candidate_paths(config)},
        'gpu_used': False, 'training': False}
    write_json(root / 'final_integrity_audit.json', result)
    return {k: v for k, v in result.items() if k not in {'api_raw_sha256', 'candidate_sha256'}}
