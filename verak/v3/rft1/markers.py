"""Reuse the accepted Sol final-marker contract under one A/C $3 ledger."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, sha_text, write_json
from ..observation.markers import FIELD, JUDGE_PROMPT, at, contract
from ..phase2 import read_jsonl, write_jsonl
from ..train.sft_markers import baseline_reuse, cases_for
from ..train.teacher_bulk import BulkAPI, atomic_new, collection_lock
from .config import PHASE, load_environment
from .evaluate import saved_rows


def oneshot_cases(row):
    def layout_rows(layout):
        return [{'sid': u['sid'], 'text': u['text'], 'paragraph': p['pid']}
                for p in layout['paragraphs'] for u in p['units']]
    before, after = layout_rows(row['initial_layout']), layout_rows(row['final_layout'])
    aid = 'oneshot:' + row['corpus_episode_id']
    sites = defaultdict(set)
    for unit in after:
        old, new = at(before, unit['sid']), at(after, unit['sid'])
        previous = lambda value: value['previous']['sid'] if value and value['previous'] else None
        if old is None or previous(old) != previous(new):
            sites[unit['sid']].update(('conjunction', 'subject_omission', 'ending_style'))
    for fact in row.get('marker_changes', []):
        sites[fact['sid']].add(FIELD[fact['type']])
    cases = []
    for sid, fields in sorted(sites.items()):
        case = {'case_id': aid + ':' + sid, 'action_id': aid, 'sid': sid, 'condition': 'oneshot',
            'essay_id': row['corpus_episode_id'], 'fields': sorted(fields), 'before': at(before, sid),
            'after_structural': at(after, sid), 'final': at(after, sid), 'completed': row['completed']}
        messages, schema = contract(case)
        case['contract_sha256'] = sha_text(json.dumps([messages, schema], ensure_ascii=False, sort_keys=True))
        cases.append(case)
    action = {'action_id': aid, 'condition': 'oneshot', 'essay_id': row['corpus_episode_id'],
        'kind': 'whole_essay_rewrite', 'completed': row['completed'], 'case_ids': [c['case_id'] for c in cases],
        'deleted_targets': [], 'sites': sorted(sites),
        'denominator_note': 'One whole-essay generation per essay, not a fabricated structural action'}
    return [action], cases


def extract(config, condition):
    root = config['paths'][PHASE + '_output'] / 'markers' / condition
    actions, cases, hashes = [], [], {}
    for row, path, events in saved_rows(config, condition):
        if row['cohort'] != 'real':
            continue
        hashes[str(path)] = file_sha(path)
        if condition == 'rft1':
            hashes[str(events)] = file_sha(events)
            saved_events = read_jsonl(events)
            # Invalid argument containers are never marker cases; keep their
            # observations for state replay without asking the old classifier to parse them.
            for event in saved_events:
                if event['event'] == 'action' and not event['action']['valid']:
                    event['action'] = {**event['action'], 'action': 'INVALID', 'args': {}}
            own_actions, own_cases = cases_for(row, saved_events, condition)
        else:
            own_actions, own_cases = oneshot_cases(row)
        actions.extend(own_actions)
        cases.extend(own_cases)
    write_jsonl(root / 'cases.jsonl', cases)
    write_jsonl(root / 'actions.jsonl', actions)
    write_json(root / 'provenance.json', {'source_sha256': hashes, 'judge_prompt_sha256': sha_text(JUDGE_PROMPT),
        'model': 'gpt-6.1-sol', 'reasoning_effort': 'high', 'max_output_tokens': 4096,
        'cases': len(cases), 'eligible_cases': sum(c['completed'] for c in cases),
        'definition': ('Accepted GLOBAL structural-action final-marker cases' if condition == 'rft1' else
            'Record-blind rewrite alignment: changed predecessor/new sentence and observed internal marker changes; per rewrite denominator')})
    return actions, cases


def configure(config):
    phase = PHASE + '_sol'
    config[phase] = {'model': 'gpt-6.1-sol', 'max_cost_usd': 3.,
                     'max_concurrent_requests': 4, 'phase_api_ceiling': 2000}
    config['paths'][phase + '_output'] = config['paths'][PHASE + '_output'] / 'sol'
    return phase


def run(config, condition, *, max_api_calls=2000):
    root = config['paths'][PHASE + '_output'] / 'markers'
    phase = configure(config)
    load_environment(config)
    with collection_lock(root):
        _, cases = extract(config, condition)
        api = BulkAPI(config, max_api_calls, phase=phase)
        api.settle_interrupted()
        prior = baseline_reuse(config)
        sft_markers = config['paths']['phase7_sft_output'] / 'markers'
        if read_json(sft_markers / 'provenance.json')['judge_prompt_sha256'] != sha_text(JUDGE_PROMPT):
            raise ValueError('Accepted SFT judge prompt changed')
        for path in (sft_markers / 'judgments').glob('*.json'):
            value = read_json(path)
            if value['status'] == 'completed':
                prior[path.stem] = {**value, 'reused': True, 'path': str(path), 'sha256': file_sha(path)}
        tasks, unique = [], set()
        for case in cases:
            key = case['contract_sha256']
            if not case['completed'] or key in unique:
                continue
            unique.add(key)
            target = root / 'judgments' / (key + '.json')
            if target.exists():
                continue
            if key in prior:
                atomic_new(target, prior[key])
            else:
                tasks.append(case)
        errors, stopped = [], False

        def one(case):
            nonlocal stopped
            if stopped:
                return
            key = case['contract_sha256']
            try:
                messages, schema = contract(case)
                response = api.request(messages, stage='rft1_oneshot_real_marker_fit', item_id=key,
                                       effort='high', max_output=4096, schema=schema)
                judgment = json.loads(response['raw'])
                if set(judgment) != set(case['fields']) or any(type(v['still_fits']) is not bool for v in judgment.values()):
                    raise ValueError('Invalid marker verdict')
                atomic_new(root / 'judgments' / (key + '.json'), {'contract_sha256': key,
                    'status': 'completed', 'reused': False, 'judgment': judgment, 'phase_call': response['phase_call']})
                print(json.dumps({'marker_condition': condition, 'case': case['case_id'],
                                  'usd': api.accounting()['confirmed_usd']}), flush=True)
            except Exception as exc:
                errors.append({'case': case['case_id'], 'type': type(exc).__name__, 'message': str(exc)})
                if isinstance(exc, CallBudgetExceeded):
                    stopped = True
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(one, tasks))
        finally:
            api.close()
            write_json(root / condition / 'status.json', {'new_unique_requests_planned': len(tasks),
                'errors': errors, 'stop_reason': 'budget_cap' if stopped else 'all_requested_attempted',
                'budget': api.accounting(), 'budget_scope': 'A and C marker checks combined <= $3'})
    return summary(config, condition)


def summary(config, condition):
    actions, cases = extract(config, condition)
    root = config['paths'][PHASE + '_output'] / 'markers'
    judgments = {}
    for case in cases:
        path = root / 'judgments' / (case['contract_sha256'] + '.json')
        if path.exists() and read_json(path)['status'] == 'completed':
            judgments[case['case_id']] = read_json(path)['judgment']
    details = []
    for action in actions:
        judged = [i for i in action['case_ids'] if i in judgments]
        bad = [i for i in judged if any(not value['still_fits'] for value in judgments[i].values())]
        details.append({**action, 'bad_cases': bad,
                        'unknown': not action['completed'] or len(judged) < len(action['case_ids'])})
    n = len(details)
    bad = sum(bool(a['bad_cases']) for a in details)
    unknown = sum(a['unknown'] and not a['bad_cases'] for a in details)
    result = {'condition': condition, 'units': n,
        'unit': 'structural_actions' if condition == 'rft1' else 'whole_essay_rewrites',
        'flagged_units': bad, 'unknown_units': unknown,
        'share': bad / n if n and not unknown else None,
        'lower_bound': bad / n if n else None, 'upper_bound': (bad + unknown) / n if n else None,
        'cases': len(cases), 'eligible_cases': sum(c['completed'] for c in cases), 'judged_cases': len(judgments),
        'flagged_cases': sum(any(not v['still_fits'] for v in j.values()) for j in judgments.values()),
        'flagged_essays': len({a['essay_id'] for a in details if a['bad_cases']}), 'details': details}
    write_json(root / condition / 'summary.json', result)
    return result
