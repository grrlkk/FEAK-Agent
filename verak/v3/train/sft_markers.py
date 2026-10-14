"""The accepted final-text Sol marker check, applied to SFT real outputs."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
from pathlib import Path

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, sha_text, write_json
from ..observation.markers import FIELD, JUDGE_PROMPT, at, contract, update_state
from ..phase2 import read_jsonl, write_jsonl
from .sft_data import PHASE
from .sft_eval import CONDITIONS, action_kind, saved_rows
from .teacher_bulk import BulkAPI, atomic_new, collection_lock, load_environment


def cases_for(row, events, condition):
    state, steps = [], []
    for event in events:
        if event['event'] == 'reset':
            state, steps = update_state(event['observation']), []
        elif event['event'] == 'action':
            before = deepcopy(state)
            state = update_state(event['observation'], state)
            if event['role'] == 'global':
                steps.append((event['action'], before, deepcopy(state)))
    final_texts = [' '.join(u['text'].split()) for p in row['final_layout']['paragraphs'] for u in p['units']]
    if [' '.join(r['text'].split()) for r in state] != final_texts:
        raise ValueError('Observations do not reproduce final layout: ' + row['corpus_episode_id'])
    actions, cases = [], []
    for action, before, after in steps:
        kind = action_kind(action)
        if not action['valid'] or kind not in {'MOVE', 'sentence_insert', 'sentence_delete'}:
            continue
        aid = f"{condition}:{row['corpus_episode_id']}:G{action['t']}"
        sites = defaultdict(set)
        for unit in after:
            old, new = at(before, unit['sid']), at(after, unit['sid'])
            prev = lambda value: value['previous']['sid'] if value and value['previous'] else None
            if old is None or prev(old) != prev(new):
                sites[unit['sid']].update(('conjunction', 'subject_omission', 'ending_style'))
        for fact in action['cohesion_changes']:
            sites[fact['sid']].add(FIELD[fact['type']])
        ar = {'action_id': aid, 'condition': condition, 'essay_id': row['corpus_episode_id'],
            'kind': kind, 'completed': row['completed'], 'case_ids': [], 'deleted_targets': [],
            'sites': sorted(sites), 't': action['t']}
        for sid, fields in sorted(sites.items()):
            final = at(state, sid)
            if final is None:
                ar['deleted_targets'].append(sid)
                continue
            cid = aid + ':' + sid
            case = {'case_id': cid, 'action_id': aid, 'sid': sid, 'condition': condition,
                'essay_id': row['corpus_episode_id'], 'fields': sorted(fields),
                'before': at(before, sid), 'after_structural': at(after, sid), 'final': final,
                'completed': row['completed']}
            messages, schema = contract(case)
            case['contract_sha256'] = sha_text(json.dumps([messages, schema], ensure_ascii=False, sort_keys=True))
            cases.append(case)
            ar['case_ids'].append(cid)
        actions.append(ar)
    return actions, cases


def extract(config):
    root = config['paths'][PHASE + '_output'] / 'markers'
    actions, cases, hashes = [], [], {}
    for condition in CONDITIONS:
        for row, path, events in saved_rows(config, condition):
            if row['cohort'] != 'real':
                continue
            hashes[str(path)], hashes[str(events)] = file_sha(path), file_sha(events)
            found_actions, found_cases = cases_for(row, read_jsonl(events), condition)
            actions.extend(found_actions)
            cases.extend(found_cases)
    write_jsonl(root / 'cases.jsonl', cases)
    write_jsonl(root / 'actions.jsonl', actions)
    write_json(root / 'provenance.json', {'source_sha256': hashes, 'judge_prompt_sha256': sha_text(JUDGE_PROMPT),
        'definition': 'Same observation-test check: final surviving sites after valid GLOBAL structural actions; '
            'changed adjacency/insertion plus internal Bareun notices. Incomplete episodes remain unknown.',
        'model': 'gpt-6.1-sol', 'reasoning_effort': 'high', 'max_output_tokens': 4096,
        'cases': len(cases), 'eligible_cases': sum(c['completed'] for c in cases)})
    return actions, cases


def baseline_reuse(config):
    old = config['paths'][PHASE + '_output'].parent / 'observation_test'
    if read_json(old / 'marker_extraction.json')['judge_prompt'] != JUDGE_PROMPT:
        raise ValueError('The accepted Sol marker prompt changed')
    result = {}
    for case in read_jsonl(old / 'marker_cases.jsonl'):
        if case['setting'] != 'current' or case['cohort'] != 'real' or not case['completed']:
            continue
        path = old / 'marker_judgments' / (case['case_id'].replace(':', '_') + '.json')
        if not path.exists():
            continue
        saved = read_json(path)
        if saved['status'] != 'completed':
            continue
        messages, schema = contract(case)
        key = sha_text(json.dumps([messages, schema], ensure_ascii=False, sort_keys=True))
        result.setdefault(key, {'judgment': saved['judgment'], 'reused': True, 'path': str(path),
                                'sha256': file_sha(path), 'contract_sha256': key, 'status': 'completed'})
    return result


def run(config, *, max_api_calls):
    root = config['paths'][PHASE + '_output'] / 'markers'
    with collection_lock(root):
        load_environment(config)
        _, cases = extract(config)
        api = BulkAPI(config, max_api_calls, phase=PHASE + '_sol')
        api.settle_interrupted()
        prior = baseline_reuse(config)
        grouped = defaultdict(list)
        unique = set()
        for case in cases:
            key = case['contract_sha256']
            if not case['completed'] or key in unique:
                continue
            unique.add(key)
            path = root / 'judgments' / (key + '.json')
            if path.exists():
                continue
            if key in prior:
                atomic_new(path, prior[key])
            else:
                grouped[case['condition']].append(case)
        # Interleave conditions; a cap stop must not simply omit the last adapter.
        tasks = [grouped[c][i] for i in range(max(map(len, grouped.values()), default=0))
                 for c in CONDITIONS if i < len(grouped[c])]
        errors, stopped = [], False
        def one(case):
            key = case['contract_sha256']
            path = root / 'judgments' / (key + '.json')
            messages, schema = contract(case)
            response = api.request(messages, stage='sft_real_marker_fit', item_id=key,
                                   effort='high', max_output=4096, schema=schema)
            value = json.loads(response['raw'])
            if set(value) != set(case['fields']) or any(type(v['still_fits']) is not bool for v in value.values()):
                raise ValueError('Invalid marker verdict')
            atomic_new(path, {'contract_sha256': key, 'status': 'completed', 'reused': False,
                              'judgment': value, 'phase_call': response['phase_call']})
            print(json.dumps({'marker': case['case_id'], 'usd': api.accounting()['confirmed_usd']}), flush=True)
        try:
            iterator = iter(tasks)
            with ThreadPoolExecutor(max_workers=4) as pool:
                active = {pool.submit(one, c): c for c in [next(iterator, None) for _ in range(4)] if c}
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        case = active.pop(future)
                        try:
                            future.result()
                        except Exception as exc:
                            errors.append({'case_id': case['case_id'], 'error': type(exc).__name__ + ': ' + str(exc)})
                            if isinstance(exc, CallBudgetExceeded):
                                stopped = True
                        following = None if stopped else next(iterator, None)
                        if following:
                            active[pool.submit(one, following)] = following
        finally:
            api.close()
            write_json(root / 'status.json', {'new_unique_requests_planned': len(tasks), 'errors': errors,
                'stop_reason': 'budget_cap' if stopped else 'all_requested_attempted', 'budget': api.accounting()})


def summary(config):
    actions, cases = extract(config)
    root = config['paths'][PHASE + '_output'] / 'markers'
    judgments = {}
    for case in cases:
        path = root / 'judgments' / (case['contract_sha256'] + '.json')
        if path.exists() and read_json(path)['status'] == 'completed':
            judgments[case['case_id']] = read_json(path)['judgment']
    details = []
    for action in actions:
        judged = [i for i in action['case_ids'] if i in judgments]
        bad = [i for i in judged if any(not v['still_fits'] for v in judgments[i].values())]
        unknown = not action['completed'] or len(judged) < len(action['case_ids'])
        details.append({**action, 'bad_cases': bad, 'unknown': unknown})
    rates = {}
    for condition in CONDITIONS:
        selected = [a for a in details if a['condition'] == condition]
        n = len(selected)
        bad = sum(bool(a['bad_cases']) for a in selected)
        unknown = sum(a['unknown'] and not a['bad_cases'] for a in selected)
        rates[condition] = {'structural_actions': n, 'flagged_actions': bad, 'unknown_actions': unknown,
            'share': bad / n if n and not unknown else None,
            'lower_bound': bad / n if n else None, 'upper_bound': (bad + unknown) / n if n else None,
            'cases': sum(len(a['case_ids']) for a in selected),
            'deleted_targets': sum(len(a['deleted_targets']) for a in selected),
            'eligible_cases': sum(c['condition'] == condition and c['completed'] for c in cases),
            'judged_cases': sum(c['condition'] == condition and c['case_id'] in judgments for c in cases)}
    return {'rates': rates, 'actions': details, 'judged_cases': len(judgments)}
