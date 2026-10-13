"""Rejudge saved structural edits with the validated, unchanged C judge."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from .common import ROOT as PIPE_ROOT, REPO, atomic_new, read_json, ref, sha_text, verified, write_json, safe_id
from .check_judge import ROOT, REPORT, JUDGE_PROMPT, freeze, local_paragraphs
from .judge import paragraphs, validate
from verak.v4.content_env import dumps
from verak.v4.render_v43 import assert_rendered

OBS = REPO/'verak/v3/outputs/observation_test'


def require_gate():
    gate = read_json(ROOT/'validation/final.json')
    if not gate['continue']:
        raise RuntimeError('Part 0 did not validate a judge; downstream work is forbidden')
    return gate['validated_setting']


def from_layout(layout):
    return [{'id': p['pid'], 'sentences': [{'id': u['sid'], 'text': u['text']} for u in p['units']]}
            for p in layout['paragraphs']]


def flat(items):
    return [dict(s, paragraph=p['id']) for p in items for s in p['sentences']]


def positions(items):
    values = flat(items)
    return {s['id']: {'paragraph': s['paragraph'], 'predecessor': values[i-1]['id'] if i else None}
            for i, s in enumerate(values)}


def affected(before, after, action):
    old, new = positions(before), positions(after)
    sites = {sid for sid, value in new.items() if sid not in old or old[sid] != value}
    for n in action.get('marker_change_notices', action.get('cohesion_changes', [])):
        if isinstance(n, dict) and n.get('sid') in new:
            sites.add(n['sid'])
    return sorted(sites)


def structural_kind(action, before, after):
    if not action.get('valid'):
        return None
    name = action.get('action')
    if name in {'MOVE', 'DELETE', 'INSERT'}:
        return name
    if name == 'EDIT' and len(flat(after)) > len(flat(before)):
        return 'SPLIT'
    return None


def next_snapshot(call):
    messages = verified(call['prefix'])['messages']
    if call['role'] == 'orchestrator':
        return json.loads(messages[1]['content'])['paragraphs']
    return assert_rendered(messages)['paragraphs']


def action_trace(result):
    """A1 call arrays append DELEGATE after workers; prefix sequence is time order."""
    calls = sorted(result['calls'], key=lambda c: int(Path(c['prefix']['path']).name.split('_')[0]))
    final = paragraphs(result['final_token_document'])
    output = []
    for index, call in enumerate(calls):
        if call['role'] != 'revision':
            continue
        before = next_snapshot(call)
        after = next_snapshot(calls[index+1]) if index+1 < len(calls) else final
        action = call.get('action', {})
        kind = structural_kind(action, before, after)
        if kind:
            output.append({'id': f'E{len(output)+1}', 'kind': kind, 'action': action,
                'before': before, 'after': after, 'sites': affected(before, after, action)})
    return output


def observation_sources():
    """The real30 arms that produced 21.7% and 8–12%; both are fully saved."""
    design = read_json(OBS/'design.json')
    marker_actions = [json.loads(line) for line in (OBS/'structural_actions.jsonl').read_text().splitlines() if line]
    grouped = defaultdict(list)
    for action in marker_actions:
        if action['cohort'] == 'real' and action['setting'] in {'current', 'text_only'}:
            grouped[(action['setting'], action['essay_id'])].append(action)
    result = []
    for arm in ('text_only', 'current'):
        for eid in design['real_ids']:
            path = OBS/arm/'episodes'/(safe_id(eid)+'.json')
            if not path.exists():
                raise FileNotFoundError('Observation arm is missing; authorized Luna fallback would be needed: '+str(path))
            episode = read_json(path)
            obs = episode['observations_by_role']['global'][0]
            question = re.search(r'^\[문항\] (.*)$', obs, re.M).group(1)
            actions = [{'id': a['action_id'], 'kind': {'sentence_insert': 'INSERT', 'sentence_delete': 'DELETE'}.get(a['kind'], a['kind']),
                        'sites': a['sites'], 'historical_case_ids': a['case_ids']} for a in grouped[(arm, eid)]]
            result.append({'source': 'observation_real30_'+arm, 'case_id': eid,
                'original': from_layout(episode['initial_layout']), 'final': from_layout(episode['final_layout']),
                'question': question, 'actions': actions, 'protocol': episode['completed'],
                'provenance': ref(path), 'lineage': {}})
    return result


def luna_sources():
    result = []
    for path in sorted((PIPE_ROOT/'C').glob('*/*/luna/edit_list.json')):
        value = read_json(path)
        actions = []
        for row in value['actions']:
            before, after = paragraphs(row['before']), paragraphs(row['after'])
            kind = structural_kind(row['action'], before, after)
            if kind:
                actions.append({'id': f'E{len(actions)+1}', 'kind': kind, 'sites': affected(before, after, row['action']),
                                'action': row['action'], 'before': before, 'after': after})
        result.append({'source': 'PIPE1_C_Luna', 'case_id': path.parent.parent.name,
            'cohort': path.parent.parent.parent.name, 'question': json.loads(value['messages'][1]['content'])['question'],
            'original': paragraphs(value['initial']), 'final': paragraphs(value['final']), 'actions': actions,
            'protocol': value['no_protocol_error'], 'provenance': ref(path), 'lineage': value['lineage']})
    if len(result) != 30:
        raise ValueError('Expected all C30 Luna outputs')
    return result


def a1_sources():
    result = []
    for case in read_json(PIPE_ROOT/'A/manifest.json')['cases']:
        row = verified(case['row'])
        path = PIPE_ROOT/'A/a1/attempts'/(safe_id(case['case_id'])+'.json')
        value = read_json(path)
        result.append({'source': 'PIPE1_A_a1_dev100', 'case_id': case['case_id'], 'question': row['question'],
            'original': row['paragraphs'], 'final': paragraphs(value['final_token_document']),
            'actions': action_trace(value), 'protocol': value['completion']['no_protocol_error'],
            'lineage': value['final'].get('lineage', {}), 'provenance': ref(path)})
    return result


def prepare_case(case):
    current = positions(case['final'])
    original = positions(case['original'])
    action_ids = defaultdict(list)
    for action in case['actions']:
        sites = set(action['sites'])
        for sid in current:
            cursor, seen = sid, set()
            while cursor in case.get('lineage', {}) and cursor not in seen:
                seen.add(cursor); cursor = case['lineage'][cursor]['parent_sid']
                if cursor in sites:
                    sites.add(sid)
        action['surviving_sites'] = sorted(sites & current.keys())
        action['absent_final_sites'] = sorted(sites-current.keys())
        for sid in action['surviving_sites']:
            action_ids[sid].append(action['id'])
    notices = [{'id': f'N{i+1}', 'sid': sid, 'type': 'structural_context',
                'before': original.get(sid), 'after': current[sid],
                'scope': [x for x in [sid, current[sid]['predecessor']] if x]}
               for i, sid in enumerate(sorted(action_ids))]
    return {**case, 'notices': notices, 'action_ids_by_sid': dict(action_ids)}


def prepare_step1():
    require_gate()
    start = time.monotonic()
    cases = [prepare_case(c) for c in observation_sources()+luna_sources()+a1_sources()]
    refs = []
    for case in cases:
        path = ROOT/'step1/cases'/case['source']/(safe_id(case['case_id'])+'.json')
        freeze(path, case); refs.append(ref(path))
    freeze(ROOT/'step1/manifest.json', {'cases': refs, 'validated_setting': require_gate(),
        'old_observation_cohort': 'the same real30 that produced the cited 21.7% / 8–12% rates; current vs text_only',
        'candidate_policy': 'all valid MOVE/INSERT/DELETE and EDIT split actions; changed adjacency/membership, inserted sites, internal notices',
        'judgment_time': 'final saved essay; incomplete episodes retained and completion-stratified',
        'unit': 'unique surviving affected sentence per essay, not one count per repeated action',
        'new_Luna_calls': 0, 'no_test_cohort': True})
    path = ROOT/'step1/prepared.json'
    if not path.exists():
        atomic_new(path, {'wall_seconds': time.monotonic()-start, 'sources': dict(Counter(c['source'] for c in cases)),
            'essays_with_candidates': sum(bool(c['notices']) for c in cases),
            'structural_actions': sum(len(c['actions']) for c in cases), 'notices': sum(len(c['notices']) for c in cases)})
    return cases


def judge_case(case, *, part='step1'):
    from .api import PaidAPI, BudgetStop
    setting = require_gate()
    base = ROOT/part/'judgments'/case['source']/safe_id(case['case_id'])
    marker = base/'result.json'
    if marker.exists():
        return read_json(marker)
    groups = [case['notices']] if setting == 'full' else [[n] for n in case['notices']]
    if not case['notices']:
        groups = []
    results = []
    api = PaidAPI('claude', root=ROOT)
    try:
        for i, notices in enumerate(groups, 1):
            a, b = case['original'], case['final']
            if setting == 'local':
                sid = notices[0]['sid']
                a = local_paragraphs(a, sid) if sid in positions(a) else []
                b = local_paragraphs(b, sid)
            payload = {'question': case['question'], 'original': a, 'revised': b,
                'notices': notices, 'variants': [{'id': 'V1', 'paragraphs': b}]}
            messages = [{'role': 'system', 'content': JUDGE_PROMPT}, {'role': 'user', 'content': dumps(payload)}]
            done = False
            for attempt in (1, 2):
                path = base/f'g{i}_try{attempt}.json'
                if path.exists():
                    result = read_json(path)
                else:
                    result = {'valid': False, 'messages': messages, 'format_retry': attempt == 2}
                    try:
                        reserve = min(16384, max(4096, 1000+450*len(notices)))
                        response = api.request(messages, item_id=f'V5_CHECK:{part}:{case["source"]}:{case["case_id"]}:g{i}:try{attempt}', max_output=reserve)
                        result['response'] = response
                        result['verdict'] = validate(response['raw'], [n['id'] for n in notices], ['V1'])
                        result['valid'] = True
                    except BudgetStop:
                        raise
                    except Exception as exc:
                        result['error'] = type(exc).__name__+': '+str(exc)
                    atomic_new(path, result)
                if result['valid']:
                    results.extend(result['verdict']['notices']); done = True; break
                if result.get('response', {}).get('status') != 'completed':
                    break
            if not done:
                results.extend({'id': n['id'], 'broken_after_revision': 'unjudged', 'reason': 'format/API failure preserved'} for n in notices)
    finally:
        api.close()
    value = {'source': case['source'], 'case_id': case['case_id'], 'verdicts': results,
             'protocol': case['protocol'], 'context': setting, 'prompt_sha256': sha_text(JUDGE_PROMPT)}
    atomic_new(marker, value)
    return value


def summarize(cases, results):
    lookup = {(r['source'], r['case_id']): r for r in results}
    output = {}
    for source in sorted({c['source'] for c in cases}):
        items = [c for c in cases if c['source'] == source]
        stats = {'essays': len(items), 'protocol_complete': sum(c['protocol'] for c in items),
            'structural_actions': 0, 'essays_with_structural_actions': 0, 'judged_essays': 0,
            'essays_with_broken_link': 0, 'broken_links': 0, 'unjudged_links': 0, 'uncertain_links': 0,
            'structural_actions_with_broken_link': 0, 'by_edit_type': {}, 'cases': []}
        types = defaultdict(Counter)
        for case in items:
            result = lookup.get((source, case['case_id']), {})
            verdicts = {r['id']: r['broken_after_revision'] for r in result.get('verdicts', [])}
            labels = {n['sid']: verdicts.get(n['id'], 'unjudged') for n in case['notices']}
            broken = {s for s, v in labels.items() if v == 'yes'}
            missing = sum(v == 'unjudged' for v in labels.values())
            stats['structural_actions'] += len(case['actions'])
            stats['essays_with_structural_actions'] += bool(case['actions'])
            stats['judged_essays'] += missing == 0
            stats['essays_with_broken_link'] += bool(broken)
            stats['broken_links'] += len(broken)
            stats['unjudged_links'] += missing
            stats['uncertain_links'] += sum(v == 'uncertain' for v in labels.values())
            for action in case['actions']:
                kind = action['kind']; sites = set(action['surviving_sites'])
                types[kind]['actions'] += 1
                types[kind]['with_broken'] += bool(sites & broken)
                types[kind]['unjudged_actions'] += any(labels.get(s) == 'unjudged' for s in sites)
                stats['structural_actions_with_broken_link'] += bool(sites & broken)
            stats['cases'].append({'case_id': case['case_id'], 'broken_sids': sorted(broken),
                'labels': labels, 'protocol': case['protocol'], 'structural_actions': len(case['actions'])})
        stats['by_edit_type'] = {k: dict(v) for k, v in types.items()}
        stats['broken_links_per_essay'] = stats['broken_links']/len(items)
        output[source] = stats
    return output


def run_judges(*, part='step1'):
    from .api import PaidAPI, BudgetStop
    require_gate()
    marker = ROOT/part/'complete.json'
    if marker.exists():
        return read_json(marker)
    started = time.monotonic()
    cases = [verified(p) for p in read_json(ROOT/part/'manifest.json')['cases']]
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(judge_case, c, part=part): c for c in cases}
        for future in as_completed(jobs):
            case = jobs[future]
            try:
                results.append(future.result())
            except BudgetStop as exc:
                errors.append({'case_id': case['case_id'], 'source': case['source'], 'error': str(exc)})
            print(part, len(results)+len(errors), '/', len(cases), case['source'], case['case_id'], flush=True)
    api = PaidAPI('claude', root=ROOT); api.close()
    metrics = summarize(cases, results)
    atomic_new(ROOT/part/'metrics.json', metrics)
    value = {'wall_seconds': time.monotonic()-started, 'errors': errors,
        'cost_cumulative': read_json(ROOT/'api/accounting.json'), 'context': require_gate(),
        'new_Luna_calls': 0, 'cases': len(cases), 'at': datetime.now(timezone.utc).isoformat()}
    atomic_new(marker, value)
    append_report(part, metrics, value)
    return value


def append_report(part, stats, complete):
    lines = [f'## {1 if part == "step1" else 2}. '+('Saved-output re-judging' if part == 'step1' else 'Local in-loop notice comparison'), '',
        f'Wall-clock for judging: {complete["wall_seconds"]:.1f} s. Validated context: **{complete["context"]}**. '
        'The C system prompt and pinned Claude settings are unchanged. No new Luna editing calls.', '',
        'Structural actions are accepted MOVE/INSERT/DELETE and EDITs that split a sentence. Targets are surviving '
        'sentences whose adjacency/paragraph membership changed, inserted sentences, or internal marker-change sites. '
        'Repeated actions pointing at the same final sentence share one link judgment. '
        'Final saved states are judged, including protocol-incomplete runs; this measures residual broken links at '
        'structurally affected positions, not causal proof that a structural edit created them. '
        'An essay with no structural candidate is assigned zero *structurally affected* broken links, not a whole-essay quality label. '
        'For the old observation test the two real30 arms are exactly those behind the cited 21.7% / 8–12% figures.', '',
        '| Source | Essays / protocol complete | Structural edits | Essays with structural edits | Essays with broken link | Broken links / essay | Unjudged / uncertain links |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for source, r in stats.items():
        lines.append(f'| {source} | {r["essays"]} / {r["protocol_complete"]} | {r["structural_actions"]} | '
            f'{r["essays_with_structural_actions"]} | {r["essays_with_broken_link"]}/{r["essays"]} '
            f'({r["essays_with_broken_link"]/r["essays"]:.1%}) | {r["broken_links_per_essay"]:.3f} | '
            f'{r["unjudged_links"]} / {r["uncertain_links"]} |')
    lines += ['', '| Source | Edit type | Actions | Actions linked to a broken final link | Unjudged actions |',
              '|---|---|---:|---:|---:|']
    for source, r in stats.items():
        for kind, values in r['by_edit_type'].items():
            lines.append(f'| {source} | {kind} | {values["actions"]} | {values["with_broken"]} | {values["unjudged_actions"]} |')
    lines += ['', 'A link may be attributable to several actions or edit types; action counts are not independent link counts. '
        'Unjudged links are never assigned `no`. Per-essay labels, provenance and completion flags are saved with the tables. '
        'One identical-input formatting retry is permitted; valid judgments are not retried.', '',
        f'Cumulative confirmed API spend: ${complete["cost_cumulative"]["confirmed_usd"]:.6f}.', '']
    with REPORT.open('a') as stream:
        stream.write('\n'+'\n'.join(lines))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['prepare', 'run', 'step2-judge'])
    args = parser.parse_args()
    if args.part == 'prepare':
        value = prepare_step1()
        print(Counter(c['source'] for c in value), 'candidates', sum(len(c['notices']) for c in value))
    else:
        run_judges(part='step2' if args.part == 'step2-judge' else 'step1')
