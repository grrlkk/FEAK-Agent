"""V5_CHECK part 0: frozen judge validation, before any downstream experiment.

Only saved dev100 / C artifacts are read. This module never runs a policy,
an analyzer, a scorer, or training, and never reads the test cohort.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import random
import re
import time

from .common import ROOT as PIPE_ROOT, REPO, atomic_new, read_json, ref, sha_text, verified, write_json
from .detector import changes, phi
from .judge import JUDGE_PROMPT, paragraphs, validate
from verak.v4.content_env import dumps
from verak.v4.scale2_projection import restore_document

ROOT = REPO/'verak/v5/outputs/check'
REPORT = REPO/'reports/V5_CHECK.md'
OPS = ('L_CONN', 'L_CONJ', 'L_REGISTER', 'L_SUBJ_INSERT')
KINDS = dict(zip(OPS, ('ec_category', 'conjunction_category', 'style_vs_majority', 'subject_omission')))
SEED = 173
CONTEXTS = ('local', 'full')
NOTICE_KEYS = ('id', 'sid', 'type', 'before', 'after', 'scope', 'ec_index_before', 'ec_index_after')


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen V5_CHECK input differs: '+str(path))
    else:
        atomic_new(path, value)


def local_paragraphs(items, sid):
    """Exactly target + its positional predecessor in this snapshot, no lookahead."""
    flat = [s for p in items for s in p['sentences']]
    indexes = [i for i, s in enumerate(flat) if s['id'] == sid]
    if len(indexes) != 1:
        raise ValueError('Target missing or ambiguous in validation snapshot')
    i = indexes[0]
    keep = {s['id'] for s in flat[max(0, i-1):i+1]}
    return [{'id': p['id'], 'sentences': [deepcopy(s) for s in p['sentences'] if s['id'] in keep]}
            for p in items if any(s['id'] in keep for s in p['sentences'])]


def make_view(original, revised, question, notice, context):
    if context not in CONTEXTS:
        raise ValueError('Unknown validation context')
    a, b = deepcopy(original), deepcopy(revised)
    n = {k: deepcopy(notice[k]) for k in NOTICE_KEYS if k in notice}
    n['id'] = 'N1'
    if context == 'local':
        a, b = local_paragraphs(a, n['sid']), local_paragraphs(b, n['sid'])
        # Scope is still the same factual notice; no hidden text is supplied.
    return {'question': question, 'original': a, 'revised': b, 'notices': [n],
            'variants': [{'id': 'V1', 'paragraphs': deepcopy(b)}]}


def diversified_sample(candidates, n, rng, *, used=None):
    """Randomized source-first selection, without dropping any difficult label."""
    used = used if used is not None else set()
    groups = defaultdict(list)
    for item in candidates:
        if item['unique_key'] not in used:
            groups[item['source_id']].append(item)
    names = sorted(groups)
    rng.shuffle(names)
    for values in groups.values():
        values.sort(key=lambda x: x['unique_key'])
        rng.shuffle(values)
    selected = []
    while len(selected) < n:
        changed = False
        for source in names:
            if not groups[source]:
                continue
            item = groups[source].pop()
            if item['unique_key'] in used:
                continue
            selected.append(item)
            used.add(item['unique_key'])
            changed = True
            if len(selected) == n:
                break
        if not changed:
            raise ValueError(f'Insufficient controls/records: {len(selected)}/{n}')
    return selected


def control_notice(op, sid, feature):
    if op == 'L_CONN':
        value = feature['ec_categories'][0]
        extra = {'ec_index_before': 0, 'ec_index_after': 0}
    elif op == 'L_CONJ':
        value, extra = feature['conjunction_category'], {}
    elif op == 'L_REGISTER':
        value = {k: feature[k] for k in ('style', 'majority_style', 'off_style')}
        extra = {}
    else:
        value, extra = feature['subject_omitted'], {}
    return {'id': 'N1', 'sid': sid, 'type': KINDS[op], 'before': deepcopy(value),
            'after': deepcopy(value), 'scope': [sid, feature['antecedent']['sid']], **extra}


def prepare(*, root=ROOT):
    root = Path(root)
    started = time.monotonic()
    manifest = verified(ref(PIPE_ROOT/'A/manifest.json'))
    positives, controls, essays = defaultdict(list), defaultdict(list), {}
    for case in manifest['cases']:
        prepared = verified(case['prepared'])
        source, corrupted = restore_document(prepared['raw_source']), restore_document(prepared['raw_corrupted'])
        ap, bp = phi(source), phi(corrupted)
        before, after = {u.sid: u.text for u in source.units}, {u.sid: u.text for u in corrupted.units}
        notices = changes(source, corrupted)
        touched = {s for r in prepared['corpus']['records'] for s in r['sids']}
        source_id = prepared['corpus']['source_id']
        essays[case['case_id']] = {'original': paragraphs(prepared['raw_source']),
            'revised': paragraphs(prepared['raw_corrupted']), 'question': prepared['corpus']['question']}
        base = {'case_id': case['case_id'], 'source_id': source_id,
                'genre': prepared['corpus']['genre'], 'prepared': case['prepared']}
        for record in prepared['corpus']['records']:
            op = record['op']
            if op not in OPS:
                continue
            matching = [n for n in notices if n['type'] == KINDS[op] and n['sid'] in record['sids']]
            if len(matching) != 1 or len(record['sids']) != 1:
                raise ValueError('Expected exactly one saved form-change notice per record: '+record['record_id'])
            positives[op].append({**base, 'operator': op, 'label': 'known_broken_record',
                'record_id': record['record_id'], 'unique_key': record['record_id'],
                'notice': matching[0]})
        for sid, feature in bp.items():
            pred = feature['antecedent']['sid']
            # Unchanged target, predecessor, every phi component and essay style;
            # neither sentence is referenced by a corruption record.
            if sid in touched or sid not in ap or before[sid] != after[sid] or feature != ap[sid]:
                continue
            if pred is None or pred in touched or before.get(pred) != after.get(pred):
                continue
            if not (feature['ec_categories'] or feature['conjunction_category']):
                continue
            eligible = {'L_CONN': bool(feature['ec_categories']), 'L_CONJ': bool(feature['conjunction_category']),
                'L_REGISTER': feature['style'] not in {None, 'unknown', 'mixed'} and not feature['off_style'],
                'L_SUBJ_INSERT': feature['subject_omitted'] and not feature['omission_uncertain']}
            for op, ok in eligible.items():
                if ok:
                    # Same source sentence in another corruption variant is not a new control.
                    controls[op].append({**base, 'operator': op, 'label': 'known_fine_unchanged',
                        'record_id': None, 'unique_key': source_id+':'+sid,
                        'notice': control_notice(op, sid, feature)})
    rng = random.Random(SEED)
    selected = []
    for op in OPS:
        selected += diversified_sample(positives[op], 15, rng)
    used = set()
    # Rare omitted-subject controls first; all other quotas remain fixed at 15.
    for op in ('L_SUBJ_INSERT', 'L_CONJ', 'L_CONN', 'L_REGISTER'):
        selected += diversified_sample(controls[op], 15, rng, used=used)
    rng.shuffle(selected)
    for i, item in enumerate(selected, 1):
        item['probe_id'] = f'J{i:03d}'
        item['inputs'] = {}
        for context in CONTEXTS:
            value = make_view(**essays[item['case_id']], notice=item['notice'], context=context)
            messages = [{'role': 'system', 'content': JUDGE_PROMPT}, {'role': 'user', 'content': dumps(value)}]
            path = root/'validation/inputs'/f'{item["probe_id"]}_{context}.json'
            freeze(path, {'messages': messages})
            item['inputs'][context] = ref(path)
    value = {'seed': SEED, 'dev100_manifest': ref(PIPE_ROOT/'A/manifest.json'),
        'prompt_sha256': sha_text(JUDGE_PROMPT), 'contexts': list(CONTEXTS),
        'positives_per_operator': 15, 'controls_per_operator': 15,
        'positive_candidates': {k: len(v) for k, v in positives.items()},
        'control_candidates': {k: len(v) for k, v in controls.items()},
        'sampling': 'random source-first within operator; no judge-dependent filtering; controls unique by source sentence',
        'known_fine_limitation': 'unchanged marker and predecessor, not independent human certification of every original sentence',
        'threshold': .8, 'uncertain': 'not detected; shown separately',
        'gate': 'prefer full if recall>=.8; else local if recall>=.8; else STOP after part0',
        'variant': 'one blinded V1 equal to revised; no repair has been run',
        'training': False, 'test_cohort_opened': False, 'probes': selected}
    freeze(root/'validation/sample.json', value)
    prior = read_json(PIPE_ROOT/'api_contract.json')
    contract = {'user_approved': True, 'scope': 'V5_CHECK only; no training; no test cohort',
        'total_cap_usd': 11., 'luna': {**prior['luna'], 'cap_usd': 3.},
        'claude': {**prior['claude'], 'cap_usd': 8.},
        'authorization': 'User V5_CHECK: Claude $8, Luna $3; validation before downstream parts',
        'settings_source': ref(PIPE_ROOT/'api_contract.json')}
    for kind in ('luna', 'claude'):
        contract[kind].pop('expected_usd', None)
    freeze(root/'api_contract.json', contract)
    csv_ref = export_c_review(root=root)
    examples = export_c_inputs(root=root)
    marker = root/'validation/prepared.json'
    if not marker.exists():
        atomic_new(marker, {'at': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': time.monotonic()-started, 'sample': ref(root/'validation/sample.json'),
            'review_csv': csv_ref, 'example_inputs': examples})
    return value


def export_c_inputs(*, root=ROOT):
    chosen = []
    for family in ('onepass', 'luna'):
        options = []
        for path in sorted((PIPE_ROOT/'C').glob(f'*/*/judge_{family}.json')):
            value = read_json(path)
            if not value['valid'] or value['messages'][0]['content'] != JUDGE_PROMPT:
                raise ValueError('C judge prompt changed or judgment invalid')
            payload = json.loads(value['messages'][1]['content'])
            if payload['notices']:
                options.append((len(value['messages'][1]['content']), str(path), payload))
        _, path, payload = min(options)
        target = root/'validation'/f'C_example_{family}.json'
        freeze(target, {'original_judge': ref(Path(path)), 'input': payload})
        chosen.append(ref(target))
    return chosen


def export_c_review(*, root=ROOT):
    population = []
    for path in sorted((PIPE_ROOT/'C').glob('*/*/judge_*.json')):
        value = read_json(path)
        if not value['valid']:
            raise ValueError('Cannot silently omit an unjudged C case')
        payload = json.loads(value['messages'][1]['content'])
        for notice in payload['notices']:
            population.append((path, notice['id']))
    sampled = random.Random(SEED+1).sample(population, 30)
    rows = []
    for path, notice_id in sampled:
        value = read_json(path)
        payload = json.loads(value['messages'][1]['content'])
        notice = next(n for n in payload['notices'] if n['id'] == notice_id)
        verdict = next(n for n in value['verdict']['notices'] if n['id'] == notice_id)
        sid = notice['sid']
        def local(items):
            return local_paragraphs(items, sid) if any(s['id'] == sid for p in items for s in p['sentences']) else []
        row = {'cohort': path.parent.parent.name, 'case_id': path.parent.name,
            'family': path.stem.removeprefix('judge_'), 'notice_id': notice_id, 'type': notice['type'],
            'sentence_id': sid, 'question': payload['question'], 'notice_json': dumps(notice),
            'before_context': dumps(local(payload['original'])), 'after_context': dumps(local(payload['revised'])),
            'original_full_context': dumps(payload['original']), 'revised_full_context': dumps(payload['revised']),
            'broken_after_revision': verdict['broken_after_revision'], 'reason': verdict['reason'],
            'variants_json': dumps(payload['variants']), 'variant_key_json': dumps(value['variant_key']),
            'variant_verdicts_json': dumps([{**{k:v[k] for k in v if k != 'notices'},
                'notice': next(n for n in v['notices'] if n['id'] == notice_id)} for v in value['verdict']['variants']]),
            'judge_path': str(path), 'judge_sha256': ref(path)['sha256']}
        rows.append(row)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    path = root/'validation/C_notices_human_review_30.csv'
    content = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content.encode('utf-8-sig'):
            raise ValueError('Frozen human-review sample differs')
    else:
        path.write_bytes(content.encode('utf-8-sig'))
    freeze(root/'validation/C_review_sample.json', {'seed': SEED+1, 'population': len(population),
        'sample_size': 30, 'sampling_unit': 'unique (C case, revision family, notice); no verdict stratification',
        'csv': ref(path)})
    return ref(path)


def gate_from_results(rows, *, expected_per_context=120):
    tables = {}
    for context in CONTEXTS:
        subset = [r for r in rows if r['context'] == context]
        tables[context] = {}
        for op in (*OPS, 'ALL'):
            rs = [r for r in subset if op == 'ALL' or r['operator'] == op]
            row = {}
            for label, name in [('known_broken_record', 'positive'), ('known_fine_unchanged', 'control')]:
                group = [r for r in rs if r['label'] == label]
                valid = [r for r in group if r['valid']]
                row[name+'_n'] = len(group)
                row[name+'_judged'] = len(valid)
                for answer in ('yes', 'no', 'uncertain'):
                    row[name+'_'+answer] = sum(r.get('answer') == answer for r in valid)
            row['recall'] = row['positive_yes']/row['positive_n'] if row['positive_n'] and row['positive_judged'] == row['positive_n'] else None
            row['false_positive_rate'] = row['control_yes']/row['control_judged'] if row['control_judged'] else None
            missing = row['control_n']-row['control_judged']
            row['control_unjudged'] = missing
            row['false_positive_bounds'] = [row['control_yes']/row['control_n'],
                (row['control_yes']+missing)/row['control_n']] if row['control_n'] else None
            tables[context][op] = row
    complete = all(sum(r['context'] == c and r['valid'] for r in rows) == expected_per_context for c in CONTEXTS)
    setting = None
    # The user's gate is recall, not ancillary variant-format compliance or FPR.
    # It needs all positive labels in the selected setting; invalid controls
    # remain explicitly missing and widen the reported FPR bounds.
    full_recall, local_recall = (tables[c]['ALL']['recall'] for c in ('full', 'local'))
    expected_positive = expected_per_context//2
    for context, recall in [('full', full_recall), ('local', local_recall)]:
        if tables[context]['ALL']['positive_judged'] == expected_positive and recall is not None and recall >= .8:
            setting = context
            break
    both_measured = full_recall is not None and local_recall is not None
    return {'tables': tables, 'all_judgments_valid': complete, 'validated_setting': setting,
        'continue': setting is not None,
        'stop_reason': None if setting else 'both recalls below .8' if both_measured else 'incomplete positive validation; no downstream dispatch',
        'C_underestimation_warning': setting == 'local'}


def judge_one(item, context, *, root=ROOT, format_retry=False):
    from .api import PaidAPI, BudgetStop
    folder = 'validation/format_retry/judgments' if format_retry else 'validation/judgments'
    path = root/folder/f'{item["probe_id"]}_{context}.json'
    if path.exists():
        return read_json(path)
    messages = verified(item['inputs'][context])['messages']
    if messages[0]['content'] != JUDGE_PROMPT:
        raise ValueError('Validation prompt differs from C')
    result = {'probe_id': item['probe_id'], 'context': context, 'operator': item['operator'],
        'label': item['label'], 'input': item['inputs'][context], 'valid': False}
    api = PaidAPI('claude', root=root)
    try:
        identity = 'V5_CHECK:0:'+item['probe_id']+':'+context+(':format_retry1' if format_retry else '')
        response = api.request(messages, item_id=identity, max_output=4096)
        result['response'] = response
        result['verdict'] = validate(response['raw'], ['N1'], ['V1'])
        result['answer'] = result['verdict']['notices'][0]['broken_after_revision']
        result['valid'] = True
    except BudgetStop:
        raise
    except Exception as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    finally:
        api.close()
    atomic_new(path, result)
    return result


def retry_formats(*, root=ROOT):
    """One identical-input retry of invalid JSON/schema, never of a valid verdict.

    First-pass results remain immutable. The retry list is frozen before dispatch.
    Transport errors and unknown paid outcomes are never retried by this function.
    """
    from .api import PaidAPI
    root = Path(root)
    marker = root/'validation/format_retry/complete.json'
    if marker.exists():
        return read_json(marker)
    first = read_json(root/'validation/complete.json')
    rows = read_json(root/'validation/result_rows.json')
    pending = [r for r in rows if not r['valid'] and r.get('response', {}).get('status') == 'completed'
               and r.get('error', '').startswith(('ValueError:', 'JSONDecodeError:'))]
    freeze(root/'validation/format_retry/contract.json', {
        'first_pass': ref(root/'validation/complete.json'),
        'retry_only': [[r['probe_id'], r['context']] for r in pending],
        'prompt_and_messages': 'exact same bytes as first-pass messages; no hints or formatting reminder',
        'max_retries': 1, 'new_sampling': 'independent provider-default request',
        'valid_no_or_uncertain_verdicts_retried': False,
        'initial_format_failures_stay_reported': True})
    sample = {p['probe_id']: p for p in read_json(root/'validation/sample.json')['probes']}
    started = time.monotonic()
    updates = {}
    for old in pending:
        value = judge_one(sample[old['probe_id']], old['context'], root=root, format_retry=True)
        updates[(old['probe_id'], old['context'])] = value
        print('format_retry', old['probe_id'], old['context'], value.get('answer', 'ERROR'), flush=True)
    merged = [updates.get((r['probe_id'], r['context']), r) for r in rows]
    api = PaidAPI('claude', root=root); api.close()
    seconds = time.monotonic()-started
    result = {**gate_from_results(merged), 'at': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': first['wall_seconds']+seconds, 'format_retry_wall_seconds': seconds,
        'first_pass_wall_seconds': first['wall_seconds'], 'initial_format_failures': len(pending),
        'initial_format_failures_by_context': dict(Counter(r['context'] for r in pending)),
        'initial_format_failed_probes': [[r['probe_id'], r['context']] for r in pending],
        'first_pass': ref(root/'validation/complete.json'),
        'model': first['model'], 'sample': first['sample'], 'api_cost': read_json(root/'api/accounting.json')}
    atomic_new(root/'validation/format_retry/result_rows.json', merged)
    atomic_new(marker, result)
    report(root=root)
    return result


def run(*, root=ROOT, workers=4):
    from .api import PaidAPI, BudgetStop
    root = Path(root)
    marker = root/'validation/complete.json'
    if marker.exists():
        return read_json(marker)
    started = time.monotonic()
    sample = verified(ref(root/'validation/sample.json'))
    if sample['prompt_sha256'] != sha_text(JUDGE_PROMPT):
        raise ValueError('Frozen judge protocol mismatch')
    jobs = [(item, context) for item in sample['probes'] for context in CONTEXTS]
    rows, budget_stop = [], None
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        def dispatch():
            pair = next(iterator, None)
            if pair:
                pending[pool.submit(judge_one, *pair, root=root)] = pair
        for _ in range(workers):
            dispatch()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                item, context = pending.pop(future)
                try:
                    result = future.result()
                    rows.append(result)
                    print('V5_CHECK.0', len(rows), '/', len(jobs), item['probe_id'], context,
                          result.get('answer', 'ERROR'), flush=True)
                except BudgetStop as exc:
                    budget_stop = str(exc)
                if budget_stop is None:
                    dispatch()
    # Missing requests are reported as unjudged, never as negative predictions.
    existing = {(r['probe_id'], r['context']) for r in rows}
    for item, context in jobs:
        if (item['probe_id'], context) not in existing:
            rows.append({'probe_id': item['probe_id'], 'context': context, 'operator': item['operator'],
                         'label': item['label'], 'valid': False, 'error': 'not dispatched at budget cap'})
    rows.sort(key=lambda r: (r['probe_id'], r['context']))
    api = PaidAPI('claude', root=root); api.close()
    result = {**gate_from_results(rows), 'at': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.monotonic()-started, 'budget_stop': budget_stop,
        'model': read_json(root/'api_contract.json')['claude']['model'],
        'sample': ref(root/'validation/sample.json'), 'api_cost': read_json(root/'api/accounting.json')}
    atomic_new(root/'validation/result_rows.json', rows)
    atomic_new(marker, result)
    report(root=root)
    return result


def report(*, root=ROOT, report_path=REPORT):
    sample = read_json(root/'validation/sample.json')
    prepared = read_json(root/'validation/prepared.json')
    path = root/'validation/final.json'
    if not path.exists():
        path = root/'validation/format_retry/complete.json'
        if not path.exists():
            path = root/'validation/complete.json'
    outcome = read_json(path) if path.exists() else None
    text = ['# V5_CHECK', '', 'No training; the test cohort is untouched. API caps: Claude $8, Luna $3.',
        'Frozen C judge: `claude-sonnet-5-5`, adaptive thinking / medium effort, provider-default sampling. '
        'The exact C system prompt is unchanged. GPU experiments and new Luna calls are gated on part 0.', '',
        '## 0. Judge validation', '',
        f'Preparation wall-clock: {prepared["wall_seconds"]:.1f} s. '
        + (f'Paid validation wall-clock: {outcome["wall_seconds"]:.1f} s.' if outcome else 'Paid validation: pending.'), '',
        f'Seed {SEED}: 15 damaged records per operator (60 total) and 15 unchanged controls per operator (60 total). '
        'Sampling prioritizes distinct source essays within each operator; it never uses judge verdicts. '
        'All inputs come from the frozen dev100. No corruption is regenerated.', '',
        'The original C judge already received full essays. As explicitly confirmed by the user, '
        '`local` keeps exactly the target and its positional predecessor in each snapshot; '
        '`full` keeps the complete original and corrupted essay as in C. The question, notice, system prompt, '
        'and output schema are identical. Each validation input contains one notice and one blinded V1 '
        'equal to the corrupted/unchanged revision; no repair is performed.', '',
        'Positive labels are the existing instance-QC-passing corruption records. Controls retain identical '
        'target text, predecessor text, and all phi fields; neither target nor predecessor is referenced by a record. '
        'Every control has an eligible connective or conjunction. These unchanged controls are a proxy for fine links, '
        'not independently human-certified grammatical sentences. This limitation also applies to the corpus damage labels.', '',
        'The decision uses overall record recall, with `uncertain` counted as not detected and reported separately. '
        'Use full context if recall ≥ 0.8; otherwise use local if recall ≥ 0.8 and flag possible underestimation in C. '
        'If both fail, stop after part 0. The selected context must have all 60 positive judgments. '
        'Missing controls do not change recall: they remain unjudged, and the report shows both observed FPR and worst-case bounds.', '',
        '| Type | Positive records | Distinct positive sources | Controls | Distinct control sources |',
        '|---|---:|---:|---:|---:|']
    for op in OPS:
        p = [i for i in sample['probes'] if i['operator'] == op and i['label'] == 'known_broken_record']
        c = [i for i in sample['probes'] if i['operator'] == op and i['label'] == 'known_fine_unchanged']
        text.append(f'| {op} | {len(p)} | {len({i["source_id"] for i in p})} | {len(c)} | {len({i["source_id"] for i in c})} |')
    if outcome:
        if outcome.get('initial_format_failures'):
            text += ['', f'First-pass formatting failures: {outcome["initial_format_failures"]}/240 '
                f'({outcome["initial_format_failures_by_context"]}). Each received one fresh request with '
                'identical messages/settings, without a reminder or extra hint. No valid yes/no/uncertain verdict '
                'was retried. Original responses and the pre-dispatch retry list remain saved. '
                f'Retry wall-clock: {outcome["format_retry_wall_seconds"]:.1f} s. The table uses the resulting '
                'valid judgments; all attempts remain charged to the same $8 cap.']
        if outcome.get('gate_implementation_note'):
            text += ['', outcome['gate_implementation_note']]
        text += ['', '| Context | Type | Recall (yes / positive) | False positives (yes / judged control) | Uncertain positive / control | Unjudged controls; FPR bounds |',
                 '|---|---|---:|---:|---:|---:|']
        for context in CONTEXTS:
            for op, r in outcome['tables'][context].items():
                def rate(key, count, denominator):
                    return f'{r[count]}/{r[denominator]} = {r[key]:.1%}' if r[key] is not None else f'unjudged ({r[count]}/{r[denominator]})'
                text.append(f'| {context} | {op} | {rate("recall", "positive_yes", "positive_n")} | '
                    f'{rate("false_positive_rate", "control_yes", "control_judged")} | {r["positive_uncertain"]} / {r["control_uncertain"]} | '
                    f'{r["control_unjudged"]}; {r["false_positive_bounds"][0]:.1%}–{r["false_positive_bounds"][1]:.1%} |')
        text += ['', '**Decision: '+('continue with '+outcome['validated_setting']+' context.' if outcome['continue'] else
                'STOP after part 0 — '+outcome['stop_reason']+'.')+'**', '']
        if outcome['C_underestimation_warning']:
            text += ['C’s broken-link rate may be underestimated because its full-context judge failed validation; '
                     'later re-judging uses the validated local setting.', '']
        if not outcome['continue']:
            text += ['Parts 1–4 were not run, as required by the validation gate. No policy inference, new Luna call, '
                     'GPU job, training, or test-cohort access was started.', '']
        cost = outcome['api_cost']
        text += [f'Actual confirmed API cost: **${cost["confirmed_usd"]:.6f}**; reserved/unknown '
                 f'${cost["reserved_usd"]:.6f}. Requests: {cost["calls"]}.', '']
        for provider, value in cost['by_provider'].items():
            text.append(f'- {provider}: {value["calls"]} calls, ${value["confirmed_usd"]:.6f} confirmed.')
    review = read_json(root/'validation/C_review_sample.json')
    text += ['', '## C notices for human review', '',
        f'Uniform sample without replacement: 30 of {review["population"]} C notices, seed {review["seed"]}. '
        'The CSV contains before/after local and full contexts, all variant texts/verdicts, original judge paths and hashes. '
        'It contains the saved C judgments, not new judgments or new human labels.', '',
        f'[Review CSV]({review["csv"]["path"]})', '', '## Exact C judge prompt', '',
        f'SHA-256: `{sha_text(JUDGE_PROMPT)}`.', '', '```text', JUDGE_PROMPT, '```', '',
        '## Two exact saved C example inputs', '',
        'The shortest nonempty-notice input from each family is printed in full below. '
        'Only JSON whitespace is pretty-printed; field values and every sentence are unchanged. '
        'These are original C examples, not the newly constructed validation probes.']
    for reference in prepared['example_inputs']:
        value = verified(reference)
        text += ['', f'### {Path(reference["path"]).stem}', '',
                 f'Source: `{value["original_judge"]["path"]}`; SHA-256 `{value["original_judge"]["sha256"]}`.', '',
                 '```json', json.dumps(value['input'], ensure_ascii=False, indent=2), '```']
    text += ['', '## Reproducibility', '',
        f'Artifacts: `{root}`. Frozen sample: `validation/sample.json`; per-request exact messages: '
        '`validation/inputs/`; raw paid responses and durable budget ledger: `api/`; '
        'per-probe verdicts: `validation/judgments/`; authoritative gate: `validation/final.json`; '
        'immutable first pass and timing: `validation/complete.json`.', '']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    later = ''
    if report_path.exists():
        existing = report_path.read_text()
        match = re.search(r'^## [1-4]\. ', existing, re.M)
        if match:
            later = '\n'+existing[match.start():]
    report_path.write_text('\n'.join(text)+later, encoding='utf-8')


def finalize_validation(*, root=ROOT):
    path = root/'validation/final.json'
    if path.exists():
        return read_json(path)
    folder = root/'validation/format_retry' if (root/'validation/format_retry/complete.json').exists() else root/'validation'
    prior = read_json(folder/'complete.json')
    rows = read_json(folder/'result_rows.json')
    value = {**prior, **gate_from_results(rows), 'result_rows': ref(folder/'result_rows.json')}
    freeze(path, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['prepare', 'run', 'retry-formats', 'finalize', 'report'])
    args = parser.parse_args()
    if args.part == 'prepare':
        result = prepare(); report()
        print('Prepared', len(result['probes']), 'probes, two contexts, no paid calls')
    elif args.part == 'run':
        value = run()
        print(json.dumps({k: value[k] for k in ('continue', 'validated_setting', 'stop_reason', 'wall_seconds')}, ensure_ascii=False))
    elif args.part == 'retry-formats':
        value = retry_formats()
        print(json.dumps({k: value[k] for k in ('continue', 'validated_setting', 'stop_reason', 'wall_seconds')}, ensure_ascii=False))
    elif args.part == 'finalize':
        value = finalize_validation()
        report()
        print(json.dumps({k: value[k] for k in ('continue', 'validated_setting', 'stop_reason', 'wall_seconds')}, ensure_ascii=False))
    else:
        report()
