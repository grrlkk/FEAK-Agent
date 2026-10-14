"""Paired, stratified bootstrap summaries; no architecture selection."""
from collections import Counter
from statistics import mean, stdev
import numpy as np

from ..common import read_json, write_json
from ..phase2 import write_jsonl
from .design import prepare
from .measurement import normalize_saved_measurement

LEVELS = ('GLOBAL', 'WORD', 'SENTENCE', 'TEXT')


def distribution(values):
    values = list(values)
    return {'n': len(values), 'mean': mean(values) if values else None,
            'std': stdev(values) if len(values) > 1 else (0. if values else None)}


def bootstrap_difference(left, right, strata, *, seed=61, samples=10000):
    """Resample pairs together within each L1-L4 stratum; difference = left-right."""
    if len(left) != len(right) or len(left) != len(strata):
        raise ValueError('Paired arrays must have identical lengths')
    if not left:
        return {'n': 0, 'difference': None, 'ci95': None}
    diff, groups = np.asarray(left)-np.asarray(right), np.asarray(strata)
    rng = np.random.default_rng(seed)
    totals = np.zeros(samples)
    for name in sorted(set(strata)):
        values = diff[groups == name]
        totals += rng.choice(values, size=(samples, len(values)), replace=True).sum(axis=1)
    estimates = totals/len(diff)
    return {'n': len(diff), 'left_mean': float(np.mean(left)), 'right_mean': float(np.mean(right)),
            'difference': float(diff.mean()), 'ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'bootstrap_samples': samples, 'seed': seed, 'unit': 'paired_essay_stratified_by_L1_L4'}


def get_metric(row, field):
    if field == 'steps':
        return sum(row['steps'].values())
    if field == 'cost':
        return row['cost_usd']
    if field.startswith('recovery_'):
        level = field[len('recovery_'):]
        values = [r['recovery'] for r in row['reward']['combined']['per_record'] if r['level'] == level]
        return mean(values) if values else None
    if field.startswith('rec_'):
        level = field[4:]
        records = row['reward']['combined']['per_record']
        # Per-level main is undefined on essays without a record at that level.
        values = [r['main'] for r in records if r['level'] == level]
        return mean(values) if values else None
    role, metric = field.split('.')
    values = (row.get('reward') or {}).get(role)
    return values[metric] if values is not None else None


def paired(left, right, fields, config):
    left = {r['corpus_episode_id']: r for r in left if r.get('completed') and r.get('reward')}
    right = {r['corpus_episode_id']: r for r in right if r.get('completed') and r.get('reward')}
    result = {}
    for field in fields:
        values = [(get_metric(left[k], field), get_metric(right[k], field), left[k]['level'])
                  for k in sorted(left.keys() & right.keys())]
        values = [(a, b, s) for a, b, s in values if a is not None and b is not None]
        result[field] = bootstrap_difference([v[0] for v in values], [v[1] for v in values],
            [v[2] for v in values], seed=config['phase6']['bootstrap_seed'], samples=config['phase6']['bootstrap_samples'])
    return result


def condition_stats(rows):
    complete = [r for r in rows if r.get('completed')]
    rewarded = [r for r in complete if r.get('reward')]
    actions = [a for r in rows for role in r.get('actions_by_role', {}).values() for a in role]
    roles = sorted({role for row in rows for role in row.get('steps', {})})
    stats = {'n': len(rows), 'completed': len(complete), 'completion_rate': len(complete)/max(1, len(rows)),
        'reward': {role: {k: distribution(r['reward'][role][k] for r in rewarded if r['reward'][role] is not None)
                         for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}
                   for role in ('global', 'korean', 'combined')},
        'by_record_level': {level: distribution(rec['main'] for r in rewarded
                         for rec in r['reward']['combined']['per_record'] if rec['level'] == level) for level in LEVELS},
        'recovery_by_record_level': {level: distribution(rec['recovery'] for r in rewarded
                         for rec in r['reward']['combined']['per_record'] if rec['level'] == level) for level in LEVELS},
        'by_operator': {op: distribution(rec['recovery'] for r in rewarded
                         for rec in r['reward']['combined']['per_record'] if rec['op'] == op)
                         for op in sorted({rec['op'] for r in rewarded for rec in r['reward']['combined']['per_record']})},
        'cost': distribution(r['cost_usd'] for r in rows),
        'total_cost': sum(r['cost_usd'] for r in rows),
        'steps': distribution(sum(r['steps'].values()) for r in rows),
        'elapsed_s': distribution(r['elapsed_s'] for r in rows),
        'roles': {role: {'steps': distribution(r['steps'].get(role, 0) for r in rows),
                        'checks': distribution(r['checks'].get(role, 0) for r in rows),
                        'check_exactly_once': sum(r['checks'].get(role) == 1 for r in rows),
                        'terminations': dict(Counter(r.get('termination', {}).get(role, 'not_applicable') for r in rows))}
                  for role in roles},
        'invalid_actions': sum(not a.get('valid', True) for a in actions), 'action_count': len(actions),
        'invalid_action_rate': sum(not a.get('valid', True) for a in actions)/max(1, len(actions)),
        'role_violations': sum(a.get('error_code') == 'role_forbidden' for a in actions),
        'errors': [{'id': r['corpus_episode_id'], 'error': r.get('runtime_error'),
                    'measurement_error': r.get('measurement_error')} for r in rows
                   if r.get('runtime_error') or r.get('measurement_error')]}
    tokens = Counter()
    for r in rows:
        usage = r.get('usage') or r.get('generation', {}).get('cost', {})
        tokens.update({k: usage.get(k, 0) for k in ('input', 'output', 'reasoning', 'cache_read', 'cache_write')})
    stats['tokens'] = dict(tokens)
    stats['cached_share'] = tokens['cache_read']/max(1, tokens['input'])
    measurement = [r['measurement'] for r in complete if 'measurement' in r]
    stats['measurement'] = {k: distribution(r[k] for r in measurement) for k in
        ('changed_share', 'text_changed_share', 'rewritten', 'deleted', 'inserted', 'moved', 'off_style_before', 'off_style_after')}
    stats['cohesion_counts'] = {k: sum(r['cohesion_counts'][k] for r in measurement)
        for k in ('polarity', 'modality', 'conjunction_relation', 'explicit_subject_inserted')}
    stats['dominant_style_changed_essays'] = sum(r['dominant_style_before'] != r['dominant_style_after'] for r in measurement)
    stats['action_types'] = dict(sum((Counter(r['action_counts']) for r in measurement), Counter()))
    facts = [f for a in actions if a.get('valid') and a.get('action') in {'EDIT', 'MOVE'}
             for f in a.get('cohesion_changes', [])]
    stats['per_action_cohesion_notices'] = dict(Counter(f['type'] for f in facts))
    stats['per_action_notice_interpretation'] = 'New observations at each action; not inherited baseline labels, not semantic errors.'
    stats['delta_q'] = distribution(r['reward']['combined']['quality']['delta'] for r in rewarded)
    aligned = [r['alignment'] for r in complete if 'alignment' in r]
    stats['baseline_alignment'] = {'matched': sum(a['matched'] for a in aligned),
        'new_units': sum(a['new_units'] for a in aligned),
        'unmatched_input': sum(a['unmatched_input'] for a in aligned),
        'matched_below_0_6': sum(s['lexical_similarity'] is not None and s['lexical_similarity'] < .6
                               for a in aligned for s in a['sentences'])}
    return stats


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def fmt(v):
    return 'NA' if v is None else f'{v:.4f}'


def summarize_phase6(config, api):
    design, _, _ = prepare(config)
    output = config['paths']['phase6_output']
    conditions = {}
    loaded = {}
    for name in ('two_stage', 'single', 'check_once', 'real', 'rewrite_sol', 'rewrite_kanana'):
        rows = [normalize_saved_measurement(read_json(path)) for path in sorted((output/name/'episodes').glob('*.json'))]
        loaded[name] = rows
        stats = condition_stats(rows)
        stats['by_level'] = {level: condition_stats([r for r in rows if r.get('level') == level])
                             for level in ('L1', 'L2', 'L3', 'L4')}
        stats['by_genre'] = {g: condition_stats([r for r in rows if r['genre'] == g])
                            for g in sorted({r['genre'] for r in rows})}
        conditions[name] = stats
        write_jsonl(output/name/'episodes.jsonl', rows)
        write_json(output/name/'metrics.json', stats)
        (output/name/'report.md').write_text(f'# Phase 6: {name}\n\n'+table(
            ['Metric', 'Value'], [['Completed', f"{stats['completed']}/{stats['n']}"],
            ['R', fmt(stats['reward']['combined']['R']['mean'])],
            ['R_rec', fmt(stats['reward']['combined']['R_rec']['mean'])],
            ['R_over', fmt(stats['reward']['combined']['R_over']['mean'])],
            ['Cost USD', fmt(stats['total_cost'])]])+'\n\nDetails: metrics.json; measurements are structural observations, not semantic error judgments.\n', encoding='utf-8')
    basic = ['combined.R', 'combined.R_rec', 'combined.R_over', 'steps', 'cost']+[
        prefix+k for prefix in ('recovery_', 'rec_') for k in LEVELS]
    comparison = paired(loaded['two_stage'], loaded['single'], basic, config)
    check_fields = ['combined.R', 'combined.R_rec', 'combined.R_q', 'combined.R_over', 'global.R', 'global.R_rec',
                    'global.R_q', 'global.R_over', 'korean.R', 'korean.R_rec', 'korean.R_q', 'korean.R_over', 'cost', 'steps']
    check_ids = set(design['check_ids'])
    check = paired(loaded['check_once'], [r for r in loaded['two_stage'] if r['corpus_episode_id'] in check_ids], check_fields, config)
    local_rows = [read_json(p) for p in sorted((output/'local_link/essays').glob('*.json'))]
    local = {'source_essays': len(local_rows), 'operators': {}}
    for op in ('L_CONN', 'L_CONJ', 'L_SUBJ_INSERT', 'L_REGISTER', 'L_POLARITY', 'L_SPACING'):
        all_rows = [r for row in local_rows for r in row['operators'] if r['op'] == op]
        good = [r for r in all_rows if r['applied']]
        local['operators'][op] = {'sources': len(all_rows), 'applicable': sum(r['proposals'] > 0 for r in all_rows),
            'verified': len(good), 'discarded_applications': sum(len(r['discarded']) for r in all_rows),
            'delta_q': distribution(r['delta_q'] for r in good),
            'negative_delta_share': sum(r['delta_q'] < 0 for r in good)/max(1, len(good)),
            'delta_rubric': [distribution(r['delta_rubric'][i] for r in good) for i in range(8)]}
    write_json(output/'local_link/metrics.json', local)
    (output/'local_link/report.md').write_text('# Phase 6: local cohesion and scorer quality\n\n'+table(
        ['Operator', 'Verified / 100', 'Mean ΔQ', 'SD'], [[op, v['verified'], fmt(v['delta_q']['mean']), fmt(v['delta_q']['std'])]
        for op, v in local['operators'].items()])+'\n\nEach applicable operator is applied separately to an unmodified source; absent patterns are not fabricated.\n', encoding='utf-8')
    judgments = [read_json(p) for p in sorted((output/'real_judgments').glob('valid_*.json'))]
    real = {'n': len(judgments), 'verification': 'LLM-verified',
        'new_content': sum(j['judgment']['new_content'] for j in judgments),
        'meaning_changed': sum(j['judgment']['meaning_changed'] for j in judgments),
        'new_content_ids': [j['source_id'] for j in judgments if j['judgment']['new_content']],
        'meaning_changed_ids': [j['source_id'] for j in judgments if j['judgment']['meaning_changed']],
        'delta_q': distribution(r['quality_measurement']['delta_q'] for r in loaded['real'] if 'quality_measurement' in r)}
    for key in ('new_content', 'meaning_changed'):
        real[key+'_rate'] = real[key]/max(1, real['n'])
    real['by_genre'] = {g: {'n': len(group),
        'new_content': sum(j['judgment']['new_content'] for j in group),
        'meaning_changed': sum(j['judgment']['meaning_changed'] for j in group)}
        for g in ('설명', '논증', '정서') for group in [[j for j in judgments if j['genre'] == g]]}
    write_json(output/'real_judgments/metrics.json', real)
    (output/'real_judgments/report.md').write_text('# Phase 6: real-essay content scope (LLM-verified)\n\n'+table(
        ['Field', 'True / judged', 'Rate'], [[k, f"{real[k]}/{real['n']}", fmt(real[k+'_rate'])]
        for k in ('new_content', 'meaning_changed')])+
        '\n\nSeparate Sol/high requests; same model family, no human-accuracy claim. Exact spans stay in local judgment JSON.\n', encoding='utf-8')
    api_rows = [read_json(p) for p in sorted((output/'api/requests').glob('*.json'))]
    with api.db() as db:
        blocked_ids = {r[0] for r in db.execute("SELECT id FROM calls WHERE status='blocked_before_send'")}
    usage = Counter()
    for row in api_rows:
        usage.update(row.get('cost', {}))
    api_stats = {**api.accounting(), 'usage': dict(usage),
        'error_types': dict(Counter(r.get('error_type') for r in api_rows if r.get('error_type'))),
        'dispatched_error_types': dict(Counter(r.get('error_type') for r in api_rows
            if r.get('error_type') and r['phase_call'] not in blocked_ids)),
        'reasoning_included_in_output': True, 'cost_is_usage_estimate_not_invoice': True,
        'rates_per_million': {'input': 2., 'output': 10., 'cache_read': .1, 'cache_write': 2.5}}
    result = {'design': design, 'filter': read_json(output/'filter/metrics.json'), 'conditions': conditions,
              'two_stage_minus_single': comparison, 'check_once_minus_default': check,
              'check_default_subset': condition_stats([r for r in loaded['two_stage'] if r['corpus_episode_id'] in check_ids]),
              'local_link': local, 'real_content': real, 'api': api_stats}
    write_json(output/'metrics.json', result)
    (output/'filter/report.md').write_text('# Phase 6: recoverability filter (LLM-verified)\n\n'+table(
        ['Split', 'Before', 'After', 'Dropped'], [[s, v['before']['essays'], v['after']['essays'], len(v['dropped_ids'])]
        for s, v in result['filter'].items()])+'\n\nEvery deletion record judged once; any false drops the entire essay, with no replacement. See metrics.json for local-level shares.\n', encoding='utf-8')
    for name, values in [('paired', comparison), ('check_comparison', check)]:
        write_json(output/name/'metrics.json', values)
        (output/name/'report.md').write_text('# Phase 6: '+name+'\n\n'+table(
            ['Metric', 'Pairs', 'Left − right', 'Bootstrap 95% CI'],
            [[k, v['n'], fmt(v['difference']), str(v['ci95'])] for k, v in values.items()])+'\n', encoding='utf-8')
    return result
