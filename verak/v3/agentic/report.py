"""Read-only comparisons against saved baselines and the final pilot report."""
from collections import Counter, defaultdict
from statistics import mean
import json
import re

import numpy as np

from ..common import read_json, write_json, file_sha
from ..phase2 import read_jsonl
from ..train.pilot import safe_id
from ..view_data import percentiles
from .data import PHASE, prepare
from .evaluation import episodes, marker_cases
from .export import export
from .environment import ALLOWED
from .accounting import audit_reward


def avg(values):
    values = list(values)
    return mean(values) if values else None


def ratio(n, d):
    return n / d if d else None


def deletion_summary(rows, expected, *, baseline=False):
    """Count executed whole-sentence deletions, including subsequently undone ones."""
    counts, attempts = {}, Counter()
    for row in rows:
        actions = row.get('actions_by_role', {}).get('global', []) if baseline else row.get('actions', [])
        if baseline:
            selected = [a for a in actions if a['action'] == 'EDIT' and a.get('args', {}).get('new_text') == '']
        else:
            selected = [a for a in actions if a['action'] == 'DELETE']
        valid = sum(bool(a.get('valid')) and (not baseline or
                    bool(re.fullmatch(r'(?:S|N)\d+', a.get('args', {}).get('target', '')))) for a in selected)
        counts[row.get('corpus_episode_id', row['episode_id'])] = valid
        attempts['valid'] += valid
        attempts['rejected'] += sum(not a.get('valid') for a in selected)
        attempts['valid_non_sentence'] += sum(bool(a.get('valid')) for a in selected) - valid
    completed = [counts[r.get('corpus_episode_id', r['episode_id'])] for r in rows if r.get('completed')]
    return {'definition': 'valid whole-sentence DELETE; baseline GLOBAL EDIT with an empty replacement and a single sentence ID; later UNDO does not erase an executed deletion',
            'attempted_episodes': len(rows), 'completed_episodes': len(completed), 'intended_episodes': expected,
            'valid_deletions': attempts['valid'], 'rejected_deletion_attempts': attempts['rejected'],
            'valid_non_sentence_empty_edits_excluded': attempts['valid_non_sentence'],
            'per_attempted_episode': avg(counts.values()), 'per_completed_episode': avg(completed),
            'per_intended_episode': ratio(attempts['valid'], expected),
            'episodes_with_deletions': sum(n > 0 for n in counts.values()),
            'maximum_per_episode': max(counts.values(), default=None),
            'distribution': dict(sorted(Counter(counts.values()).items())), 'by_episode': counts}


def summary(rows, expected, *, baseline=False):
    complete = [r for r in rows if r.get('completed')]
    roles = ('global', 'korean', 'combined') if baseline else ('orchestrator', 'composition', 'cohesion', 'combined')
    stats = {'expected': expected, 'attempted': len(rows), 'completed': len(complete),
             'completion': len(complete) / expected, 'rewards': {}, 'recovery': {},
             'cost': sum(r.get('confirmed_episode_cost', r.get('cost_usd', 0)) for r in rows),
             'errors': dict(Counter(r.get('runtime_error', {}).get('message', 'unknown') for r in rows if r.get('runtime_error')))}
    stats['teacher_attempted'] = sum(bool(r.get('calls')) for r in rows)
    stats['termination'] = dict(Counter('FINISH' if r.get('completed') else
        (r.get('runtime_error') or {}).get('type', r.get('termination', 'unfinished')) for r in rows))
    stats['cost_per_attempt'] = ratio(stats['cost'], len(rows))
    stats['deletions'] = deletion_summary(rows, expected, baseline=baseline)
    for role in roles:
        values = [r['reward'][role] for r in complete if r.get('reward')]
        stats['rewards'][role] = {'n': len(values), **{k: avg(v[k] for v in values if k in v)
                                                   for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}}
    ops = defaultdict(list)
    for r in complete:
        for rec in (r.get('reward') or {}).get('combined', {}).get('per_record', []):
            ops[rec['op']].append(rec)
    stats['recovery'] = {op: {'n': len(rs), 'main': avg(r['main'] for r in rs), 'coupled_weighted': avg(r['recovery'] for r in rs)}
                         for op, rs in ops.items()}
    calls = [c for r in rows for c in r.get('calls', [])]
    # New runs retain paths for every paid request, including incomplete outputs.
    # Saved baselines retain their historical aggregate usage without regeneration.
    usage = Counter()
    for row in rows:
        if baseline and row.get('usage'):
            usage.update({key + '_tokens': row['usage'].get(key, 0) for key in ('input', 'output', 'reasoning')})
        else:
            requests = [read_json(p) for p in row['api_request_paths']] if row.get('api_request_paths') else row.get('calls', [])
            for request in requests:
                value = request.get('usage') or {}
                usage.update({key: value.get(key, 0) for key in ('input_tokens', 'output_tokens')})
                usage['reasoning_tokens'] += (value.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
    stats['usage'] = dict(usage)
    stats['usage_per_attempt'] = {key: ratio(value, len(rows)) for key, value in usage.items()}
    stats['usage_note'] = 'reasoning is already included in output; baseline uses saved aggregates, new uses every recorded API attempt'
    if not baseline:
        stats['context_by_role'] = {role: {
            'lengths': percentiles([c['input_tokens'] for c in calls if c['role'] == role]) if any(c['role'] == role for c in calls) else None,
            'history_compacted_turns': sum(c.get('history_dropped', 0) > 0 for c in calls if c['role'] == role)}
            for role in roles if role != 'combined'}
    return stats


def paired(new, old, corpus, metric='R'):
    left = {r['episode_id']: r for r in new if r.get('completed') and r.get('reward')}
    right = {r.get('corpus_episode_id', r['episode_id']): r for r in old if r.get('completed') and r.get('reward')}
    ids = sorted(left.keys() & right.keys())
    if not ids:
        return {'n': 0, 'delta': None, 'ci95': None}
    values = {i: left[i]['reward']['combined'][metric] - right[i]['reward']['combined'][metric] for i in ids}
    groups = defaultdict(list)
    for i in ids:
        groups[corpus[i]['level']].append(values[i])
    rng = np.random.default_rng(89)
    boot = np.zeros(10000)
    for group in groups.values():
        boot += rng.choice(group, size=(10000, len(group)), replace=True).sum(axis=1)
    boot /= len(ids)
    return {'n': len(ids), 'ids': ids, 'delta': mean(values.values()),
            'new': avg(left[i]['reward']['combined'][metric] for i in ids),
            'baseline': avg(right[i]['reward']['combined'][metric] for i in ids),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'method': 'episode-level level-stratified bootstrap; no source/question clustering correction'}


def orchestration(rows):
    totals, tools, used, patterns = Counter(), Counter(), Counter(), Counter()
    blocked, preview_details, off_topic = [], [], []
    editor_returns = {role: Counter() for role in ('composition', 'cohesion')}
    auto_report_episodes = set()
    for row in rows:
        actions = row['actions']
        seen = set()
        delegates = []
        for i, a in enumerate(actions):
            name = a['action']
            tools[name] += 1
            seen.add(name)
            if not a['valid']:
                totals['rejected_actions'] += 1
                continue
            if name == 'DELEGATE':
                delegates.append(a['args']['agent'])
            if name == 'REPORT' and a['args']['status'] == 'blocked':
                nxt = next((x for x in actions[i+1:] if x['role'] == 'orchestrator'), None)
                control = next((x for x in actions[i+1:] if x['role'] == 'orchestrator' and x['valid'] and
                                x['action'] in {'DELEGATE', 'FINISH'}), None)
                blocked.append({'episode': row['episode_id'], 'reporting_role': a['role'], 'report': a['args'],
                                'next_orchestrator_action': {k: nxt[k] for k in ('action', 'args', 'valid')} if nxt else None,
                                'next_control_action': {k: control[k] for k in ('action', 'args')} if control else None})
            if name == 'SCORE':
                totals['score_calls'] += 1
            if name == 'FINISH':
                totals['finishes'] += 1
                prior = a['result']['last_audit']
                totals['audit_before_finish'] += prior is not None
                totals['audit_current_before_finish'] += prior is not None and prior['text_hash'] == a['before_hash']
                totals['finish_with_audit_facts'] += any(a['result']['audit_at_finish'].values())
                totals['finish_with_last_requested_audit_facts'] += bool(prior and any(prior['result'].values()))
            if name in {'MOVE', 'INSERT', 'DELETE'}:
                totals['structural_actions'] += 1
                previous = next((x for x in reversed(actions[:i]) if x['role'] == 'composition'), None)
                exact = bool(previous and previous['valid'] and previous['action'] == 'PREVIEW' and
                             previous['args']['action'] == {'action': name, 'args': a['args']} and previous['delegation'] == a['delegation'])
                totals['structural_preceded_by_exact_preview'] += exact
                matched = any(x['role'] == 'composition' and x['valid'] and x['action'] == 'PREVIEW' and
                              x['delegation'] == a['delegation'] and x['before_hash'] == a['before_hash'] and
                              x['args']['action'] == {'action': name, 'args': a['args']} for x in actions[:i])
                totals['structural_preceded_by_preview_in_same_state'] += matched
            if name == 'PREVIEW':
                nxt = next((x for x in actions[i+1:] if x['role'] == 'composition' and x['delegation'] == a['delegation']
                            and x['valid'] and x['action'] in {'MOVE', 'INSERT', 'DELETE', 'UNDO', 'REPORT'}), None)
                changed = bool(nxt and {'action': nxt['action'], 'args': nxt['args']} != a['args']['action'])
                totals['previews'] += 1
                totals['preview_followed_by_different_action'] += changed
                preview_details.append({'episode': row['episode_id'], 'preview': a['args']['action'],
                                        'next': {k: nxt[k] for k in ('action', 'args')} if nxt else None})
            if name == 'QUERY' and a['args']['target'] == 'off_topic':
                off_topic.append(row['episode_id'])
        used.update(seen)
        patterns[' -> '.join(delegates) or '(none)'] += 1
        for sequence in row.get('sequences', []):
            counts = editor_returns[sequence['role']]
            counts['delegations'] += 1
            counts['final_notice_sent'] += bool(sequence.get('final_notice_sent'))
            counts['explicit_reports'] += sequence['terminal'] == 'REPORT'
            counts['explicit_reports_after_final_notice'] += sequence['terminal'] == 'REPORT' and bool(sequence.get('final_notice_sent'))
            auto = sequence['terminal'] == 'AUTO_REPORT' and sequence.get('auto_report') is True
            counts['auto_reports'] += auto
            counts['unfinished'] += sequence['terminal'] not in {'REPORT', 'AUTO_REPORT'}
            if auto:
                auto_report_episodes.add(row['episode_id'])
            totals['editor_turns_without_report_at_budget'] += auto or (sequence['terminal'] is None and sum(
                a['role'] == sequence['role'] and a['delegation'] == sequence['delegation'] for a in actions) == 16)
    keys = ('delegations', 'final_notice_sent', 'explicit_reports', 'explicit_reports_after_final_notice', 'auto_reports', 'unfinished')
    returns = {role: {key: counts[key] for key in keys}
               for role, counts in editor_returns.items()}
    returns['all_editors'] = {key: sum(counts[key] for counts in editor_returns.values()) for key in keys}
    for counts in returns.values():
        counts['auto_report_rate_per_delegation'] = ratio(counts['auto_reports'], counts['delegations'])
        counts['auto_report_rate_after_final_notice'] = ratio(counts['auto_reports'], counts['final_notice_sent'])
        counts['explicit_report_rate_after_final_notice'] = ratio(counts['explicit_reports_after_final_notice'], counts['final_notice_sent'])
    tool_names = set().union(*ALLOWED.values())
    return {'episodes': len(rows), 'totals': dict(totals), 'tool_calls': {k: tools[k] for k in sorted(tool_names)},
            'tool_episode_use': dict(used), 'tool_episode_rates': {k: used[k] / len(rows) for k in sorted(tool_names)} if rows else {},
            'delegations_mean': avg(r.get('delegations', 0) for r in rows), 'delegation_orders': dict(patterns),
            'blocked_responses': blocked, 'preview_details': preview_details,
            'editor_returns': returns, 'auto_report_episodes': sorted(auto_report_episodes),
            'off_topic_query_episodes': sorted(set(off_topic)),
            'audit_before_finish_rate': ratio(totals['audit_before_finish'], totals['finishes']),
            'audit_current_before_finish_rate': ratio(totals['audit_current_before_finish'], totals['finishes']),
            'preview_rate': ratio(totals['structural_preceded_by_preview_in_same_state'], totals['structural_actions']),
            'immediate_preview_rate': ratio(totals['structural_preceded_by_exact_preview'], totals['structural_actions']),
            'preview_changed_rate': ratio(totals['preview_followed_by_different_action'], totals['previews'])}


def marker_metrics(config):
    actions, cases = marker_cases(config)
    root = config['paths'][PHASE + '_output']
    judgments = {r['case_id']: r['judgment'] for p in (root / 'marker_judgments').glob('*.json') for r in [read_json(p)]}
    result = {}
    for cohort in ('corrupted', 'real'):
        rows = [a for a in actions if a['cohort'] == cohort]
        bad = unknown = observed_bad = observed = 0
        for a in rows:
            flagged = any(any(not v['still_fits'] for v in judgments[c].values()) for c in a['case_ids'] if c in judgments)
            missing = not a['completed'] or any(c not in judgments for c in a['case_ids'])
            bad += flagged
            unknown += missing and not flagged
            observed += not missing
            observed_bad += not missing and flagged
        result[cohort] = {'actions': len(rows), 'bad': bad, 'unknown': unknown, 'observed': observed,
                          'surviving_candidate_sites': sum(len(a['case_ids']) for a in rows),
                          'deleted_targets': sum(len(a['deleted_targets']) for a in rows),
                          'actions_without_surviving_sites': sum(not a['case_ids'] for a in rows),
                          'lower': ratio(bad, len(rows)), 'upper': ratio(bad + unknown, len(rows)),
                          'observed_rate': ratio(observed_bad, observed)}
    result['judged_cases'] = len(judgments)
    result['eligible_cases'] = sum(c['completed'] for c in cases)
    return result


def graph_flag_summary(value):
    return {'source_flagged': value.get('source_flagged', 0),
            'source_sentences': value.get('source_sentences', 0),
            'source_false_flag_proxy': ratio(value.get('source_flagged', 0), value.get('source_sentences', 0)),
            'offtopic_flagged': value.get('offtopic_flagged', 0),
            'offtopic_evaluable': value.get('offtopic_evaluable', 0),
            'offtopic_total': value.get('offtopic_total', 0),
            'offtopic_flag_rate': ratio(value.get('offtopic_flagged', 0), value.get('offtopic_evaluable', 0))}


def report(config):
    from transformers import AutoTokenizer
    design, train, dev, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    new = [audit_reward(r) for r in episodes(config)]
    write_json(root / 'reward_step_audit.json', {r['episode_id']: {'audit': r.get('reward_step_audit'), 'reward': r.get('reward')}
                                               for r in new if r.get('reward')})
    old = {i: read_json(v['path']) for i, v in design['baseline_files'].items()}
    groups = {cohort: [r for r in new if r['cohort'] == cohort] for cohort in ('corrupted', 'real')}
    baseline = {cohort: [old[i] for i in design['corrupted_ids' if cohort == 'corrupted' else 'real_ids']]
                for cohort in groups}
    version = config[PHASE].get('version', 2)
    metrics = {'version': version, 'design': design, 'new': {}, 'baseline': {}, 'paired': {},
               'question_audit': read_json(root / 'question_audit.json'),
               'graph_quality': read_json(root / 'graph_quality.json'),
               'prompt_tokens': read_json(root / 'prompt_tokens.json'),
               'budget': read_json(root / 'api/accounting.json')}
    graph_rows = [read_json(p) for p in (root / 'graphs').glob('*.json')]
    metrics['graph_status'] = {'total': len(graph_rows), 'completed': sum(g['status'] == 'completed' for g in graph_rows),
                               'failures': [{'id': g['id'], 'error': g.get('error')} for g in graph_rows if g['status'] != 'completed']}
    for cohort, size in [('corrupted', 92), ('real', 30)]:
        metrics['new'][cohort] = summary(groups[cohort], size)
        metrics['baseline'][cohort] = summary(baseline[cohort], size, baseline=True)
    for key in ('R', 'R_rec', 'R_q', 'R_over', 'R_step'):
        metrics['paired'][key] = paired(groups['corrupted'], baseline['corrupted'], train, key)
    metrics['by_level'] = {}
    for level in sorted({train[i]['level'] for i in design['corrupted_ids']}):
        ids = {i for i in design['corrupted_ids'] if train[i]['level'] == level}
        left = [r for r in groups['corrupted'] if r['episode_id'] in ids]
        right = [r for r in baseline['corrupted'] if r.get('corpus_episode_id', r['episode_id']) in ids]
        metrics['by_level'][level] = {'new': summary(left, len(ids)), 'baseline': summary(right, len(ids), baseline=True),
                                      'paired_R': paired(left, right, train)}
    metrics['orchestration'] = orchestration(new)
    queried = set(metrics['orchestration']['off_topic_query_episodes'])
    metrics['off_topic_query_association'] = {str(used): {
        'episodes': sum((r['episode_id'] in queried) == used and any(x['op'] == 'G_OFFTOPIC' for x in train.get(r['episode_id'], {}).get('records', [])) for r in new),
        'recovery': avg(rec['main'] for r in new if r.get('completed') and r.get('reward') and (r['episode_id'] in queried) == used
                        for rec in r['reward']['combined']['per_record'] if rec['op'] == 'G_OFFTOPIC')}
        for used in (False, True)}
    metrics['markers'] = marker_metrics(config)
    observation_metrics = read_json(config['paths']['repo'] / 'verak/v3/outputs/observation_test/metrics.json')
    metrics['baseline_marker_metrics'] = {c: observation_metrics['markers']['rates'][c]['current'] for c in ('corrupted', 'real')}
    relevance = [read_json(p) for p in (root / 'relevance').glob('*.json')]
    metrics['relevance'] = {setting: {'expected': 30,
        'eligible_completed_essays': metrics['new' if setting == 'agentic' else 'baseline']['real']['completed'],
        'judged': sum(r['setting'] == setting for r in relevance),
        'reused_saved_judgments': sum(r['setting'] == setting and bool(r.get('reused_from')) for r in relevance),
        'changes': dict(Counter(r['judgment']['relevance_change'] for r in relevance if r['setting'] == setting)),
        'off_topic_edits': sum(len(r['judgment']['off_topic_edits']) for r in relevance if r['setting'] == setting),
        'essays_with_off_topic_edits': sum(bool(r['judgment']['off_topic_edits']) for r in relevance if r['setting'] == setting)}
        for setting in ('agentic', 'baseline')}
    judged = {setting: {r['id']: r['judgment'] for r in relevance if r['setting'] == setting}
              for setting in ('agentic', 'baseline')}
    matched = sorted(judged['agentic'].keys() & judged['baseline'].keys())
    metrics['relevance_paired'] = {'n': len(matched), 'ids': matched, 'baseline_to_agentic': dict(Counter(
        judged['baseline'][i]['relevance_change'] + ' -> ' + judged['agentic'][i]['relevance_change'] for i in matched))}
    metrics['evaluation_stages'] = {stage: read_json(root / (stage + '_status.json')) if (root / (stage + '_status.json')).exists()
                                    else {'status': 'not_run'} for stage in ('markers', 'relevance')}
    for status in metrics['evaluation_stages'].values():
        for error in status.get('errors', []):
            if isinstance(error.get('item'), (tuple, list)):
                # The full failed request remains in its raw stage-status file.
                error['item'] = error['item'][:2]
    metrics['export'] = export(new, train, tokenizer, root / 'export')
    metrics['low_use_tools'] = {k: {'episode_rate': v, 'calls': metrics['orchestration']['tool_calls'][k]}
                                for k, v in metrics['orchestration']['tool_episode_rates'].items() if v < .05}
    metrics['drop_tool_candidates'] = [k for k in metrics['low_use_tools']
        if metrics['orchestration']['tool_calls'][k] == 0 or k in {'PLAN', 'PROGRESS'}]
    metrics['environment_preflight'] = read_json(root / 'environment_preflight_failed/status.json') if (root / 'environment_preflight_failed/status.json').exists() else None
    metrics['protocol_preflight'] = read_json(root / 'protocol_preflight/status.json') if (root / 'protocol_preflight/status.json').exists() else None
    metrics['graph_quality']['intersection_retention'] = {
        k: {'intersection_over_union': ratio(metrics['graph_quality']['intersection_counts'].get(k + '_intersection', 0),
                                             metrics['graph_quality']['intersection_counts'].get(k + '_union', 0)),
            'intersection_over_mean_raw': ratio(2 * metrics['graph_quality']['intersection_counts'].get(k + '_intersection', 0),
                metrics['graph_quality']['intersection_counts'].get(k + '_left', 0) + metrics['graph_quality']['intersection_counts'].get(k + '_right', 0))}
        for k in ('sentence_edges', 'paragraph_edges')}
    metrics['graph_quality']['retention_population'] = 'semantically valid final intersections; invalid graph listed separately'
    metrics['graph_quality']['offtopic_flag_rate'] = ratio(metrics['graph_quality'].get('offtopic_flagged', 0),
                                                        metrics['graph_quality'].get('offtopic_evaluable', 0))
    metrics['graph_quality']['source_false_flag_proxy'] = ratio(metrics['graph_quality'].get('source_flagged', 0),
                                                             metrics['graph_quality'].get('source_sentences', 0))
    if version == 3:
        previous = config['paths']['repo'] / 'verak/v3/outputs/agentic_pilot'
        previous_quality = previous / 'graph_quality.json'
        previous_design = previous / 'design.json'
        comparison = {'v3_explicit_both': graph_flag_summary(metrics['graph_quality']),
                      'v2_no_path_to_Q': None, 'same_intended_population': None}
        if previous_quality.exists():
            comparison['v2_no_path_to_Q'] = graph_flag_summary(read_json(previous_quality))
            comparison['v2_source_path'] = str(previous_quality)
            comparison['v2_source_sha256'] = file_sha(previous_quality)
        if previous_design.exists():
            old_design = read_json(previous_design)
            comparison['same_intended_population'] = all(old_design[key] == design[key]
                for key in ('quality_corrupted_ids', 'quality_source_ids'))
        metrics['graph_flag_comparison'] = comparison
    metrics['runtime_files_unchanged'] = all(file_sha(p) == h for p, h in read_json(root / 'pilot_runtime.json')['files'].items())
    ci = metrics['paired']['R']['ci95']
    adopt = all(metrics['new'][c]['completion'] >= .9 for c in groups) and ci is not None and ci[1] >= 0
    metrics['recommendation'] = 'agentic' if adopt else 'current_two_stage'
    metrics['baseline_hashes_unchanged'] = all(file_sha(v['path']) == v['sha256'] for v in design['baseline_files'].values())
    metrics['all_observations_have_question'] = all(c['messages'][-1]['content'].find('[문항] ' + examples[
        train[r['episode_id']]['source_id'] if r['episode_id'] in train else r['episode_id']].question) >= 0
        for r in new for c in r.get('calls', []))
    test_log = root / 'tests_live.txt'
    metrics['test_result'] = test_log.read_text().strip().splitlines()[-1] if test_log.exists() else 'not_run'
    metrics['preflight_test_results'] = {path.name: path.read_text().strip().splitlines()[-1]
        for path in sorted(root.glob('*preflight*test*.txt')) if path.read_text().strip()}
    for name in ('tests_preflight.txt', 'check_env.txt'):
        path = root / name
        if path.exists() and path.read_text().strip():
            metrics['preflight_test_results'][name] = path.read_text().strip().splitlines()[-1]
    from .audit import audit
    metrics['validation'] = audit(config, new, tokenizer)
    write_json(root / 'metrics.json', metrics)
    render(config, metrics, new)
    print(json.dumps({'recommendation': metrics['recommendation'], 'budget': metrics['budget'], 'export': metrics['export']}, ensure_ascii=False))
    return metrics


def render(config, m, rows):
    root = config['paths'][PHASE + '_output']
    version = config[PHASE].get('version', 2)
    budget = config[PHASE]['max_cost_usd']
    output_rel = root.relative_to(config['paths']['repo']).as_posix()
    fmt = lambda v: '—' if v is None else f'{v:.4f}'
    orchestration_view = {k: v for k, v in m['orchestration'].items() if k not in {'preview_details', 'blocked_responses'}}
    blocked = m['orchestration']['blocked_responses']
    responses = Counter()
    for record in blocked:
        control = record['next_control_action']
        destination = 'no later control action' if control is None else (control['args']['agent'] if control['action'] == 'DELEGATE' else 'FINISH')
        responses[record['reporting_role'] + ' blocked -> ' + destination] += 1
    orchestration_view['blocked_response_counts'] = dict(responses)
    orchestration_view['blocked_response_examples'] = blocked[:3]
    previews = m['orchestration']['preview_details']
    orchestration_view['preview_changed_examples'] = [p for p in previews if p['next'] is not None and p['next'] != p['preview']][:3]
    lines = ['Recommendation: ' + ('adopt the three-agent design.' if m['recommendation'] == 'agentic'
                                  else 'retain the current two-stage design.'), '', f'# V3 Agentic Pilot (v{version})', '',
             'This is a design experiment on 92 agent_train corruptions and 30 Phase-6 real essays, not held-out evaluation. '
             'Both two-stage baselines were read from saved runs, including their failures; neither was regenerated.', '',
             f"Confirmed API usage cost: **${m['budget']['confirmed_usd']:.6f} / ${budget:g}**; outstanding conservative reservation: "
             f"${m['budget']['reserved_usd']:.6f}; live requests: {m['budget']['pending']}.", '',
             '## Completion and outcomes', '',
             '| Cohort | Design | Completed / intended | Combined R | R_over | DELETE / attempted essay | DELETE / completed essay | Teacher USD / attempted essay |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort in ('corrupted', 'real'):
        for source in ('baseline', 'new'):
            s = m[source][cohort]
            lines.append(f"| {cohort} | {source} | {s['completed']}/{s['expected']} | {fmt(s['rewards']['combined']['R'])} | {fmt(s['rewards']['combined']['R_over'])} | {fmt(s['deletions']['per_attempted_episode'])} | {fmt(s['deletions']['per_completed_episode'])} | {fmt(s['cost_per_attempt'])} |")
    lines += ['', 'Paired differences use only jointly completed corruptions; completion failures remain in the intended-population denominator. '
              'The recommendation requires at least 90% completion in both cohorts and a paired combined-R 95% CI that does not lie wholly below zero. '
              'This is a design decision rule, not proof of equivalence.', '',
              'R_over in the cohort table uses all completed corruption episodes; the paired table uses jointly completed episodes. '
              'Deletion counts include every valid whole-sentence DELETE, even if later undone. Saved baseline equivalents are GLOBAL EDIT actions '
              'with an empty replacement for one sentence ID. Unattempted essays are excluded from the attempted-episode deletion mean; '
              'intended-population and completed-episode counts are also saved.', '',
              'Cohort costs cover teacher API attempts. Graph extraction, quality inputs, preflights and Sol evaluation are listed '
              'separately in the shared budget ledger below.', '',
              '| Paired metric | n | New | Baseline | Difference | 95% CI |',
              '|---|---:|---:|---:|---:|---|']
    for key, result in m['paired'].items():
        interval = '—' if result['ci95'] is None else ' / '.join(fmt(x) for x in result['ci95'])
        lines.append(f"| {key} | {result['n']} | {fmt(result.get('new'))} | {fmt(result.get('baseline'))} | {fmt(result['delta'])} | {interval} |")
    lines += ['',
              '## Question audit and graph quality', '',
              'Question IDs use the existing question hash. ' +
              ('The v3 audit verifies the already repaired active corpus and records any additional repairs explicitly. '
               if version == 3 else 'Only question metadata was repaired; original active corpus files are archived locally. ') +
              'Graph quality covers all 57 active dev G_OFFTOPIC essays and their 48 distinct sources. Missing graph extractions remain missing. '
              'A source flag is a false-flag proxy, not a human judgment of irrelevance.', '',
              '```json', json.dumps({'question_audit': m['question_audit'], 'graph_quality': m['graph_quality'],
                                   'graph_status': m['graph_status'], 'prompt_tokens': m['prompt_tokens']}, ensure_ascii=False, indent=2), '```', '',
              'The graph contains only the intersection of two independent identical-prompt extractions. All semantic edge constraints and degree limits apply to the intersected state; '
              'raw invalid assertions are preserved, never arbitrarily pruned. ' +
              ('Pre-intersection validation failures were preserved and resumed from the same cached request, adding only a missing second extraction. '
               if version == 2 else '') + 'No extraction was regenerated to improve its answer. '
              'Graph extraction uses an 8,192 output-token allowance; every editor/orchestrator request uses the required 1,024.', '',
              ('V3 removes the no-path-to-Q rule. A current sentence is an off_topic_candidate only when both extraction responses explicitly '
               'name it in off_topic. Other discourse edges retain the exact intersection. A support relation to a relevant sentence can be evidence '
               'of indirect relevance, but relevance does not automatically propagate through every descendant or connection; '
               'missing Q paths do not establish irrelevance. No replacement relevance inference runs after editing.' if version == 3 else
               'Off-topic candidates are sentences without a directed graph path to Q.'), '',
              '```json', json.dumps(m.get('graph_flag_comparison', {'v2_no_path_to_Q': graph_flag_summary(m['graph_quality'])}), ensure_ascii=False, indent=2), '```', '',
              ('AUDIT contains only off-register sentences, conjunction/omitted-subject predecessor changes since the start, dangling edges, and '
               'explicit-agreement off_topic_candidate. It has no unsupported list. QUERY(unsupported) remains a graph query: '
               if version == 3 else '') +
              '`unsupported` means a Q-addressing node without an incoming supports/example_of edge. No separate claim classifier was added. '
              'Paragraph summaries are deterministic first-sentence excerpts (up to 80 characters). Discourse relations are not re-inferred after edits.', '',
              'The action contract reuses the existing editor primitives: MOVE accepts a sentence, a contiguous sentence range or a paragraph; '
              'INSERT anchors before/after a sentence; DELETE removes one sentence. Deleting a paragraph therefore requires its individual sentences '
              'to be deleted. These permissions and the exact prompts were fixed before the formal pilot.', '',
              ('V3 Composition DELETE is permitted only for a literal sentence ID named in the current Orchestrator task, after an exact deletion '
               'PREVIEW in the same state and delegation. Each delegation permits at most two executed DELETE actions; UNDO does not refund this limit. '
               'Editors are prompted to perform only their assigned task and then REPORT. SCORE is available at most once and only as the first '
               'Orchestrator action; there are at most three DELEGATE calls. Cohesion delegation requires factual disturbed markers inside its scope. '
               'The Orchestrator is prompted to FINISH when AUDIT has no issues.' if version == 3 else
               'V2 retains its original four-delegation and two-SCORE limits.'), '',
              '## Orchestration and tools', '', '```json', json.dumps(orchestration_view, ensure_ascii=False, indent=2), '```', '',
              'All blocked-report responses and every PREVIEW/next-action pair are preserved in '
              f'`{output_rel}/metrics.json`; the report shows counts and examples.', '',
              'AUDIT lists factual graph/profile findings, including historical predecessor changes; remaining findings do not establish remaining semantic errors. '
              'Completion means Orchestrator FINISH plus terminal reward calculation for corruptions. ' +
              ('With two editor steps remaining, the controller sends a final notice. If the editor reaches its 16-action limit without REPORT, '
               'the controller returns status "done (budget)" and terminal AUTO_REPORT. Auto-report counts and rates are shown for each editor '
               'and all editors, using both all delegations and final-notice recipients as denominators. ' if version == 3 else
               'An editor reaching 16 actions returns a controller-origin budget notice. ') +
              'Controller returns are counted separately from agent REPORT actions and are ineligible for export. '
              'PREVIEW coverage requires the exact action and arguments to have been previewed in the same document state and delegation; '
              'the stricter immediately-preceding rate is also reported. A changed decision compares a preview to the next valid write/UNDO/REPORT.', '',
              'QUERY(off_topic) association (observational, not causal):', '', '```json',
              json.dumps(m['off_topic_query_association'], ensure_ascii=False, indent=2), '```', '',
              'Tool recommendation: ' + ('drop ' + ', '.join(m['drop_tool_candidates']) if m['drop_tool_candidates'] else 'no removal candidate') + '. '
              'The rule flags unused tools and rarely used standalone ledgers (under 5% of episodes), whose incremental benefit was not measured. '
              'This is a simplification recommendation; the pilot has no tool ablation and cannot establish zero causal benefit. '
              'Concrete PREVIEW decisions are recorded above.', '',
              '```json', json.dumps(m['low_use_tools'], ensure_ascii=False, indent=2), '```', '',
              '## Rewards, recovery and runtime', '',
              'Editor recovery and over-edit reuse the existing definitions. For repeated delegations, role recovery uses the latest observed value '
              'for each eligible record/term, and role costs sum each invocation’s own over-edit and write steps. Read tools and ledgers cost no R_step. '
              'Combined R retains the existing all-action step cost except SCORE/QUERY/AUDIT/PREVIEW/PLAN/PROGRESS; '
              'DELEGATE/REPORT/FINISH therefore count, while editor-specific rewards charge writes only. Orchestrator additionally pays 0.02 per delegation. '
              'Baseline GLOBAL/KOREAN rewards retain their historical quality and step terms, so role rewards are descriptive rather than identical objectives. '
              'The saved baseline combined R is used unchanged for the requested design comparison.', '',
              'The terminal accounting audit corrects any preliminary write-only combined step count from saved actions. '
              'Recovery, over-edit, scores, policy inputs and actions are unchanged. Raw episode files and the explicit corrected reward map '
              '(`reward_step_audit.json`) are both preserved; comparisons and exports use the corrected values.', '',
              'Unlabelled real essays have no recovery-based reward; those entries are unavailable for both designs. '
              'Input/output/reasoning totals and averages per attempted essay include paid incomplete responses where recorded; '
              'reasoning tokens are part of output, not an additional charge.', '',
              '```json', json.dumps({'new': m['new'], 'baseline': m['baseline'], 'by_level': m['by_level']}, ensure_ascii=False, indent=2), '```', '',
              '## Residual markers and real-essay relevance', '',
              'Marker checks reuse the exact Sol high prompt and final predecessor/target input contract. Every valid structural action is retained in the denominator. '
              'Missing judgments and unfinished episodes are unknown, not passes; residual misfit is not causal attribution to the structural action. '
              'Deleted target counts and actions with no surviving candidate sites accompany the rates: they have no remaining site to judge, '
              'so a low residual-marker rate alone does not establish preservation of content.', '',
              '```json', json.dumps({'agentic_markers': m['markers'], 'saved_baseline_markers': m['baseline_marker_metrics'],
                                   'relevance': m['relevance'], 'relevance_paired': m['relevance_paired'],
                                   'stage_status': m['evaluation_stages']}, ensure_ascii=False, indent=2), '```', '',
              'Judged relevance counts are explicit; unjudged and unfinished essays are not counted as unchanged. '
              'Reused baseline Sol judgments are counted separately and require the identical saved baseline input and judgment contract. '
              'A lower-priority stage marked not_run was not dispatched when the remaining budget could not fund the preceding stage.', '',
              '## Export, verification and budget', '',
              'Exports preserve the exact inference input and action JSON, with loss only on the target. Observations, prior reports, ledgers and tool results are masked. '
              'The 0.80 threshold is literal for every agent; no-GLOBAL Composition sequences receive no special reward exemption. ' +
              ('The separately authorized two-stage bulk generation has its own $25 ledger and report, `imple/reports/V3_TEACHER_BULK_TWO_STAGE.md`. '
               'No SFT or RFT was run.' if version == 3 else 'No bulk generation, SFT or RFT was run.'), '', '```json',
              json.dumps({'export': m['export'], 'budget': m['budget'], 'baseline_hashes_unchanged': m['baseline_hashes_unchanged'],
                          'all_observations_have_question': m['all_observations_have_question'], 'validation': m['validation']}, ensure_ascii=False, indent=2), '```', '',
              f'All new extraction, teacher and Sol calls in this pilot share a ${budget:g} ledger; conservative reservations block requests before crossing the cap. '
              'Priority: question audit → graph extraction/quality → 92 corruptions → 30 real essays → Sol markers → Sol relevance. '
              'Usage costs use the existing pinned ledger rates, with reasoning already included in output tokens.', '',
              ('Environment preflight: the first launch lacked the Bareun key used by existing launchers. Its 12 started essays '
              'were archived separately (11 result files plus partial events), all costs remain charged, and one interrupted request retains '
              'its conservative reservation. The actual pilot began only after a live Bareun preflight; PREVIEW JSON nesting was also clarified '
              'before the final runtime was frozen. These environment-invalid attempts are excluded from design outcomes, not erased.' if version == 2 else
               'V3 uses a distinct output directory and ledger; earlier v2 graphs, trajectories, requests and reports are preserved. '
               'Live stages check Bareun availability before paid model calls. Saved contracts are audited without new model or analyzer calls.'), '',
              ('A subsequent one-essay integration run verified live edits but exposed multiple/malformed JSON responses. It is archived as protocol preflight. '
              'The formal pilot constrains action JSON syntax with role-specific schemas; semantic errors and budgets are still checked by the environment. '
              'No baseline was rerun and every preflight cost remains in the same cap.' if version == 2 else
               'The formal pilot retains role-specific action JSON schemas and the 8,192-token context. Syntactic constraints do not establish semantic '
               'correctness. Graph agreement remains a model judgment; source false flags are a proxy, and the small paired design comparison '
               'does not establish held-out performance. Editor auto-reports mean budget exhaustion, not verified completion of the assigned task.'), '',
              '```json', json.dumps({'environment_preflight': m['environment_preflight'], 'protocol_preflight': m['protocol_preflight'],
                                    'preflight_test_results': m.get('preflight_test_results', {}),
                                    'runtime_files_unchanged': m['runtime_files_unchanged']},
                                   ensure_ascii=False, indent=2), '```', '',
              'Final test suite: `' + m['test_result'] + '`.', '',
              f'Validation logs: `{output_rel}/tests_live.txt`; all raw events, requests, failures, graph responses and exports are local under '
              f'`{output_rel}/`.', '', '## Three full example trajectories', '',
              'Examples favor short completed episodes involving all three roles, including a blocked report and a PREVIEW. '
              'This readability selection does not affect aggregate outcomes.', '']
    # Prefer one blocked-report episode, one preview episode, then other short
    # completed episodes; selection affects examples only, never aggregate metrics.
    complete = sorted((r for r in rows if r.get('completed')), key=lambda r: len(r['calls']))
    all_roles = [r for r in complete if {a['role'] for a in r['actions']} == {'orchestrator', 'composition', 'cohesion'}]
    if len(all_roles) >= 3:
        complete = all_roles
    chosen = []
    for predicate in (lambda r: any(a['action'] == 'REPORT' and a['args'].get('status') == 'blocked' for a in r['actions']),
                      lambda r: any(a['action'] == 'PREVIEW' for a in r['actions']), lambda r: True):
        match = next((r for r in complete if r not in chosen and predicate(r)), None)
        if match:
            chosen.append(match)
    for row in complete:
        if len(chosen) >= 3:
            break
        if row not in chosen:
            chosen.append(row)
    for row in chosen:
        lines += ['### ' + row['episode_id'], '']
        for event in read_jsonl(root / 'events' / (safe_id(row['episode_id']) + '.jsonl')):
            if event['event'] == 'call':
                lines += [f"**{event['role']} / delegation {event['delegation']} / turn {event['turn']}**", '']
                for message in event['messages']:
                    lines += ['```text', message['role'] + '\n' + message['content'], '```', '']
                lines += ['```json', event['raw'], '```', '']
            elif event['event'] in {'action', 'editor_return', 'end'}:
                lines += ['```json', json.dumps(event, ensure_ascii=False, indent=2), '```', '']
    filename = 'V3_AGENTIC_PILOT_V3.md' if version == 3 else 'V3_AGENTIC_PILOT.md'
    path = config['paths']['repo'] / 'imple/reports' / filename
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
