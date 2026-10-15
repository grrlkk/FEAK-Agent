"""Read saved trajectories only; no analyzers, scorers, or generation calls."""
from collections import Counter, defaultdict
from statistics import mean
import re

from ..common import read_json, write_json, file_sha
from ..phase2 import read_jsonl, write_jsonl
from .pilot import safe_id
from .pilot_report import action_type
from .teacher_comparison import PHASE, prepare

NOTICE_TYPES = ('polarity_change', 'modality_change', 'ec_relation_change', 'conjunction_change',
    'dependency_change', 'subject_omission_change', 'focus_change', 'style_change')


def layout_text(snapshot):
    return {u['sid']: u['text'] for p in snapshot['paragraphs'] for u in p['units']}


def aliases(row):
    return {u['sid']: f'S{i+1}' for i, u in enumerate(
        u for p in row['initial_layout']['paragraphs'] for u in p['units'])}


def profile_lines(content):
    marker = '[Korean document profile:'
    if marker not in content:
        return {}
    body = content.split(marker, 1)[1].split('\n', 1)[1]
    return {m[1]: line for line in body.splitlines()
            if (m := re.match(r'^((?:S\d+[a-z]*|N\d+))(?: \||$)', line))}


def is_full(content):
    return ('[전체 갱신]' in content or ('[부분 갱신' not in content and '[글]' in content)) and bool(profile_lines(content))


def visibility(call, public_sid, raw_history=None):
    users = [m['content'] for m in call['messages'] if m['role'] == 'user']
    full = [v for v in users if is_full(v)]
    latest_lines = next((profile_lines(v) for v in reversed(users) if public_sid in profile_lines(v)), {})
    latest_text = next((m[1] for v in reversed(users)
        if (m := re.search(r'^'+re.escape(public_sid)+r' \| (.*)$', v.split('[Korean document profile:', 1)[0], re.M))), None)
    result = {'immediate_observation_full': bool(users and is_full(users[-1])),
        'full_profile_in_context': bool(full),
        'target_in_full_profile': any(public_sid in profile_lines(v) for v in full),
        'target_profile_visible': public_sid in latest_lines,
        'latest_target_line': latest_lines.get(public_sid), 'latest_visible_text': latest_text,
        'history_compacted': call.get('history_compacted', False)}
    if raw_history is not None:
        known = visibility({'messages': raw_history}, public_sid)
        result.update(target_profile_current=result['latest_target_line'] == known['latest_target_line'],
            target_text_current=result['latest_visible_text'] == known['latest_visible_text'],
            latest_known_target_line=known['latest_target_line'])
    return result


def record_trace(row, record):
    mapping = aliases(row)
    sites = {mapping[s] for s in record['sids'] if s in mapping}
    initial = layout_text(row['initial_layout'])
    middle = layout_text(row['stage1_layout'])
    final = layout_text(row['final_layout'])
    calls = {int(c['turn'].split(':')[0]): c for c in row['calls'] if c['role'] == 'korean'}
    history = row['messages_by_role']['korean']
    indices = [i for i, m in enumerate(history) if m['role'] == 'assistant']
    raw_context = {c['turn']: history[:i] for c, i in zip(
        [c for c in row['calls'] if c['role'] == 'korean'], indices)}
    trace = []
    for action in row['actions_by_role']['korean']:
        if action['action'] != 'EDIT':
            continue
        target = action['args'].get('target', '').split(':', 1)[0]
        if target not in sites:
            continue
        call = calls[action['t']]
        trace.append({'t': action['t'], 'valid': action['valid'], 'args': action['args'],
            'thought': action['thought'], 'error_code': action.get('error_code'),
            'visibility': visibility(call, target, raw_context[call['turn']])})
    last_call = next((c for c in reversed(row['calls']) if c['role'] == 'korean'), None)
    recovery = next(r for r in row['reward']['korean']['per_record'] if r['record_id'] == record['record_id'])
    ordered = list(middle)
    neighborhood = sorted({i for sid in record['sids'] if sid in middle
        for i in range(max(0, ordered.index(sid)-2), min(len(ordered), ordered.index(sid)+2))})
    return {'main_recovery': recovery['main'], 'R': row['reward']['korean']['R'],
        'R_q': row['reward']['korean']['R_q'], 'R_over': row['reward']['korean']['R_over'],
        'steps': row['steps']['korean'], 'termination': row['termination']['korean'],
        'source_sids': record['sids'], 'public_sids': sorted(sites),
        'site_present_after_global': all(s in middle for s in record['sids']),
        'initial': {s: initial.get(s) for s in record['sids']},
        'after_global': {s: middle.get(s) for s in record['sids']},
        'context_after_global': [{'sid': mapping.get(ordered[i], ordered[i]), 'text': middle[ordered[i]]} for i in neighborhood],
        'global_actions': [{k: a.get(k) for k in ('t', 'action', 'args', 'thought', 'valid')}
                           for a in row['actions_by_role']['global']],
        'final': {s: final.get(s) for s in record['sids']}, 'edits': trace,
        'last_turn_visibility': {s: visibility(last_call, s, raw_context[last_call['turn']]) for s in sorted(sites)} if last_call else {}}


def disturbance(rows):
    details = []
    for row in rows:
        edited = {sid for a in row['actions_by_role'].get('korean', [])
            if a['valid'] and a['action'] == 'EDIT' for sid in a['changed_public_sids']}
        for a in row['actions_by_role'].get('global', []):
            kind = action_type(a)
            if kind not in {'MOVE', 'sentence_insert', 'sentence_delete'}:
                continue
            facts = a['cohesion_changes']
            types = defaultdict(set)
            for notice in facts:
                types[notice['type']].add(notice['sid'])
            details.append({'id': row['corpus_episode_id'], 't': a['t'], 'action': kind,
                'valid': a['valid'], 'types': {k: sorted(v) for k, v in types.items()},
                'notices': facts, 'edited_sids': sorted(edited & {f['sid'] for f in facts})})
    valid = [a for a in details if a['valid']]
    def counts(actions, notice_type=None):
        sites = [(a, set(a['types'].get(notice_type, [])) if notice_type else
                  {s for v in a['types'].values() for s in v}) for a in actions]
        emitted = [(a, s) for a, s in sites if s]
        touched = sum(bool(s & set(a['edited_sids'])) for a, s in emitted)
        site_count = sum(len(s) for _, s in emitted)
        edited_count = sum(len(s & set(a['edited_sids'])) for a, s in emitted)
        return {'actions': len(actions), 'with_notice': len(emitted),
            'notice_share': len(emitted)/len(actions) if actions else None,
            'later_korean_edited_actions': touched,
            'later_edit_share_among_notice_actions': touched/len(emitted) if emitted else None,
            'action_sentence_pairs': site_count, 'later_edited_action_sentence_pairs': edited_count,
            'later_edit_site_share': edited_count/site_count if site_count else None}
    return {'episodes': len(rows), 'attempted_structural_actions': len(details),
        'rejected_structural_actions': len(details)-len(valid), 'accepted': counts(valid),
        'by_type': {t: counts(valid, t) for t in NOTICE_TYPES},
        'by_action': {a: counts([v for v in valid if v['action'] == a])
                      for a in ('MOVE', 'sentence_insert', 'sentence_delete')}, 'details': details}


def diagnose(config):
    design, corpus = prepare(config)
    root = config['paths'][PHASE+'_output']
    roots = {k: config['paths'][key] for k, key in
        [('pilot1', 'phase7_pilot_output'), ('pilot2', 'phase7_pilot2_output')]}
    rows = {k: [read_json(p) for p in sorted((r/'episodes').glob('*.json'))] for k, r in roots.items()}
    lookup = {k: {r['corpus_episode_id']: r for r in v} for k, v in rows.items()}
    l1 = [id for id in design['pilot_ids'] if corpus[id]['level'] == 'L1']
    if len(l1) != 25:
        raise ValueError('Expected the same 25 L1 essays')
    from .pilot2 import recompute_old
    original_rewards = {id: lookup['pilot1'][id]['reward']['korean'] for id in l1}
    for id in l1:
        lookup['pilot1'][id] = recompute_old(lookup['pilot1'][id], corpus[id])
    records = []
    for id in l1:
        for record in corpus[id]['records']:
            if record['level'] == 'GLOBAL':
                continue
            records.append({'id': id, 'record_id': record['record_id'], 'op': record['op'],
                'source': {s: layout_text(corpus[id]['source_layout']).get(s) for s in record['sids']},
                **{k: record_trace(lookup[k][id], record) for k in roots}})
    transitions = Counter((r['pilot1']['main_recovery'], r['pilot2']['main_recovery']) for r in records)
    summary = {'essays': len(l1), 'records': len(records),
        'transitions': [{'before': a, 'after': b, 'count': n} for (a, b), n in sorted(transitions.items())],
        'operators': {op: {'n': sum(r['op'] == op for r in records),
            **{k: mean(r[k]['main_recovery'] for r in records if r['op'] == op) for k in roots}}
            for op in sorted({r['op'] for r in records})},
        'pilot1_original_formula': {m: mean(original_rewards[id][m] for id in l1) for m in ('R', 'R_over')},
        'roles': {k: {m: mean(lookup[k][id]['reward']['korean'][m] for id in l1)
            for m in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')} for k in roots}}
    for k in roots:
        edits = [e for r in records for e in r[k]['edits'] if e['valid']]
        summary[k+'_visibility'] = {'accepted_target_EDITS': len(edits),
            **{name: sum(e['visibility'][name] for e in edits) for name in
               ('immediate_observation_full', 'full_profile_in_context', 'target_in_full_profile', 'target_profile_visible',
                'target_profile_current', 'target_text_current')},
            'unattempted_records': sum(not r[k]['edits'] for r in records),
            'missing_site_after_global': sum(not r[k]['site_present_after_global'] for r in records)}
    changes = sorted([r for r in records if r['pilot2']['main_recovery'] < r['pilot1']['main_recovery']], key=lambda r:r['id'])
    examples = changes[:3]
    if len(examples) < 3:
        examples += [r for r in records if r not in examples][:3-len(examples)]
    real_root = config['paths']['phase6_output']/'real/episodes'
    rows['phase6_real'] = [read_json(p) for p in sorted(real_root.glob('*.json'))]
    disturbances = {k: disturbance(v) for k, v in rows.items()}
    # Extra same-92 tables separate cohort composition from runtime changes.
    for k in ('pilot1', 'pilot2'):
        disturbances[k+'_paired92'] = disturbance([lookup[k][id] for id in design['pilot_ids']])
    write_jsonl(root/'l1_per_record.jsonl', records)
    write_json(root/'l1_diagnostic.json', {'summary': summary, 'examples': examples})
    write_json(root/'marker_disturbance.json', disturbances)
    paths = [p for r in roots.values() for p in (r/'episodes').glob('*.json')]+list(real_root.glob('*.json'))
    write_json(root/'diagnostic_provenance.json', {'api_calls': 0, 'bareun_calls': 0, 'scorer_calls': 0,
        'files': {str(p): file_sha(p) for p in paths},
        'definitions': {'fix': 'accepted KOREAN EDIT targeting a corruption site; not a semantic success claim',
            'disturbance': 'emitted notices after accepted GLOBAL structural actions; not a damage verdict',
            'later_edit': 'any accepted stage-2 EDIT to a notice-bearing public sentence ID, including later-UNDOne edits'}})
    print({'L1': summary, 'marker_actions': {k:v['accepted'] for k,v in disturbances.items()}}, flush=True)
