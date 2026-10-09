"""v2 main recovery and over-insertion; v1 reward functions remain unmodified."""
from copy import deepcopy

from ..corrupt.document import MARKER
from ..reward.overedit import edit_distance, overedit
from ..reward.recovery import recover_records, coupled_recovery
from ..reward.total import LOCAL_LEVELS, _breakdown, quality_reward
from .config import require_v2
from .operators import content_morphemes, relation

NEW_OPERATORS = {'G_DEL_LINK', 'L_FUSE'}


def insertion_origins(actions):
    """Legacy EDIT insertions also count; splitting an insertion keeps its origin."""
    origins = {}
    for index, action in enumerate(actions):
        if not action.get('valid', True):
            continue
        name, args = action['action'], action.get('args') or {}
        target = args.get('target', '')
        inserted = name == 'INSERT' or (name == 'EDIT' and isinstance(target, str)
                                        and target.startswith(('before:', 'after:')))
        created = action.get('created_sids', [])
        if inserted:
            if 'created_sids' not in action:
                raise ValueError('v2 insertion attribution requires created_sids')
            origins.update({sid: index for sid in created})
        if name == 'SPLIT':
            parents = set(action.get('changed_sids', [])) - set(created)
            inherited = {origins[sid] for sid in parents if sid in origins}
            if inherited:
                if len(inherited) != 1:
                    raise ValueError('Ambiguous split insertion origin')
                origins.update({sid: next(iter(inherited)) for sid in created})
    return origins


def fuse_recovery(source, final, record):
    target = record['recovery_target']
    paragraph = next((p for p in final.paragraphs if p.pid == target['paragraph']), None)
    position, ids = target['position'], target['source_sids']
    units = paragraph.units[position:position+len(ids)] if paragraph else []
    details, mapping = [], {}
    if len(units) != len(ids):
        return 0., {}, {'restored_sentences': len(units), 'expected_sentences': len(ids), 'checks': []}
    for i, (sid, current) in enumerate(zip(ids, units)):
        original = source.locate(sid)[2]
        original_content = content_morphemes(original)
        content_ok = bool(original_content) and content_morphemes(current) == original_content
        markers_ok = MARKER.findall(current.text) == MARKER.findall(original.text)
        observed = relation(current)
        expected = target['source_classes'][i-1] if i else None
        conjunction_ok = i == 0 or (observed in {'NONE', 'ADDITION'} if expected == 'NONE' else observed == expected)
        details.append({'source_sid': sid, 'restored_sid': current.sid, 'content_retained': content_ok,
            'markers_retained': markers_ok, 'source_conjunction_class': expected,
            'restored_conjunction_class': observed, 'conjunction_ok': conjunction_ok})
        mapping[current.sid] = sid
    success = all(d['content_retained'] and d['markers_retained'] and d['conjunction_ok'] for d in details)
    return float(success), mapping if success else {}, {'checks': details,
        'definition': 'all source sentence boundaries and content-morpheme multisets restored at source positions'}


def link_recovery(final, record, inserted_ids, judge, occupied=()):
    target = record['recovery_target']
    paragraph = next((p for p in final.paragraphs if p.pid == target['paragraph']), None)
    candidates = [(i, u) for i, u in enumerate(paragraph.units if paragraph else [])
        if u.sid in inserted_ids and u.sid not in occupied and abs(i-target['position']) <= 1]
    judged = []
    for index, unit in candidates:
        value = judge(record, unit)
        if value not in (0., .5, 1.):
            raise ValueError('A cached Luna recovery judgment must be 0, 0.5, or 1; missing is not zero')
        judged.append({'sid': unit.sid, 'position': index, 'score': float(value)})
    positive = [r for r in judged if r['score'] > 0]
    best = min(positive, key=lambda r: (-r['score'], abs(r['position']-target['position']), r['sid'])) if positive else None
    return best['score'] if best else 0., best['sid'] if best else None, judged


def recovery(source, corrupted, final, records, actions, judge, *, coupled_weight=.3):
    inserted = insertion_origins(actions)
    legacy = [r for r in records if r['op'] not in NEW_OPERATORS]
    old = {r['record_id']: r for r in recover_records(source, final, legacy,
        corrupted=corrupted, coupled_weight=coupled_weight)}
    results, matched, canonical = [], set(), {}
    for record in records:
        if record['op'] not in NEW_OPERATORS:
            results.append(old[record['record_id']])
            continue
        if record['op'] == 'G_DEL_LINK':
            main, sid, detail = link_recovery(final, record, inserted, judge, matched)
            if sid:
                matched.add(sid)
                if main == 1:
                    canonical[sid] = record['sids'][0]
        else:
            main, mapping, detail = fuse_recovery(source, final, record)
            canonical.update(mapping)
        results.append({'record_id': record['record_id'], 'op': record['op'], 'level': 'GLOBAL',
                        'main': main, 'recovery_details': detail})
    annotations = {a.sid: a for a in deepcopy(final.structure().annotations)}
    # A semantically matched new sentence can restore a predecessor even with a fresh runtime ID.
    # This identity mapping exists only inside hidden reward computation, never in the policy view.
    for ann in annotations.values():
        ann.predecessor_id = canonical.get(ann.predecessor_id, ann.predecessor_id)
    source_annotations = {a.sid: a for a in source.structure().annotations}
    for record, result in zip(records, results):
        if record['op'] not in NEW_OPERATORS:
            continue
        coupled, changes = coupled_recovery(record, annotations, source_annotations)
        result.update(coupled=coupled, coupled_changes=changes,
            recovery=(1-coupled_weight)*result['main'] + coupled_weight*coupled if coupled is not None else result['main'])
    return results, matched, inserted


def overedit_v2(source, before, after, records, *, actions=None, origins=(), matched=(), preexisting_spell_spans=()):
    """Add unmatched surviving insertions as virtual untouched sentence costs."""
    adapted = None
    if actions is not None:
        adapted = []
        for action in actions:
            value = deepcopy(action)
            if value.get('action') in {'INSERT', 'SPLIT'}:
                value['action'] = 'EDIT'
            adapted.append(value)
    result = overedit(source, before, after, records, actions=adapted,
                      preexisting_spell_spans=preexisting_spell_spans)
    old, new = {u.sid: u for u in before.units}, {u.sid: u for u in after.units}
    changed = {sid for a in (actions or []) if a.get('valid', True) for sid in a.get('changed_sids', [])}
    insertions = []
    for sid in sorted(set(origins) - set(matched)):
        if sid not in new:
            continue  # Deleted/undone insertions do not incur endpoint over-edit.
        a = [(t.form, t.tag) for t in old[sid].tokens] if sid in old else []
        b = [(t.form, t.tag) for t in new[sid].tokens]
        distance = edit_distance(a, b)
        if actions is not None and distance and sid not in changed:
            raise ValueError('Unattributed v2 insertion change: ' + sid)
        normalizer = max(len(a), len(b))
        result['distance'] += distance
        result['normalizer'] += normalizer
        insertions.append({'sid': sid, 'distance': distance, 'normalizer': normalizer})
    result['morpheme'] = result['distance'] / result['normalizer'] if result['normalizer'] else 0.
    result['value'] = .5 * result['morpheme'] + .5 * result['order']
    result['unmatched_insertions'] = insertions
    result['matched_deletion_insertions'] = sorted(matched)
    return result


def rewards_v2(source, corrupted, final, records, *, config, genre, q_corrupted, q_stage1, q_final,
               stage1, stage1_actions, stage2_actions, judge, preexisting_spell_spans=()):
    require_v2(config)
    weights = config['reward']
    weight = weights.get('dependents_weight', .3)
    all_actions = list(stage1_actions) + list(stage2_actions)
    middle, middle_matches, middle_origins = recovery(source, corrupted, stage1, records, stage1_actions, judge,
                                                     coupled_weight=weight)
    end, end_matches, end_origins = recovery(source, corrupted, final, records, all_actions, judge, coupled_weight=weight)
    combined = _breakdown([r['recovery'] for r in end], end, quality_reward(q_corrupted, q_final, genre, weights),
        overedit_v2(source, source, final, records, origins=end_origins, matched=end_matches,
                    preexisting_spell_spans=preexisting_spell_spans), len(all_actions), weights)
    global_records = [r for r in middle if r['level'] == 'GLOBAL']
    global_result = _breakdown([r['main'] for r in global_records], global_records,
        quality_reward(q_corrupted, q_stage1, genre, weights),
        overedit_v2(source, corrupted, stage1, records, actions=stage1_actions,
            origins=middle_origins, matched=middle_matches, preexisting_spell_spans=preexisting_spell_spans),
        len(stage1_actions), weights)
    korean_records = [r for r in end if r['level'] in LOCAL_LEVELS or r['coupled'] is not None]
    korean_terms = [r['main'] for r in end if r['level'] in LOCAL_LEVELS] + [r['coupled'] for r in end
        if r['level'] == 'GLOBAL' and r['coupled'] is not None]
    korean_result = _breakdown(korean_terms, korean_records, quality_reward(q_stage1, q_final, genre, weights),
        overedit_v2(source, stage1, final, records, actions=stage2_actions,
            origins=end_origins, matched=end_matches, preexisting_spell_spans=preexisting_spell_spans),
        len(stage2_actions), weights)
    return {'mode': 'two_stage', 'global': global_result, 'korean': korean_result, 'combined': combined}
