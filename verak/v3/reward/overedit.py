"""Morpheme over-edit on unreferenced source sentences, with role attribution."""
from difflib import SequenceMatcher


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, x in enumerate(left, 1):
        current = [i]
        for j, y in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def referenced_ids(records):
    return {sid for r in records for sid in r['sids']} | {
        c['sid'] for r in records for c in r.get('coupled_changes', [])}


def mapped_span(before, after, start, end):
    """Map exact boundaries through surface edits; abstain inside a larger rewrite."""
    def boundary(position, side):
        possible = []
        for tag, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if tag == 'equal' and a <= position <= b:
                possible.append(c + position - a)
            elif position == a:
                possible.append(c)
            elif position == b:
                possible.append(d)
        if not possible:
            return None
        return (max if side == 'start' else min)(possible)
    a, b = boundary(start, 'start'), boundary(end, 'end')
    return (a, b) if a is not None and b is not None and a <= b else None


def _tokens(unit, excluded):
    if unit is None:
        return []
    return [(t.form, t.tag) for t in unit.tokens
            if not any(t.start < b and t.end > a for a, b in excluded)]


def action_ids(actions, before, after):
    """Explicit changed_sids are preferred; simple stable-ID targets also work.

    UNDO needs no special replay: attribution is the union of attempted valid
    changes, while distance is measured between actual stage endpoints.
    """
    units = {u.sid for d in (before, after) for u in d.units}
    paragraphs = {p.pid: {u.sid for u in p.units} for d in (before, after) for p in d.paragraphs}
    result = set()
    for action in actions:
        if action.get('valid', True) is False or action.get('action') not in {'EDIT', 'MOVE'}:
            continue
        result.update(action.get('changed_sids', []))
        target = action.get('args', {}).get('target', '')
        if target.startswith(('before:', 'after:')):
            # Inserted IDs must be in changed_sids. The anchor was not edited.
            continue
        target = target.split(':', 1)[0]
        if target in units:
            result.add(target)
        elif target in paragraphs:
            result.update(paragraphs[target])
        elif '-' in target:
            first, last = target.split('-', 1)
            ordered = list(dict.fromkeys(u.sid for d in (before, after) for u in d.units))
            if first in ordered and last in ordered:
                a, b = ordered.index(first), ordered.index(last)
                result.update(ordered[min(a,b):max(a,b)+1])
    return result


def overedit(source, before, after, records, *, actions=None, preexisting_spell_spans=()):
    """Role distance uses its own start/end states; combined uses source/end.

    Spell spans use {sid, start, end} in source-sentence-local character offsets,
    or {start,end} in source-document offsets. No spelling API is called here.
    """
    excluded = referenced_ids(records)
    old = {u.sid: u for u in before.units}
    new = {u.sid: u for u in after.units}
    source_annotations = {a.sid: a for a in source.structure().annotations}
    attributed = action_ids(actions, before, after) if actions is not None else None
    costs, normalizers, detail = 0, 0, []
    for original in source.units:
        sid = original.sid
        if sid in excluded:
            continue
        left, right = old.get(sid), new.get(sid)
        changed = (left.text if left else None) != (right.text if right else None)
        if attributed is not None and changed and sid not in attributed:
            raise ValueError(f'Unattributed role change to {sid}')
        spans = []
        for span in preexisting_spell_spans:
            if 'sid' in span:
                if span['sid'] == sid:
                    spans.append((span['start'], span['end']))
            else:
                ann = source_annotations[sid]
                if span['start'] < ann.end and span['end'] > ann.start:
                    spans.append((max(0, span['start']-ann.start), min(len(original.text), span['end']-ann.start)))
        def mask(unit):
            if unit is None:
                return []
            return [mapped for a,b in spans
                    if (mapped := mapped_span(original.text, unit.text, a,b)) is not None]
        a, b = _tokens(left, mask(left)), _tokens(right, mask(right))
        cost = edit_distance(a,b) if changed else 0
        normalizer = max(len(a),len(b))
        costs += cost
        normalizers += normalizer
        detail.append({'sid': sid, 'distance': cost, 'normalizer': normalizer})
    return {'value': costs / normalizers if normalizers else 0., 'distance': costs,
            'normalizer': normalizers, 'sentences': detail}
