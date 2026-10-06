"""Record-blind rewrite alignment and observable changes (not error judgments)."""
from collections import Counter
from difflib import SequenceMatcher
import re

from scipy.optimize import linear_sum_assignment
import numpy as np


def align_rewrite(corrupted, rewritten, threshold=.35):
    """Match only against the input the baseline actually saw, never source/records."""
    result = rewritten.clone()
    old, new = corrupted.units, result.units
    def normalize(text):
        return re.sub(r'\s+', '', text)
    matrix = np.array([[SequenceMatcher(None, normalize(a.text), normalize(b.text), autojunk=False).ratio()
                        for b in new] for a in old])
    matches = {}
    if old and new:
        for i, j in zip(*linear_sum_assignment(-matrix)):
            if matrix[i, j] >= threshold:
                matches[int(j)] = (old[i].sid, float(matrix[i, j]))
    used = {u.sid for u in old}
    serial, details = 0, []
    for j, unit in enumerate(new):
        if j in matches:
            unit.sid, similarity = matches[j]
        else:
            serial += 1
            while f'B{serial}' in used:
                serial += 1
            unit.sid, similarity = f'B{serial}', None
        details.append({'final_index': j, 'sid': unit.sid, 'lexical_similarity': similarity})
    overlap = np.array([[len({u.sid for u in a.units} & {u.sid for u in b.units})
                         for b in result.paragraphs] for a in corrupted.paragraphs])
    paragraph_matches = {}
    if corrupted.paragraphs and result.paragraphs:
        for i, j in zip(*linear_sum_assignment(-overlap)):
            if overlap[i, j]:
                paragraph_matches[int(j)] = corrupted.paragraphs[i].pid
    number = max((int(p.pid[1:]) for p in corrupted.paragraphs), default=0)
    for j, paragraph in enumerate(result.paragraphs):
        if j in paragraph_matches:
            paragraph.pid = paragraph_matches[j]
        else:
            number += 1
            paragraph.pid = f'P{number}'
    return result, {'method': 'corrupted_only_hungarian_lexical', 'threshold': threshold,
        'matched': len(matches), 'new_units': len(new)-len(matches),
        'unmatched_input': len(old)-len(matches), 'sentences': details}


def changes(before, after, actions=()):
    old, new = {u.sid: u for u in before.units}, {u.sid: u for u in after.units}
    removed, added = set(old)-set(new), set(new)-set(old)
    common = set(old) & set(new)
    rewritten = {s for s in common if old[s].text != new[s].text}
    # Relative order among surviving units avoids treating a deletion as moving everything after it.
    previous = [u.sid for u in before.units if u.sid in common]
    final = [u.sid for u in after.units if u.sid in common]
    moved = {s for i, s in enumerate(previous) if final.index(s) != i or
             before.paragraphs[before.locate(s)[0]].pid != after.paragraphs[after.locate(s)[0]].pid}
    old_structure, new_structure = before.structure(), after.structure()
    oa = {a.sid: a for a in old_structure.annotations}
    facts = []
    for a in new_structure.annotations:
        b = oa.get(a.sid)
        if b is None:
            continue
        for name, level in (('polarity', 'WORD'), ('modality', 'WORD')):
            if getattr(a, name) != getattr(b, name):
                facts.append({'sid': a.sid, 'kind': name, 'level': level,
                              'before': getattr(b, name), 'after': getattr(a, name)})
        def conjunction(ann):
            return (ann.initial_conj or {}).get('coarse_class')
        if conjunction(a) != conjunction(b):
            facts.append({'sid': a.sid, 'kind': 'conjunction_relation', 'level': 'SENTENCE',
                          'before': conjunction(b), 'after': conjunction(a)})
        prior = Counter(e['surface'] for e in b.subject_evidence)
        current = Counter(e['surface'] for e in a.subject_evidence)
        for surface, n in (current-prior).items():
            # Count inserted subject phrases, not a renamed/replaced existing subject.
            inserted = [a.text[c:d] for op, _, _, c, d in
                        SequenceMatcher(None, b.text, a.text, autojunk=False).get_opcodes() if op == 'insert']
            if surface not in b.text and any(surface in fragment for fragment in inserted):
                facts.extend({'sid': a.sid, 'kind': 'explicit_subject_inserted', 'level': 'SENTENCE',
                              'surface': surface, 'omitted_before': b.subject_omitted} for _ in range(n))
    counts = Counter(f['kind'] for f in facts)
    acts = Counter()
    for action in actions:
        if not action.get('valid', True):
            continue
        kind, args = action.get('action'), action.get('args', {})
        if kind == 'MOVE':
            acts['MOVE'] += 1
        elif kind == 'EDIT':
            if args.get('target', '').startswith(('before:', 'after:')):
                acts['sentence_insert'] += 1
            elif not args.get('new_text', '').strip() and ':' not in args.get('target', ''):
                acts['sentence_delete'] += 1
            else:
                acts['in_sentence_EDIT'] += 1
    return {'source_sentences': len(old), 'final_sentences': len(new),
        'sentences_changed': len(rewritten | removed | moved),
        'changed_share': len(rewritten | removed | moved)/max(1, len(old)),
        'text_changed_share': len(rewritten | removed)/max(1, len(old)),
        'rewritten': len(rewritten), 'deleted': len(removed), 'inserted': len(added), 'moved': len(moved),
        'action_counts': dict(acts), 'facts': facts,
        'cohesion_counts': {k: counts[k] for k in ('polarity', 'modality', 'conjunction_relation', 'explicit_subject_inserted')},
        'off_style_before': sum(a.off_style for a in old_structure.annotations),
        'off_style_after': sum(a.off_style for a in new_structure.annotations),
        'dominant_style_before': old_structure.dominant_style, 'dominant_style_after': new_structure.dominant_style,
        'interpretation': 'Observed changes; no claim that an observed change is an error.'}


def normalize_saved_measurement(row):
    """Recompute the insertion-only count from saved diffs, without another model call.

    Initial integration-check files used new subject evidence as a broader count.
    This makes the final analysis definition identical for pilot and full-run rows.
    """
    if 'measurement' not in row:
        return row
    old = {u['sid']: u['text'] for p in row['initial_layout']['paragraphs'] for u in p['units']}
    new = {u['sid']: u['text'] for p in row['final_layout']['paragraphs'] for u in p['units']}
    facts = []
    for fact in row['measurement']['facts']:
        if fact['kind'] == 'explicit_subject_inserted':
            a, b = old[fact['sid']], new[fact['sid']]
            inserted = [b[c:d] for tag, _, _, c, d in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag == 'insert']
            if not any(fact['surface'] in piece for piece in inserted):
                continue
        facts.append(fact)
    row['measurement']['facts'] = facts
    row['measurement']['cohesion_counts']['explicit_subject_inserted'] = sum(
        f['kind'] == 'explicit_subject_inserted' for f in facts)
    row['measurement']['version'] = 'phase6_inserted_subject_diff_v2'
    return row
