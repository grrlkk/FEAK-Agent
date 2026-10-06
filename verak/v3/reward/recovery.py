"""Calibrated partial credit for a reconstructed supporting sentence."""

import math
from .overedit import mapped_span

ACTIVE_OPERATORS = {'G_PARA_SWAP', 'G_SENT_MOVE', 'G_DELETE_SUPPORT', 'G_OFFTOPIC',
                    'L_CONN', 'L_POLARITY', 'L_CONJ', 'L_SUBJ_INSERT', 'L_REGISTER', 'L_SPACING'}


def graded_similarity(similarity, threshold, margin=0.05):
    """Zero at tau-margin, one at tau+margin, linear in between."""
    if not all(math.isfinite(v) for v in (similarity, threshold, margin)):
        raise ValueError("Similarity, threshold and margin must be finite")
    if not -1 <= similarity <= 1 or not -1 <= threshold <= 1 or margin <= 0:
        raise ValueError("Cosine similarity/threshold or margin out of range")
    if similarity <= threshold - margin:
        return 0.0
    if similarity >= threshold + margin:
        return 1.0
    return (similarity - (threshold - margin)) / (2 * margin)


def deletion_recovery(similarity, threshold, *, at_deletion_site, margin=0.05):
    """Similarity earns credit only for a sentence placed at the deletion site.

    The future environment must establish placement from its stable sentence IDs;
    this component never searches the whole essay for a convenient matching line.
    A gap left empty has similarity=None and receives zero credit.
    """
    if not at_deletion_site or similarity is None:
        return 0.0
    return graded_similarity(similarity, threshold, margin)


def _at_site(record, ann):
    sid = record['sids'][0]
    target = record['recovery_target']
    original = record['original_text'][sid]
    return mapped_span(original, ann.text, *target['site'])


def main_recovery(source, final, record, *, similarity=None, tau=None, annotations=None,
                  excluded_reconstruction_ids=()):
    """Record-derived targets; no antecedent identity and no hidden-answer prompts."""
    op, target = record['op'], record['recovery_target']
    if op not in ACTIVE_OPERATORS:
        raise ValueError(f'Unsupported active operator: {op}')
    annotations = annotations or {a.sid: a for a in final.structure().annotations}
    sid = record['sids'][0]
    ann = annotations.get(sid)
    if op == 'G_PARA_SWAP':
        order = target['paragraph_ids']
        positions = {p.pid: i for i,p in enumerate(final.paragraphs)}
        pairs = [(a,b) for i,a in enumerate(order) for b in order[i+1:]]
        # Concordant-pair fraction is Kendall tau mapped to [0,1]. Missing pairs get zero.
        return sum(a in positions and b in positions and positions[a] < positions[b]
                   for a,b in pairs) / len(pairs) if pairs else 1.
    if op == 'G_SENT_MOVE':
        if ann is None or ann.paragraph != target['paragraph']:
            return 0.
        pi, si, _ = source.locate(sid)
        neighbors = {u.sid for j,u in enumerate(source.paragraphs[pi].units) if abs(j-si)==1}
        pi, si, _ = final.locate(sid)
        adjacent = {u.sid for j,u in enumerate(final.paragraphs[pi].units) if abs(j-si)==1}
        return 1. if neighbors & adjacent or not neighbors else .5
    if op == 'G_DELETE_SUPPORT':
        source_ids = {u.sid for u in source.units}
        candidates = [u for p in final.paragraphs if p.pid == target['paragraph']
                      for i,u in enumerate(p.units) if abs(i-target['position']) <= 1
                      and u.sid not in excluded_reconstruction_ids
                      and (u.sid == sid or u.sid not in source_ids)]
        if any(u.text == target['original'] for u in candidates):
            return 1.
        if not candidates:
            return 0.
        if similarity is None or tau is None:
            raise ValueError('Reconstruction recovery requires calibrated similarity and tau')
        return max(deletion_recovery(similarity(target['original'], u.text), tau,
                                     at_deletion_site=True) for u in candidates)
    if op == 'G_OFFTOPIC':
        inserted = record['corrupted_text'][target['inserted_id']]
        return float(all(u.sid != target['inserted_id'] and u.text != inserted for u in final.units))
    if ann is None:
        return 0.
    if op == 'L_REGISTER':
        return float(ann.final_ending is not None and ann.style == target['style'])
    if op == 'L_CONJ':
        return float(bool(ann.initial_conj and ann.initial_conj['eligible'] and
                          ann.initial_conj['coarse_class'] == target['coarse_class']))
    if op == 'L_SUBJ_INSERT':
        import re
        # Only the inserted sentence-initial subject, not identical words elsewhere.
        return float(not re.match(r'^\s*'+re.escape(target['inserted_subject'])+r'(?=\s|[,，]|$)', ann.text))
    site = _at_site(record, ann)
    if site is None:
        return 0.
    a,b = site
    if op == 'L_CONN':
        return float(any(c['eligible'] and c['kind']=='EC' and
            c['coarse_class']==target['coarse_class'] and c['span'][1]-ann.start == b
            and c['span'][0]-ann.start <= a for c in ann.connectives))
    return float(ann.text[a:b] == target['original'])


def coupled_recovery(record, annotations, source_annotations=None, *, blocked_subject_ids=()):
    details = []
    for change in record.get('coupled_changes', []):
        ann = annotations.get(change['sid'])
        predecessor = change.get('recovery_predecessor_id',change['previous_before'])
        restored = ann is not None and ann.predecessor_id == predecessor
        if change['kind'] == 'DEP':
            # Conservative false omission alone is not evidence of an explicit subject.
            alternative = (ann is not None and bool(ann.subject_evidence)
                           and change['sid'] not in blocked_subject_ids)
            weight = .5
        elif change['kind'] == 'CONJ':
            # A class alone cannot certify compatibility with arbitrary new context.
            # Only source-class restoration at the recorded site is an automatic alternative.
            conj = ann.initial_conj if ann is not None else None
            original = (source_annotations or {}).get(change['sid'])
            original_conj = original.initial_conj if original is not None else None
            alternative = bool(conj and conj['eligible'] and original_conj and
                conj['coarse_class'] == change['coarse_class_before'] and
                conj['form'] != original_conj['form'])
            weight = 1.
        else:
            raise ValueError('Only CONJ/DEP coupled targets are supported')
        details.append({'sid': change['sid'], 'kind': change['kind'], 'weight': weight,
                        'recovered': float(restored or alternative),
                        'recovery_predecessor_id': predecessor,
                        'predecessor_restored': restored, 'alternative': alternative})
    denominator = sum(c['weight'] for c in details)
    value = sum(c['weight']*c['recovered'] for c in details)/denominator if denominator else None
    return value, details


def recover_records(source, final, records, *, similarity=None, tau=None, coupled_weight=.3):
    if not 0 <= coupled_weight <= 1:
        raise ValueError('Coupled weight out of range')
    annotations = {a.sid: a for a in final.structure().annotations}
    source_annotations = {a.sid: a for a in source.structure().annotations}
    off_topic_ids={r['recovery_target']['inserted_id'] for r in records if r['op']=='G_OFFTOPIC'}
    # A subject deliberately inserted as damage is not an agent's explicit-subject
    # repair. It must first be removed/replaced before it can earn DEP credit.
    bad_subject_ids={r['sids'][0] for r in records if r['op']=='L_SUBJ_INSERT' and
                     main_recovery(source,final,r,annotations=annotations)==0}
    results = []
    for i,record in enumerate(records):
        main = main_recovery(source, final, record, similarity=similarity, tau=tau, annotations=annotations,
                             excluded_reconstruction_ids=off_topic_ids)
        coupled, changes = coupled_recovery(record, annotations, source_annotations,
                                            blocked_subject_ids=bad_subject_ids)
        if changes and record['level'] != 'GLOBAL':
            raise ValueError('Coupled parts belong to GLOBAL records')
        combined = (1-coupled_weight)*main + coupled_weight*coupled if coupled is not None else main
        results.append({'record_id': record.get('record_id',str(i)), 'op': record['op'],
            'level': record['level'], 'main': main, 'coupled': coupled,
            'coupled_changes': changes, 'recovery': combined})
    return results
