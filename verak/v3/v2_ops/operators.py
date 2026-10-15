"""Record-driven v2 deletion and sentence fusion, with generation-only relations."""
from collections import Counter
from dataclasses import dataclass

from ..common import sha_text
from ..corrupt.document import frozen_lexicons, positional_changes, protected
from ..ko.coarse import coarse_conjunction

RELATIONS = {'RESULT': 'CAUSE', 'ADVERSATIVE': 'ADVERSATIVE', 'ADDITION': 'ADDITION', 'NONE': 'ADDITION'}
EXCLUDED_RELATIONS = {'EXAMPLE', 'RESTATEMENT'}
CONTENT_TAGS = {'NNG', 'NNP', 'NNB', 'NP', 'NR', 'VV', 'VA', 'VX', 'VCP', 'VCN',
                'MM', 'MAG', 'MAJ', 'XR', 'XPN', 'XSN', 'XSV', 'XSA', 'SN', 'SL', 'SH', 'NA'}


def conjunction(unit):
    return coarse_conjunction(unit.text, frozen_lexicons()['conjunction'])


def relation(unit):
    value = conjunction(unit)
    return 'NONE' if value is None else value['coarse_class'] if value['eligible'] else 'UNSUPPORTED'


def content_morphemes(unit):
    """Content/negation/modality morphemes; initial discourse conjunction is checked separately."""
    initial = conjunction(unit)
    cutoff = len(initial['form']) if initial and initial['eligible'] and unit.text.startswith(initial['form']) else 0
    return Counter((token.form, token.tag) for token in unit.tokens
                   if token.tag in CONTENT_TAGS and token.start >= cutoff)


def label_targets(document):
    wanted = {p.units[0].sid for p in document.paragraphs if p.units}
    if document.paragraphs:
        wanted.update(u.sid for u in document.paragraphs[-1].units)
    return [u.sid for u in document.units if u.sid in wanted]


def ending_prefix(unit):
    endings = [i for i, t in enumerate(unit.tokens) if t.tag == 'EF']
    if len(endings) != 1:
        raise ValueError('one_unquoted_final_EF_required')
    i = endings[0]
    if not i or any(t.tag not in {'SF', 'SS', 'SSC', 'SSO'} and (t.form, t.tag) != ('요', 'JX')
                    for t in unit.tokens[i+1:]):
        raise ValueError('nonterminal_EF')
    final, previous = unit.tokens[i], unit.tokens[i-1]
    if previous.tag not in {'VV', 'VA', 'VX', 'VCP', 'VCN', 'XSV', 'XSA', 'EP'}:
        raise ValueError('unsupported_final_stem')
    if protected(unit.text, final.start, len(unit.text)):
        raise ValueError('protected_final_span')
    prefix = (unit.text[:final.start] if final.start >= previous.end
              else unit.text[:previous.start] + previous.form)
    if not prefix or not 0xAC00 <= ord(prefix[-1]) <= 0xD7A3:
        raise ValueError('realized_hangul_stem_required')
    return prefix


def connective(prefix, source_class, variant=0):
    """This mapping is local to L_FUSE; it never changes v1 EC/L_CONN sets."""
    if source_class not in RELATIONS:
        raise ValueError('unsupported_boundary_relation')
    code = ord(prefix[-1]) - 0xAC00
    coda, vowel = code % 28, (code // 28) % 21
    if source_class == 'ADVERSATIVE':
        return prefix + '지만'
    if source_class in {'ADDITION', 'NONE'}:
        return prefix + ('고' if variant % 2 == 0 else '며' if coda in {0, 8} else '으며')
    # RESULT -> CAUSE. Use ordinary surfaces, then verify Bareun content retention.
    if prefix.endswith('하'):
        return prefix[:-1] + '해서'
    if not coda and vowel in {0, 4}:  # 가서 / 서서; uncontracted *가아서 is not used.
        return prefix + '서'
    if not coda and vowel == 18:  # 쓰다 -> 써서; choose the unmarked 어 allomorph.
        last = chr(ord(prefix[-1]) + (4 - 18) * 28)
        return prefix[:-1] + last + '서'
    return prefix + ('아서' if vowel in {0, 8} else '어서')


@dataclass(frozen=True)
class Fusion:
    paragraph: str
    position: int
    sids: tuple
    source_classes: tuple


def fusion_candidates(source):
    values, counts = [], Counter()
    annotations = {a.sid: a for a in source.structure().annotations}
    for paragraph in source.paragraphs:
        for length in (2, 3):
            for start in range(len(paragraph.units) - length + 1):
                units = paragraph.units[start:start+length]
                counts['candidate_windows'] += 1
                classes = tuple(relation(u) for u in units[1:])
                excluded = set(classes) & EXCLUDED_RELATIONS
                if excluded:
                    counts['excluded_EXAMPLE_or_RESTATEMENT'] += 1
                    for cls in excluded:
                        counts['excluded_' + cls] += 1
                    continue
                if any(c not in RELATIONS for c in classes):
                    counts['excluded_unsupported_conjunction'] += 1
                    continue
                if any(annotations[u.sid].multi_unit for u in units):
                    counts['excluded_multi_unit'] += 1
                    continue
                try:
                    for unit in units[:-1]:
                        ending_prefix(unit)
                except ValueError as exc:
                    counts[str(exc)] += 1
                    continue
                values.append(Fusion(paragraph.pid, start, tuple(u.sid for u in units), classes))
    counts['eligible_windows'] = len(values)
    return values, dict(counts)


def record(source, changed, *, op, sids, target, params):
    original = {u.sid: u.text for u in source.units if u.sid in sids}
    return {'op': op, 'level': 'GLOBAL', 'owner': 'global', 'sids': list(sids),
        'original_text': original, 'corrupted_text': {u.sid: u.text for u in changed.units if u.sid in sids},
        'params': params, 'recovery_target': target, 'inverse': source.snapshot(),
        'coupled_changes': positional_changes(source.structure(), changed.structure()),
        'source_hash': sha_text(source.text), 'corrupted_hash': sha_text(changed.text)}


def delete_link(source, sid, label):
    if sid not in label_targets(source) or label not in {'topic', 'bridge', 'summary'}:
        raise ValueError('G_DEL_LINK requires an eligible cached Sol label')
    changed = source.clone()
    pi, si, unit = changed.locate(sid)
    paragraph = changed.paragraphs[pi]
    paragraph.units.pop(si)
    if not changed.units:
        raise ValueError('cannot_delete_entire_essay')
    target = {'paragraph': paragraph.pid, 'position': si, 'original': unit.text,
        'role': label, 'position_tolerance': 1, 'requires_insert_origin': True}
    return changed, record(source, changed, op='G_DEL_LINK', sids=[sid], target=target,
                           params={'deleted_label': label})


def fuse(source, candidate, analysis, *, variant=0):
    changed = source.clone()
    paragraph = next(p for p in changed.paragraphs if p.pid == candidate.paragraph)
    originals = paragraph.units[candidate.position:candidate.position+len(candidate.sids)]
    if tuple(u.sid for u in originals) != candidate.sids:
        raise ValueError('fusion_candidate_changed')
    pieces = []
    for i, unit in enumerate(originals):
        text = connective(ending_prefix(unit), candidate.source_classes[i], variant) if i < len(originals)-1 else unit.text
        if i:
            initial = conjunction(unit)
            if initial:
                if not initial['eligible'] or candidate.source_classes[i-1] not in RELATIONS:
                    raise ValueError('unsupported_removed_conjunction')
                if not text.startswith(initial['form']):
                    raise ValueError('conjunction_surface_not_at_start')
                text = text[len(initial['form']):].lstrip(' ,，')
        pieces.append(text)
    text = ' '.join(pieces)
    parsed = analysis.pieces(text)
    if len(parsed) != 1 or parsed[0][0] != text:
        raise ValueError('fused_surface_not_one_bareun_sentence')
    merged = originals[0]
    original_content = sum((content_morphemes(u) for u in originals), Counter())
    merged.text, merged.tokens = text, parsed[0][1]
    if content_morphemes(merged) != original_content:
        raise ValueError('fused_surface_changes_content_morphemes')
    paragraph.units[candidate.position:candidate.position+len(candidate.sids)] = [merged]
    analysis.refresh(changed, {paragraph.pid})
    if content_morphemes(merged) != original_content:
        raise ValueError('paragraph_analysis_changes_fused_content')
    target = {'paragraph': paragraph.pid, 'position': candidate.position,
        'source_sids': list(candidate.sids), 'source_classes': list(candidate.source_classes),
        'source_sentences': [source.locate(sid)[2].text for sid in candidate.sids],
        'boundaries': len(candidate.sids)-1}
    return changed, record(source, changed, op='L_FUSE', sids=candidate.sids, target=target,
        params={'source_classes': list(candidate.source_classes),
                'generation_classes': [RELATIONS[c] for c in candidate.source_classes],
                'generation_only_v2_addition': any(c in {'NONE', 'ADDITION'} for c in candidate.source_classes)})
