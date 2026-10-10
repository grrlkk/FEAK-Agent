"""Exact, conservative v4.2 -> v1 reward projection using saved provenance.

This module does not analyze text, call models, modify a corpus record, or change
the v1 reward functions. An unrepresentable split is unknown, never zero reward.
"""
from copy import deepcopy
from dataclasses import asdict, replace

from verak.src.schemas import Token
from verak.v3.corrupt.document import Document, Paragraph, Unit
from .common import sha_text


class ProjectionUnknown(ValueError):
    pass


def save_document(document):
    return {'layout': document.snapshot(), 'tokens': {u.sid: [asdict(t) for t in u.tokens] for u in document.units},
            'text_sha256': sha_text(document.text)}


def restore_document(value):
    state, tokens = value['layout'], value['tokens']
    ids = [u['sid'] for p in state['paragraphs'] for u in p['units']]
    if len(set(ids)) != len(ids) or set(ids) != set(tokens):
        raise ValueError('Saved document token/SID inventory differs')
    doc = Document([Paragraph(p['pid'], [Unit(u['sid'], u['text'],
        [Token(**t) for t in tokens[u['sid']]], u['leading']) for u in p['units']])
        for p in state['paragraphs']], list(state['gaps']), state['tail'], state['analyzer_version'])
    if sha_text(doc.text) != value['text_sha256']:
        raise ValueError('Saved document text hash differs')
    if any(t.start < 0 or t.end < t.start or t.end > len(u.text) for u in doc.units for t in u.tokens):
        raise ValueError('Saved morpheme offset is outside its sentence')
    return doc


def ancestors(sid, lineage):
    result, seen = [sid], {sid}
    while sid in lineage and lineage[sid]['parent_sid'] != sid:
        sid = lineage[sid]['parent_sid']
        if sid in seen:
            raise ProjectionUnknown('cyclic_sentence_lineage')
        seen.add(sid)
        result.append(sid)
    return result


def root_sid(sid, normalization, lineage):
    original = normalization['sid_mapping']
    for ancestor in ancestors(sid, lineage):
        if ancestor in original:
            return ancestor
    return ancestors(sid, lineage)[-1]


def public_document(document):
    """Reading-order aliases hide source positions, as the v1 reset already does."""
    result = document.clone()
    sentences, paragraphs = {}, {}
    number = 0
    for index, paragraph in enumerate(result.paragraphs, 1):
        public_pid = f'P{index}'
        paragraphs[public_pid] = paragraph.pid
        paragraph.pid = public_pid
        for unit in paragraph.units:
            number += 1
            public_sid = f'S{number}'
            sentences[public_sid] = unit.sid
            unit.sid = public_sid
    if any(sid.startswith('V42N_') for sid in sentences.values()):
        raise ValueError('Private source IDs conflict with the new-unit reward namespace')
    return result, {'public_to_private_sentences': sentences, 'public_to_private_paragraphs': paragraphs}


def private_id(value, aliases):
    if aliases is None:
        return value
    if value in aliases['public_to_private_sentences']:
        return aliases['public_to_private_sentences'][value]
    if value in aliases['public_to_private_paragraphs']:
        return aliases['public_to_private_paragraphs'][value]
    if value in {'START', '', None}:
        return value
    return 'V42N_'+value


def private_document(document, aliases):
    """Undo only ID aliases; the normalized text and current positions stay exact."""
    if aliases is None:
        return document.clone()
    result = document.clone()
    for paragraph in result.paragraphs:
        paragraph.pid = aliases['public_to_private_paragraphs'][paragraph.pid]
        for unit in paragraph.units:
            unit.sid = private_id(unit.sid, aliases)
    if len({u.sid for u in result.units}) != len(result.units) or result.text != document.text:
        raise ProjectionUnknown('private_ID_mapping_is_not_lossless')
    return result


def project_document(document, normalization, lineage=None):
    """Join complete, ordered, adjacent fragments back to their original SID.

    Whole original-unit deletion is representable. Partial deletion of initial
    fragments, interleaving, reversed fragments, or moving them to different
    paragraphs is not. New INSERT IDs remain new IDs. Whitespace and all current
    text/tokens are preserved exactly; only the reward unit boundaries change.
    """
    lineage = deepcopy(lineage if lineage is not None else normalization['lineage'])
    initial = normalization['sid_mapping']
    normalized_ids = [u['sid'] for p in normalization['normalized_layout']['paragraphs'] for u in p['units']]
    if set(normalized_ids) != {s for ids in initial.values() for s in ids}:
        raise ProjectionUnknown('normalization_inventory_mismatch')
    positions, groups = {}, {}
    for pi, paragraph in enumerate(document.paragraphs):
        for si, unit in enumerate(paragraph.units):
            parent = root_sid(unit.sid, normalization, lineage)
            positions[unit.sid] = (pi, si)
            groups.setdefault(parent, []).append(unit)
    for parent, units in groups.items():
        coords = [positions[u.sid] for u in units]
        if len({pi for pi, _ in coords}) != 1:
            raise ProjectionUnknown('fragments_cross_paragraphs:' + parent)
        indexes = [si for _, si in coords]
        if indexes != list(range(indexes[0], indexes[0]+len(indexes))):
            raise ProjectionUnknown('fragments_not_contiguous:' + parent)
        if parent in initial:
            expected = initial[parent]
            # An initially split fragment may itself be split by a later EDIT.
            # Its surviving root ID must remain; an absent initial fragment is a
            # partial DELETE, which v1 cannot represent as one stable sentence.
            if not set(expected) <= positions.keys():
                raise ProjectionUnknown('initial_fragment_partially_deleted:' + parent)
            origins = []
            for unit in units:
                origin = next((a for a in ancestors(unit.sid, lineage) if a in expected), None)
                if origin is None:
                    raise ProjectionUnknown('fragment_origin_missing:' + unit.sid)
                if not origins or origins[-1] != origin:
                    origins.append(origin)
            if origins != expected:
                raise ProjectionUnknown('fragment_order_changed:' + parent)
    projected = document.clone()
    for paragraph in projected.paragraphs:
        merged, used = [], set()
        for unit in paragraph.units:
            parent = root_sid(unit.sid, normalization, lineage)
            if parent in used:
                continue
            used.add(parent)
            parts = groups[parent]
            text, tokens = '', []
            for index, part in enumerate(parts):
                if index:
                    text += part.leading
                offset = len(text)
                text += part.text
                tokens.extend(replace(t, start=t.start+offset, end=t.end+offset) for t in part.tokens)
            merged.append(Unit(parent, text, tokens, parts[0].leading))
        paragraph.units = merged
    if projected.text != document.text:
        raise ProjectionUnknown('projection_changed_current_text')
    ids = [u.sid for u in projected.units]
    if len(set(ids)) != len(ids):
        raise ProjectionUnknown('projection_duplicate_original_ID')
    return projected


def reward_actions(actions, normalization, lineage, aliases=None):
    """Normalize trace fields for v1 attribution; preserve the untouched v4 trace."""
    result = []
    def sid(value):
        return private_id(root_sid(value, normalization, lineage), aliases) if isinstance(value, str) else value
    for action in actions:
        item = deepcopy(action)
        value = action.get('value') or {}
        name = action.get('action', 'INVALID')
        item['v42_action'] = name
        item['changed_sids'] = sorted({sid(s) for s in action.get('changed_sids', [])})
        if name == 'MOVE':
            position = value.get('to', {})
            side, anchor = next(iter(position.items())) if isinstance(position, dict) and len(position) == 1 else ('invalid', '')
            item['args'] = {'target': sid(value.get('sentence', '')), 'position': side+':'+str(sid(anchor))}
        elif name == 'DELETE':
            item.update(action='EDIT', args={'target': str(sid(value.get('sentence', ''))), 'new_text': ''})
        elif name == 'INSERT':
            # START remains a structural insertion marker for the selection gate;
            # changed_sids, not the anchor, owns the v1 morpheme attribution.
            item.update(action='EDIT', args={'target': 'after:'+str(sid(value.get('after', ''))),
                                            'new_text': value.get('text', '')})
        elif name == 'EDIT':
            item['args'] = {'target': str(sid(value.get('sentence', '')))+':'+str(value.get('old', '')),
                            'new_text': value.get('new', '')}
        else:
            item['args'] = {k: v for k, v in value.items() if k != 'action'}
        result.append(item)
    return result


def structural_split_attempt(raw, document):
    """Audit a Revision EDIT even if rejected/undone; never gate its execution."""
    from .content_env import parse_json
    from .environment_v42 import sentence_spans
    try:
        value = parse_json(raw)
        if not isinstance(value, dict) or value.get('action') != 'EDIT':
            return False
        if any(not isinstance(value.get(k), str) for k in ('sentence', 'old', 'new')):
            return False
        unit = document.locate(value['sentence'])[2]
        if not value['old'] or unit.text.count(value['old']) != 1:
            return False
        text = unit.text.replace(value['old'], value['new'], 1)
        return len(sentence_spans(text)) > len(sentence_spans(unit.text))
    except (KeyError, ValueError, TypeError):
        return False


def initial_recovery(source, raw_corrupted, normalized_source, normalized_corrupted, records):
    from verak.v3.reward.recovery import recover_records
    before = recover_records(source, raw_corrupted, records, corrupted=raw_corrupted)
    after = recover_records(normalized_source, normalized_corrupted, records, corrupted=normalized_corrupted)
    return [{'record_id': a['record_id'], 'op': a['op'], 'raw_main': b['main'], 'normalized_initial_main': a['main'],
             'environment_recovered_full': b['main'] < 1. and a['main'] == 1.,
             'environment_main_gain': a['main']-b['main']}
            for b, a in zip(before, after)]
