"""Pure v4 evaluation candidate matching; never imported by v1/v2 rewards."""


def _paragraphs(snapshot):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('paragraphs'), list):
        raise ValueError('A document snapshot with paragraphs is required')
    rows = snapshot['paragraphs']
    ids = [p['pid'] for p in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate paragraph IDs')
    return rows


def link_recovery_candidates(source, corrupted, final, record, inserted_ids, *, occupied=(), version):
    """Return candidate units, not semantic grades or a new reward definition.

    Ordinary matching preserves the original paragraph and position +/-1 rule.
    If this record emptied its source paragraph, also admit an inserted unit at
    the *immediate source neighbor's* adjacent boundary: last in the preceding
    paragraph or first in the following paragraph. No skipping empty neighbors,
    distant paragraphs, arbitrary within-neighbor positions, or non-inserted units.
    The original source paragraph need not remain as an empty wrapper in final.
    """
    if version != 'v4':
        raise ValueError('Adjacent-boundary evaluation is v4 only')
    if record.get('op') != 'G_DEL_LINK':
        raise ValueError('This matcher handles G_DEL_LINK only')
    target = record['recovery_target']
    pid, position = target['paragraph'], target['position']
    if type(position) is not int or position < 0:
        raise ValueError('Invalid source position')
    before = _paragraphs(source)
    damaged = _paragraphs(corrupted)
    after = _paragraphs(final)
    source_index = next((i for i, p in enumerate(before) if p['pid'] == pid), None)
    if source_index is None:
        raise ValueError('The recorded source paragraph is absent')
    original = before[source_index]
    if not original['units'] or position >= len(original['units']):
        raise ValueError('Recorded source position is absent')
    deleted = set(record.get('sids', []))
    if original['units'][position]['sid'] not in deleted:
        raise ValueError('Deleted source sentence does not match its recorded position')
    damaged_paragraph = next((p for p in damaged if p['pid'] == pid), None)
    emptied = (all(u['sid'] in deleted for u in original['units']) and
               (damaged_paragraph is None or not damaged_paragraph['units']))
    previous = before[source_index-1]['pid'] if emptied and source_index else None
    following = before[source_index+1]['pid'] if emptied and source_index+1 < len(before) else None
    allowed = set(inserted_ids) - set(occupied)
    results = []
    for paragraph in after:
        for index, unit in enumerate(paragraph['units']):
            if unit['sid'] not in allowed:
                continue
            rule = None
            if paragraph['pid'] == pid and abs(index-position) <= 1:
                rule = 'original_paragraph_position_plus_minus_1'
            elif emptied and paragraph['pid'] == previous and index == len(paragraph['units'])-1:
                rule = 'empty_source_previous_paragraph_end'
            elif emptied and paragraph['pid'] == following and index == 0:
                rule = 'empty_source_following_paragraph_start'
            if rule:
                results.append({'sid': unit['sid'], 'text': unit['text'], 'paragraph': paragraph['pid'],
                    'position': index, 'rule': rule, 'source_paragraph': pid,
                    'source_position': position, 'source_paragraph_emptied': emptied})
    return results
