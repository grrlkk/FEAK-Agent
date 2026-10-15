"""Salvage student MAP fields, then use the frozen training edge filter verbatim."""
from collections import Counter
from copy import deepcopy

from . import maps, orchestrator_v44
from .content_env import parse_json
from .prep3_maps import filter_edges, POLICY

VERSION = 'v4.5_map_salvage_20261012'


class EmptyMapError(ValueError):
    def __init__(self, diagnostics):
        super().__init__('MAP has no valid graph content after filtering')
        self.diagnostics = diagnostics


def filter_map(value, paragraphs):
    dropped = []
    defaulted = []
    def drop(field, original, reason, **where):
        dropped.append({'field': field, 'value': deepcopy(original), 'reason': reason, **where})
    if not isinstance(value, dict):
        raise EmptyMapError({'dropped_fields': [{'field': 'map', 'value': value, 'reason': 'not_object'}],
                             'dropped_edges': [], 'raw_edges': 0, 'kept_edges': 0})
    fields = {'genre', 'paragraph_roles', 'sentence_relations', 'paragraph_relations', 'off_topic'}
    for name in sorted(set(value)-fields):
        drop(name, value[name], 'unknown_map_field')
    genre = value.get('genre')
    if not isinstance(genre, str) or genre not in maps.GENRE_ROLES:
        drop('genre', genre, 'unknown_genre'); genre = None
    # The training filter indexes IDs through a genre-validated row. This local
    # placeholder is never observed or used as a genre prediction: restore None
    # below. No source/gold genre enters student validation.
    index_genre = genre or next(iter(maps.GENRE_ROLES))
    row = {'question': '', 'paragraphs': paragraphs, 'genre': index_genre}
    sids, pids, members = maps.indexes(row)
    role_names = set(maps.GENRE_ROLES[genre]) if genre else {r for rs in maps.GENRE_ROLES.values() for r in rs}
    roles = {p: {'paragraph': p, 'role': None, 'key_sentence': None} for p in pids}
    provided = set()
    raw_roles = value.get('paragraph_roles', [])
    if not isinstance(raw_roles, list):
        drop('paragraph_roles', raw_roles, 'not_array'); raw_roles = []
    for index, item in enumerate(raw_roles):
        if not isinstance(item, dict):
            drop('paragraph_roles', item, 'not_object', index=index); continue
        pid = item.get('paragraph')
        if not isinstance(pid, str) or pid not in roles:
            drop('paragraph_roles', item, 'unknown_paragraph_ID', index=index); continue
        if pid in provided:
            drop('paragraph_roles', item, 'duplicate_paragraph_entry', index=index); continue
        provided.add(pid)
        for name in sorted(set(item)-{'paragraph', 'role', 'key_sentence'}):
            drop('paragraph_roles.'+name, item[name], 'unknown_role_field', index=index)
        for name in ('role', 'key_sentence'):
            if name not in item:
                defaulted.append({'field': 'paragraph_roles.'+name, 'paragraph': pid, 'to': None})
        role = item.get('role')
        if role is not None and (not isinstance(role, str) or role not in role_names):
            drop('paragraph_roles.role', role, 'role_not_allowed_for_genre', index=index); role = None
        key = item.get('key_sentence')
        if key is not None and (not isinstance(key, str) or members.get(key) != pid):
            drop('paragraph_roles.key_sentence', key, 'key_not_in_paragraph', index=index); key = None
        roles[pid].update(role=role, key_sentence=key)
    for pid in pids:
        if pid not in provided:
            defaulted.append({'field': 'paragraph_roles', 'paragraph': pid, 'to': roles[pid]})
    clean = {'genre': index_genre, 'paragraph_roles': list(roles.values()), 'off_topic': []}
    for field in ('sentence_relations', 'paragraph_relations', 'off_topic'):
        entries = value.get(field, [])
        if not isinstance(entries, list):
            drop(field, entries, 'not_array'); entries = []
        clean[field] = deepcopy(entries)
    off_topic = []
    for index, sid in enumerate(clean['off_topic']):
        if not isinstance(sid, str) or sid not in sids:
            drop('off_topic', sid, 'unknown_sentence_ID', index=index)
        elif sid in off_topic:
            drop('off_topic', sid, 'duplicate_off_topic_ID', index=index)
        else:
            off_topic.append(sid)
    clean['off_topic'] = off_topic
    actual_roles = deepcopy(clean['paragraph_roles'])
    if genre is None:
        for item in clean['paragraph_roles']: item['role'] = None
    checked = filter_edges(clean, row)
    checked['map']['paragraph_roles'] = actual_roles
    checked['map']['genre'] = genre
    diagnostic = {k: deepcopy(v) for k, v in checked.items() if k != 'map'}
    diagnostic.update(metadata_policy=VERSION, training_edge_policy=POLICY['version'],
        dropped_fields=dropped, defaulted_fields=defaulted,
        dropped_fields_by_reason=dict(Counter(x['reason'] for x in dropped)),
        supplied_valid_paragraph_IDs=sorted(provided))
    # A supplied valid paragraph node (even explicitly uncertain role/key) is
    # graph content. Auto-filled placeholders or a genre alone are not evidence.
    if not (provided or checked['kept_edges'] or off_topic):
        raise EmptyMapError(diagnostic)
    return checked['map'], diagnostic


def validate(raw, paragraphs):
    value = parse_json(raw)
    if not isinstance(value, dict) or value.get('action') != 'MAP':
        return orchestrator_v44.validate(raw, paragraphs)
    if set(value) != {'action', 'map'}:
        raise ValueError('Invalid Orchestrator action or fields')
    clean, diagnostic = filter_map(value['map'], paragraphs)
    return {**value, 'map': clean}, diagnostic


def protected_map(value, paragraphs):
    # Same two-hop protection as v4.4; do not invent an extraction/intersection.
    row = {'question': '', 'genre': value['genre'] or next(iter(maps.GENRE_ROLES)), 'paragraphs': paragraphs}
    clean, diagnostic = maps.consensus(value, value, row, strict=False)
    clean['genre'] = value['genre']
    return clean, diagnostic
