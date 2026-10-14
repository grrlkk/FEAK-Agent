"""Lossless, one-current-snapshot v4.3 observations; no historical essay copies."""
from copy import deepcopy
import json

from .content_env import dumps

VERSION = 'v4.3_current_snapshot_20261010'
SNAPSHOT_KEYS = ('paragraphs', 'korean_markers', 'declared_relations',
                 'sentence_lineage', 'rendering', 'source_citations')


def pack_payload(payload):
    """Factor repeated notices and handoff journal entries, without truncation.

    Paragraph rows are [paragraph ID, [[sentence ID, full text], ...]]. The
    remaining profile/fact fields retain their values exactly. References index
    this same observation; no prior conversation state is required to read it.
    """
    value = deepcopy(payload)
    if 'current_snapshot' in value or 'renderer_version' in value:
        raise ValueError('Input must be an unpacked current state')
    snapshot = {key: value.pop(key) for key in SNAPSHOT_KEYS if key in value}
    snapshot['paragraphs'] = [[p['id'], [[s['id'], s['text']] for s in p['sentences']]]
                              for p in snapshot['paragraphs']]
    notices = []
    def notice_index(item):
        if item not in notices:
            notices.append(deepcopy(item))
        return notices.index(item)
    def factor(item):
        if isinstance(item, list):
            return [factor(x) for x in item]
        if not isinstance(item, dict):
            return item
        return {key: {'$notice_refs': [notice_index(x) for x in val]}
                if key in {'marker_change_notices', 'notices'} and isinstance(val, list)
                else factor(val) for key, val in item.items()}
    handoff = value.get('revision_handoff')
    journal = value.get('work_journal', [])
    if isinstance(handoff, dict) and 'action_log' in handoff:
        refs, cursor = [], 0
        for action in handoff['action_log']:
            found = next((i for i in range(cursor, len(journal))
                          if journal[i].get('role') == 'revision'
                          and journal[i].get('value') == action['action']
                          and journal[i]['valid'] == action['valid']), None)
            if found is None:
                raise ValueError('Handoff must refer to the complete public journal')
            refs.append(found); cursor = found + 1
        handoff['action_log'] = {'$journal_refs': refs}
    value = factor(value)
    result = {'renderer_version': VERSION, 'current_snapshot': factor(snapshot),
              **value, 'notice_pool': notices}
    if unpack_payload(result) != payload:
        raise ValueError('Current-state renderer lost information')
    return result


def unpack_payload(packed):
    """Exact inverse used by audits; repeated occurrence references are retained."""
    if packed.get('renderer_version') != VERSION:
        raise ValueError('Unknown renderer version')
    value = deepcopy(packed)
    value.pop('renderer_version'); notices = value.pop('notice_pool')
    def expand(item):
        if isinstance(item, list):
            return [expand(x) for x in item]
        if not isinstance(item, dict):
            return item
        if set(item) == {'$notice_refs'}:
            return [deepcopy(notices[i]) for i in item['$notice_refs']]
        return {key: expand(val) for key, val in item.items()}
    value = expand(value)
    snapshot = value.pop('current_snapshot')
    snapshot['paragraphs'] = [{'id': pid, 'sentences': [{'id': sid, 'text': text}
                             for sid, text in sentences]} for pid, sentences in snapshot['paragraphs']]
    value.update(snapshot)
    handoff = value.get('revision_handoff')
    if isinstance(handoff, dict) and isinstance(handoff.get('action_log'), dict):
        refs = handoff['action_log']
        if set(refs) != {'$journal_refs'}:
            raise ValueError('Invalid handoff journal references')
        handoff['action_log'] = [{'action': deepcopy(value['work_journal'][i]['value']),
                                'valid': value['work_journal'][i]['valid']}
                               for i in refs['$journal_refs']]
    return value


def render(payload, system):
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': dumps(pack_payload(payload))}]


def assert_rendered(messages):
    if len(messages) != 2 or messages[0]['role'] != 'system' or messages[1]['role'] != 'user':
        raise ValueError('v4.3 requires exactly one current observation')
    packed = json.loads(messages[1]['content'])
    return unpack_payload(packed)
