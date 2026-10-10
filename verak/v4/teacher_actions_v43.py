"""Teacher collection only: retain raw output, execute/train its first action.

Inference parsing remains content_env.parse_json, with exactly one action. No
prose/fence repair or search for a later valid action is performed here.
"""
import json

from .content_env import dumps, parse_json

VERSION = 'v4.3_teacher_first_action_20261010'


def first_action(raw):
    diagnostic = {'version': VERSION, 'teacher_only': True, 'trimmed': False,
                  'kind': None, 'first_json_parsed': False,
                  'discarded_suffix_characters': 0, 'discarded_action_count': 0}
    if not isinstance(raw, str):
        raise TypeError('Teacher raw response must be a string')
    stripped = raw.lstrip()
    try:
        # raw_decode only locates the first complete JSON value. Strict parsing
        # of that exact slice still rejects duplicate keys and NaN/Infinity.
        _, end = json.JSONDecoder().raw_decode(stripped)
        value = parse_json(stripped[:end])
        diagnostic['first_json_parsed'] = True
        suffix = stripped[end:]
        if isinstance(value, dict) and set(value) == {'actions'} and isinstance(value['actions'], list):
            actions = value['actions']
            if not actions:
                return raw, diagnostic
            value = actions[0]
            diagnostic.update(trimmed=True, kind='actions_wrapper',
                              discarded_action_count=len(actions)-1)
        elif isinstance(value, list):
            if not value:
                return raw, diagnostic
            diagnostic.update(trimmed=True, kind='action_array', discarded_action_count=len(value)-1)
            value = value[0]
        if suffix.strip():
            # Only a JSON-object/array prefix is eligible, never a scalar or
            # arbitrary leading prose. The first action still faces all guards.
            if not isinstance(value, dict) or (not diagnostic['trimmed'] and suffix.lstrip()[0] not in '{['):
                return raw, diagnostic
            diagnostic.update(trimmed=True, kind=diagnostic['kind'] or 'multiple_values',
                              discarded_suffix_characters=len(suffix))
            cursor = suffix.lstrip(); count = 0
            while cursor:
                try:
                    _, stop = json.JSONDecoder().raw_decode(cursor)
                except ValueError:
                    diagnostic['suffix_contains_malformed_content'] = True
                    break
                count += 1; cursor = cursor[stop:].lstrip()
            diagnostic['discarded_action_count'] += count
        return dumps(value), diagnostic
    except (ValueError, TypeError) as exc:
        diagnostic['parse_error'] = type(exc).__name__ + ': ' + str(exc)
        return raw, diagnostic
