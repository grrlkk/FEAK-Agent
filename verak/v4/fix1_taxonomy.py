"""File-only exhaustive worker rejection audit; never generates new actions."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path

from .common import REPO, read_json, write_json, file_sha
from .environment_v45 import match_span, occurrences, whitespace_view
from .environment_v46 import terminal_only
from .render_v43 import assert_rendered

ROOT = REPO / 'verak/v4/outputs/fix1'
TYPES = ('span_not_found', 'span_ambiguous', 'no_op_edit',
         'boundary_terminal_only', 'boundary_other', 'other')
NEAR_POLICY = 'Minimum character Levenshtein distance to any nonempty sentence substring; old length >=4, distance <=max(1,floor(.15*max_length)); heuristic, not a morphological judgment.'


def nearest_span(text, old):
    """Exact substring edit-distance DP, with free text prefixes and suffixes."""
    if not old or not text:
        return None
    prev = [(0, j) for j in range(len(text)+1)]
    for i, char in enumerate(old, 1):
        cur = [(i, 0)]
        for j, other in enumerate(text, 1):
            choices = [(prev[j-1][0]+(char != other), prev[j-1][1]),
                       (prev[j][0]+1, prev[j][1]), (cur[j-1][0]+1, cur[j-1][1])]
            cur.append(min(choices, key=lambda x: (x[0], abs(j-x[1]-i), x[1])))
        prev = cur
    candidates = [(distance, start, end) for end, (distance, start) in enumerate(prev) if end > start]
    distance, start, end = min(candidates, key=lambda x: (x[0], abs(x[2]-x[1]-len(old)), x[1]))
    real = text[start:end]
    differences = [{'op': op, 'real': real[a:b], 'requested': old[c:d],
                    'real_offsets': [a, b], 'requested_offsets': [c, d]}
                   for op, a, b, c, d in SequenceMatcher(None, real, old, autojunk=False).get_opcodes()
                   if op != 'equal']
    return {'span': real, 'offsets': [start, end], 'distance': distance,
            'near_copy': len(old) >= 4 and distance <= max(1, int(.15*max(len(old), len(real)))),
            'differences_real_to_requested': differences}


def classify(action, payload, *, near=True):
    value = action.get('value') or {}
    texts = {s['id']: s['text'] for p in payload.get('paragraphs', []) for s in p['sentences']}
    sentence = texts.get(value.get('sentence'))
    error = action.get('error', '')
    info = {'type': 'other', 'error': error, 'action': value, 'sentence_text': sentence}
    if action.get('action') == 'EDIT' and isinstance(sentence, str) and isinstance(value.get('old'), str):
        old = value['old']; span, diagnostic = match_span(sentence, old)
        info.update(span_match=diagnostic, old_characters=len(old), old_eojeols=len(old.split()))
        if '부분 문자열' in error:
            exact = occurrences(sentence, old)
            normalized = occurrences(whitespace_view(sentence)[0], whitespace_view(old)[0]) if old.strip() else []
            info['type'] = 'span_ambiguous' if max(len(exact), len(normalized)) > 1 else 'span_not_found'
            if info['type'] == 'span_not_found' and near:
                info['nearest'] = nearest_span(sentence, old)
                info['exact_in_other_sentences'] = [sid for sid, text in texts.items()
                    if sid != value.get('sentence') and old and old in text]
            if info['type'] == 'span_ambiguous':
                info['ambiguous_short'] = len(old.split()) <= 1
                info['occurrences'] = exact
        elif '바꾸지 않는 행동' in error:
            info['type'] = 'no_op_edit'
        elif '종결부호' in error or '문장 경계' in error or '두 문장까지만' in error:
            after = None if span is None else sentence[:span[0]]+value.get('new', '')+sentence[span[1]:]
            info['type'] = 'boundary_terminal_only' if after is not None and terminal_only(sentence, after) else 'boundary_other'
            info['attempted_sentence'] = after
    if info['type'] == 'other':
        info['subtype'] = ('search_dependency' if 'ModuleNotFoundError' in error else
            'scope' if '배정된' in error else 'mask' if '가림표시' in error else
            'no_op_other' if '바꾸지 않는 행동' in error else 'other')
    return info


def rejection_rows(attempt, path, *, near=True):
    calls = attempt.get('calls', [])
    rows = []
    for index, call in enumerate(calls):
        if call['role'] not in ('revision', 'korean') or call['action']['valid']:
            continue
        prefix = read_json(Path(call['prefix']['path']))
        if file_sha(call['prefix']['path']) != call['prefix']['sha256']:
            raise ValueError('Saved request changed')
        payload = assert_rendered(prefix['messages'])
        following = next((c for c in calls[index+1:] if c['role'] == call['role'] and c['stage'] == call['stage']), None)
        if following is None:
            stages = attempt.get('stages', [])
            stage = stages[call['stage']-1] if isinstance(call['stage'], int) else {}
            next_kind = stage.get('ending', 'no_next_action')
        elif following['action'].get('action') == 'STOP':
            next_kind = 'STOP'
        elif following['raw'].strip() == call['raw'].strip() or (
                call['action'].get('value') is not None and following['action'].get('value') == call['action']['value']):
            next_kind = 'repeat'
        else:
            current = call['action'].get('value') or {}
            after = following['action'].get('value') or {}
            next_kind = 'retry' if (current.get('action') == after.get('action')
                and current.get('sentence') == after.get('sentence')) else 'other_action'
        rows.append({'episode_path': str(path), 'case_id': attempt['case_id'], 'role': call['role'],
            'call_index': index, 'stage': call['stage'], 'next': next_kind,
            'next_action': following['action'] if following else None,
            'prefix': call['prefix'], **classify(call['action'], payload, near=near)})
    return rows


def summary(rows):
    result = {}
    for role in ('revision', 'korean'):
        result[role] = {}
        for kind in TYPES:
            subset = [r for r in rows if r['role'] == role and r['type'] == kind]
            result[role][kind] = {'count': len(subset), 'next': dict(Counter(r['next'] for r in subset)),
                'near_copy': sum(bool((r.get('nearest') or {}).get('near_copy')) for r in subset),
                'short_ambiguous': sum(r.get('ambiguous_short', False) for r in subset)}
    return result


def audit():
    paths = sorted((REPO/'verak/v4/outputs/train44').glob('**/attempts/*.json'))
    paths += sorted((REPO/'verak/v4/outputs/smoke45').glob('**/attempts/*.json'))
    if len(paths) != 101:
        raise ValueError(f'Saved diagnostic inventory differs: {len(paths)}')
    rows = []; inventory = []; by_run = defaultdict(list)
    for path in paths:
        attempt = read_json(path)
        found = rejection_rows(attempt, path)
        rows.extend(found)
        group = str(path.parent.parent.relative_to(REPO/'verak/v4/outputs'))
        by_run[group].extend(found)
        inventory.append({'path': str(path), 'sha256': file_sha(path), 'rejections': len(found)})
    examples = {}
    for kind in TYPES:
        unique = {}
        for r in rows:
            if r['type'] == kind:
                signature = json.dumps([r['role'], r['sentence_text'], r['action'], r['error']], ensure_ascii=False, sort_keys=True)
                unique.setdefault(signature, r)
        candidates = list(unique.values())
        if kind == 'span_not_found':
            candidates.sort(key=lambda r: (not (r.get('nearest') or {}).get('near_copy', False), r['role'], r['case_id']))
        examples[kind] = candidates[:5]
    value = {'episodes': len(paths), 'rejections': len(rows), 'inventory': inventory,
        'near_copy_policy': NEAR_POLICY, 'summary': summary(rows),
        'by_run': {k: summary(v) for k, v in by_run.items()}, 'rows': rows,
        'scope': 'All saved executions: 30 original V4_TRAIN + 11 SEARCH-runtime rechecks + 60 V4_SMOKE2; not 101 distinct essays.',
        'examples': examples,
        'GPU_calls': 0, 'external_API_calls': 0}
    write_json(ROOT/'taxonomy.json', value)
    return value


if __name__ == '__main__':
    result = audit()
    print(json.dumps({k: result[k] for k in ('episodes', 'rejections', 'summary')}, ensure_ascii=False))
