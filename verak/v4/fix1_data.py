"""Deterministic corrections from the frozen SFT training partition only."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import shutil

from .common import atomic_new, file_sha, read_json, sha_text
from .content_env import dumps
from .dryrun_common import tokenizer
from .full_data import ROOT as TRAIN, ref
from .fix1_taxonomy import ROOT
from .environment_v45 import occurrences, match_span
from .render_v43 import assert_rendered, render
from verak.v3.train.formatting import encode_labels, lengths

ROLES = ('korean', 'revision')
RECOVERY_PER_TYPE = 200
RECOVERY_STOP_PER_TYPE = 50
STOP_EXTRA = 500
ERROR = {'span_not_found': 'ActionError: 부분 문자열이 해당 문장에 정확히 한 번 있어야 합니다.',
         'span_ambiguous': 'ActionError: 부분 문자열이 해당 문장에 정확히 한 번 있어야 합니다.',
         'no_op_edit': 'ValueError: 글이나 관계를 바꾸지 않는 행동입니다. 고칠 것이 없으면 STOP 한다.'}


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sentence_map(payload):
    return {s['id']: s['text'] for p in payload['paragraphs'] for s in p['sentences']}


def widen(text, action):
    old, new = action['old'], action['new']
    span, _ = match_span(text, old)
    if span is None:
        raise ValueError('A selected teacher EDIT has no unique recoverable span')
    start, end = span
    actual = text[start:end]
    words = [m.span() for m in re.finditer(r'\S+', text)]
    left = next((i for i, (a, b) in enumerate(words) if b > start), None)
    right = next((i for i in range(len(words)-1, -1, -1) if words[i][0] < end), None)
    if left is None or right is None or left > right:
        # A whitespace-only correction includes both adjacent eojeols.
        left = max((i for i, (_, b) in enumerate(words) if b <= start), default=0)
        right = min((i for i, (a, _) in enumerate(words) if a >= end), default=len(words)-1)
    a = min(start, words[left][0]); b = max(end, words[right][1])
    while len(occurrences(text, text[a:b])) != 1:
        if left > 0:
            left -= 1; a = words[left][0]
        elif right+1 < len(words):
            right += 1; b = words[right][1]
        else:
            raise ValueError('The full sentence is not unique inside itself')
    replacement = text[a:start] + new + text[end:b]
    result = {**action, 'old': text[a:b], 'new': replacement}
    assert text[:start]+new+text[end:] == text[:a]+replacement+text[b:]
    assert len(occurrences(text, result['old'])) == 1
    assert a == 0 or text[a-1].isspace() or text[a].isspace()
    assert b == len(text) or text[b].isspace() or text[b-1].isspace()
    return result, {'original_span': [start, end], 'widened_span': [a, b],
                    'original_exact_text': actual, 'same_result': True}


def encode(sample, messages, target, tok, fix):
    result = deepcopy(sample)
    result['messages'] = deepcopy(messages)+[{'role': 'assistant', 'content': dumps(target)}]
    result.update(encode_labels(tok, result['messages'], last_assistant_only=True))
    prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if result['input_ids'][:len(prefix)] != prefix:
        raise ValueError('Training/inference prefix mismatch')
    result.update(prefix_tokens=len(prefix), total_tokens=len(result['input_ids']),
                  target_tokens=sum(x != -100 for x in result['labels']), fix1=fix)
    if len(prefix)+1024 > 8192 or result['total_tokens'] > 8192:
        raise OverflowError('Augmented sample exceeds unchanged 8192 context/reserve')
    if any(x != -100 for x in result['labels'][:len(prefix)]):
        raise ValueError('Observation supervision leak')
    return result


def dropped_copy(text, old):
    # Drop one real Korean character (often a particle/ending), not a fabricated
    # sentinel. Skip a mutation if it is still found elsewhere in the sentence.
    for i in reversed(range(len(old))):
        if not ('가' <= old[i] <= '힣'):
            continue
        candidate = old[:i]+old[i+1:]
        if len(candidate.strip()) >= 2 and not occurrences(text, candidate):
            return candidate
    return None


def rejected_action(payload, target, kind):
    texts = sentence_map(payload)
    if target['action'] == 'EDIT':
        sid = target['sentence']; text = texts[sid]; action = deepcopy(target)
    elif target['action'] == 'STOP' and target.get('status') == 'done':
        allowed = set(payload['allowed_scope'])
        choice = next(((sid, text) for sid, text in texts.items() if sid in allowed), None)
        if choice is None:
            return None
        sid, text = choice
        action = {'action': 'EDIT', 'sentence': sid, 'old': text, 'new': text}
    else:
        return None
    if kind == 'no_op_edit':
        action['new'] = action['old']
    elif kind == 'span_not_found':
        candidate = dropped_copy(text, action['old'])
        if candidate is None:
            return None
        action['old'] = candidate
    else:
        # A repeated short part of the intended quote, preferentially a Korean
        # syllable. This fails before any document mutation or boundary check.
        candidates = [m.group() for m in re.finditer(r'[가-힣]{1,3}', action['old'])]
        candidates += [c for c in action['old'] if '가' <= c <= '힣']
        candidate = next((c for c in candidates if len(occurrences(text, c)) > 1), None)
        if candidate is None:
            return None
        action['old'] = candidate
        if target['action'] == 'STOP':
            action['new'] = candidate+' '
    # Structural audit: the selected invalid action really has the stated fault.
    matches = occurrences(text, action['old'])
    assert ((kind == 'span_not_found' and len(matches) == 0)
            or (kind == 'span_ambiguous' and len(matches) > 1)
            or (kind == 'no_op_edit' and len(matches) == 1 and action['old'] == action['new']))
    return action


def after_rejection(payload, bad, role, kind):
    value = deepcopy(payload)
    if value['steps_left'] <= 1:
        return None
    # At most one new rejection; never manufacture a state after controller stop.
    if value.get('last_result', {}).get('controller_stop'):
        return None
    streak = 0
    for entry in reversed(value['work_journal']):
        if entry['role'] != role or entry['valid']:
            break
        streak += 1
    if streak >= 2:
        return None
    record = {'action': 'EDIT', 'value': deepcopy(bad), 'valid': False,
              'changed_sids': [], 'marker_change_notices': [], 'text_changed': False,
              'error': ERROR[kind]}
    turn = 1+sum(r['role'] == role for r in value['work_journal'])
    value['work_journal'].append({'role': role, 'turn': turn, **record})
    value['last_result'] = {k: v for k, v in record.items() if k != 'value'}
    value['steps_left'] -= 1
    if value['steps_left'] <= 2:
        value['notice'] = '남은 행동이 2개 이하입니다. 필요한 최소 수정 후 STOP으로 보고하세요.'
    # Rejected actions leave the actual essay/profile/relations/handoff untouched.
    for key in ('paragraphs', 'korean_markers', 'declared_relations', 'revision_handoff'):
        if key in payload:
            assert value[key] == payload[key]
    return value


def key(sample):
    return sha_text(dumps([sample['source_id'], sample['attempt'], sample['delegation'],
                           sample['turn'], sample['messages'][-1]['content']]))


def stage_key(sample):
    return (sample['source_id'], sample['attempt'], sample['role'], sample['delegation'],
            sample['kind'], sample['provenance']['path'])


def choose_diverse(candidates, count):
    ordered = sorted(candidates, key=key)
    selected = []; others = []; seen = set()
    for sample in ordered:
        if sample['source_group'] not in seen:
            selected.append(sample); seen.add(sample['source_group'])
        else:
            others.append(sample)
    return (selected+others)[:count]


def build():
    manifest_path = ROOT/'data/manifest.json'
    if manifest_path.exists():
        return verify()
    original = read_json(TRAIN/'data/manifest.json')
    tok = tokenizer(); result = {'version': 'fix1_deterministic_20261012',
        'source_manifest': ref(TRAIN/'data/manifest.json'), 'roles': {},
        'rules': {'widen': 'replace every train EDIT target, preserve exact resulting text; whole eojeols, extend left then right until unique',
            'recovery_per_type_per_role': RECOVERY_PER_TYPE,
            'recovery_STOP_quota_per_type': RECOVERY_STOP_PER_TYPE,
            'recovery_budget': 'EDIT augmentations only from teacher stages ending STOP(done) with >=2 steps left, so one added rejection still permits the demonstrated finish',
            'extra_STOP_done_copies_per_role': STOP_EXTRA, 'ordering': 'SHA256 and source-diverse first',
            'validation': 'original source holdout and exact original validation targets unchanged',
            'training_lr': 5e-5, 'epochs': 1, 'initial_adapters': {'korean': 2, 'revision': 1}},
        'new_API_calls': 0, 'diagnostic_examples_used_as_training': False,
        'test_cohort_untouched': True, 'shared_validation_groups': original['shared_validation_groups']}
    for role in ROLES:
        source = original['roles'][role]['train']
        if file_sha(source['path']) != source['sha256']:
            raise ValueError('Frozen original SFT data changed')
        raw = load(source['path']); base = []; count = Counter(); changes = []; exclusions = []
        for index, sample in enumerate(raw):
            assert sample['partition'] == 'train'
            assert sample['source_group'] not in original['shared_validation_groups']
            payload = assert_rendered(sample['messages'][:-1]); target = json.loads(sample['messages'][-1]['content'])
            widened = target
            if target['action'] == 'EDIT':
                widened, audit = widen(sentence_map(payload)[target['sentence']], target)
                count['EDIT_targets'] += 1
                count['EDIT_widened'] += widened != target
                changes.append({'source_id': sample['source_id'], 'sample_index': index,
                                'old_target': target, 'target': widened, **audit})
            try:
                if widened == target:
                    row = deepcopy(sample)
                    row['fix1'] = {'kind': 'base_widened', 'source_index': index}
                else:
                    row = encode(sample, sample['messages'][:-1], widened, tok,
                                 {'kind': 'base_widened', 'source_index': index})
            except OverflowError:
                # Never silently drop a requested base training sample.
                raise ValueError('A widened base example overflowed; requires explicit repair')
            base.append(row)
        spare_done = set()
        for sample in base:
            target = json.loads(sample['messages'][-1]['content'])
            if (target['action'] == 'STOP' and target.get('status') == 'done'
                    and assert_rendered(sample['messages'][:-1])['steps_left'] >= 2):
                spare_done.add(stage_key(sample))
        additions = []
        for kind in ERROR:
            pools = {'EDIT': [], 'STOP': []}
            for sample in base:
                payload = assert_rendered(sample['messages'][:-1]); target = json.loads(sample['messages'][-1]['content'])
                if target['action'] == 'EDIT' and stage_key(sample) not in spare_done:
                    continue
                bad = rejected_action(payload, target, kind)
                if bad is None:
                    continue
                updated = after_rejection(payload, bad, role, kind)
                if updated is None:
                    continue
                pools[target['action']].append(sample)
            chosen = choose_diverse(pools['STOP'], RECOVERY_STOP_PER_TYPE)
            chosen += choose_diverse(pools['EDIT'], RECOVERY_PER_TYPE-len(chosen))
            accepted = []
            for sample in chosen:
                payload = assert_rendered(sample['messages'][:-1]); target = json.loads(sample['messages'][-1]['content'])
                bad = rejected_action(payload, target, kind)
                updated = after_rejection(payload, bad, role, kind)
                try:
                    recovery = encode(sample, render(updated, sample['messages'][0]['content']), target, tok,
                        {'kind': 'recovery_'+kind, 'source_index': sample['fix1']['source_index'],
                         'rejected_action': bad, 'synthetic_notice': True,
                         'target_evidence': 'accepted teacher action' if target['action'] == 'EDIT' else 'selected teacher STOP(done)'})
                except OverflowError:
                    exclusions.append({'kind': kind, 'source_index': sample['fix1']['source_index'], 'reason': 'context'})
                    continue
                accepted.append(recovery)
            additions.extend(accepted)
            count['recovery_'+kind] = len(accepted)
            count['recovery_'+kind+'_STOP'] = sum(json.loads(s['messages'][-1]['content'])['action'] == 'STOP' for s in accepted)
        stop = [s for s in base if (lambda a: a['action'] == 'STOP' and a.get('status') == 'done')(
            json.loads(s['messages'][-1]['content']))]
        for sample in choose_diverse(stop, STOP_EXTRA):
            duplicate = deepcopy(sample); duplicate['fix1']['kind'] = 'STOP_done_extra_copy'
            additions.append(duplicate)
            count['STOP_done_extra_copy'] += 1
        all_rows = base+additions
        count['base_examples'] = len(base); count['added_examples'] = len(additions)
        count['original_STOP_done'] = len(stop)
        count['final_STOP_done'] = sum((lambda a: a['action'] == 'STOP' and a.get('status') == 'done')(
            json.loads(s['messages'][-1]['content'])) for s in all_rows)
        path = ROOT/f'data/{role}.train.jsonl.gz'; path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as out:
            for row in all_rows:
                out.write(json.dumps(row, ensure_ascii=False)+'\n')
        validation = original['roles'][role]['validation']
        if file_sha(validation['path']) != validation['sha256']:
            raise ValueError('Frozen validation changed')
        val_path = ROOT/f'data/{role}.validation.jsonl.gz'
        shutil.copyfile(validation['path'], val_path)
        assert file_sha(val_path) == validation['sha256']
        from .common import write_json
        write_json(ROOT/f'data/{role}.widen_audit.json', changes)
        write_json(ROOT/f'data/{role}.excluded_augmentations.json', exclusions)
        result['roles'][role] = {'counts': dict(count), 'original': ref(Path(source['path'])),
            'train': {**ref(path), 'turns': len(all_rows), 'sources': len({s['source_id'] for s in all_rows}),
                'target_tokens': sum(s['target_tokens'] for s in all_rows), 'lengths': lengths([s['total_tokens'] for s in all_rows])},
            'validation': {**ref(val_path), 'turns': validation['turns']},
            'STOP_done_share_before': len(stop)/len(base), 'STOP_done_share_after': count['final_STOP_done']/len(all_rows),
            'widen_audit': ref(ROOT/f'data/{role}.widen_audit.json'), 'excluded_augmentations': len(exclusions)}
        assert result['roles'][role]['STOP_done_share_after'] > result['roles'][role]['STOP_done_share_before']
    atomic_new(manifest_path, result)
    return result


def verify():
    value = read_json(ROOT/'data/manifest.json')
    refs = [value['source_manifest']]
    for role in ROLES:
        refs += [value['roles'][role][part] for part in ('original', 'train', 'validation', 'widen_audit')]
    for item in refs:
        if file_sha(item['path']) != item['sha256']:
            raise ValueError('Fix1 data changed: '+item['path'])
    return value


if __name__ == '__main__':
    value = build()
    print(json.dumps({r: value['roles'][r]['counts'] for r in ROLES}, ensure_ascii=False))
