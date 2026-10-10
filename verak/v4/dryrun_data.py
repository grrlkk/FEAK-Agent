"""Freeze existing eligible B2 demonstrations, then export exact action targets."""
from collections import Counter
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time

from .dryrun_common import (ROOT, B2, ROLES, SEED, REVISION, FROZEN, read_json,
    write_json, file_sha, sha_text, atomic_new, reference, checked, tokenizer, verify_manifest)
from .common import safe_id, collection_lock
from .scale3_content import selection
from . import policy_prompts_v43 as prompts
from .runtime_v43 import export as frozen_export
from .content_env import parse_json
from .teacher_actions_v43 import first_action
from verak.v3.train.sft_data import source_split
from verak.v3.train.formatting import lengths


def choose(candidates, *, count=200, seed=SEED):
    """One already-eligible attempt per source/role; preference never changes the gate."""
    selected = {}
    for role in ROLES:
        by_source = {}
        for candidate in candidates:
            if not candidate['selection']['export_keep'] or not candidate['valid_calls'].get(role, 0):
                continue
            by_source.setdefault(candidate['source_id'], []).append(candidate)
        if len(by_source) < count:
            raise ValueError(f'{role}: only {len(by_source)} eligible sources; require {count}')
        ordered = sorted(by_source, key=lambda source: (sha_text(f'{seed}:{source}'), source))[:count]
        selected[role] = {}
        for source in ordered:
            values = by_source[source]
            best = min(values, key=lambda c: (str(c['attempt']) != str(c['preferred']), c['attempt']))
            selected[role][source] = deepcopy(best)
    return selected


def freeze(*, root=ROOT, source_root=B2, count=200):
    destination = root / 'data/source_manifest.json'
    if destination.exists():
        result = read_json(destination)
        if result['sources_per_role'] != count:
            raise ValueError('Existing dry-run source count is immutable')
        return result
    candidates, errors, inventory = [], [], []
    # Capture the directory listing once. Subsequent collector completions do not
    # enter this dry run, including on export resume.
    judgment_paths = sorted((source_root / 'content/judgments').glob('*.json'))
    for path in judgment_paths:
        judgment = read_json(path)
        inventory.append(reference(path))
        if judgment.get('status') != 'completed':
            continue
        source = judgment['source_id']
        plan_path = source_root / 'content/items' / (safe_id(source) + '.json')
        row_path = source_root / 'essays' / (safe_id(source) + '.json')
        plan, row = read_json(plan_path), read_json(row_path)
        if plan.get('status') != 'completed':
            errors.append({'source_id': source, 'reason': 'judged source lacks completed plan'})
            continue
        for verdict in judgment['attempts']:
            attempt_path = source_root / 'content/attempts' / f'{safe_id(source)}_a{verdict["attempt"]}.json'
            attempt = read_json(attempt_path)
            if attempt['source_id'] != source or attempt['attempt'] != verdict['attempt']:
                raise ValueError('Saved judgment/attempt identity mismatch')
            candidates.append({'source_id': source, 'genre': row['genre'], 'attempt': attempt['attempt'],
                'preferred': judgment['preferred'], 'selection': selection(plan['items'], attempt, verdict),
                'valid_calls': {role: sum(c['role'] == role and c['action']['valid']
                                         for c in attempt['calls']) for role in ROLES},
                'files': {'attempt': reference(attempt_path), 'plan': reference(plan_path),
                          'row': reference(row_path), 'judgment': reference(path)}})
    selected = choose(candidates, count=count)
    mapped = {'global': selected['revision'], 'korean': selected['korean']}
    corpus = {s: {'source_id': s} for values in mapped.values() for s in values}
    split = source_split(mapped, corpus, seed=SEED)
    split['roles']['revision'] = split['roles'].pop('global')
    result = {'schema_version': 1, 'frozen_at': time.time(), 'sources_per_role': count,
        'seed': SEED, 'candidate_frame': candidates, 'judgment_inventory': inventory,
        'errors': errors, 'roles': selected, 'split': split,
        'selection_function_sha256': sha_text(inspect.getsource(selection)),
        'preference': 'saved preferred eligible attempt, otherwise lowest attempt index',
        'source_order': 'SHA256(seed:source_id), then source_id; frozen once',
        'B3_included': False, 'B3_omission': 'GPU-reference corruption selections not yet final',
        'contract': reference(source_root / 'contract.json'),
        'sample': reference(source_root / 'sample.json'), 'paid_calls': 0}
    atomic_new(destination, result)
    return result


def cases_for(source):
    union = {}
    for role in ROLES:
        for selected in source['roles'][role].values():
            key = (selected['source_id'], selected['attempt'])
            union.setdefault(key, {'selected': selected, 'roles': []})['roles'].append(role)
    cases, audit = [], []
    for (source_id, index), item in sorted(union.items()):
        selected = item['selected']; files = selected['files']
        row, plan, attempt, judgment = (checked(files[k]) for k in ('row', 'plan', 'attempt', 'judgment'))
        verdict = next(v for v in judgment['attempts'] if v['attempt'] == index)
        if not selection(plan['items'], attempt, verdict)['export_keep']:
            raise ValueError('A frozen selected attempt is no longer eligible')
        attempt = deepcopy(attempt)
        attempt['calls'] = [c for c in attempt['calls'] if c['role'] in item['roles']]
        for call in attempt['calls']:
            # Even rejected calls retained as context must have the exact system
            # and lossless current-snapshot shape. Only accepted targets export.
            prompts.assert_messages(call['public_messages'], call['role'])
            if not call['action']['valid']:
                continue
            if first_action(call['raw'])[0] != call['canonical_action']:
                raise ValueError('Teacher canonical target/raw mismatch')
            if parse_json(call['canonical_action']) != call['action']['value']:
                raise ValueError('Teacher canonical target/execution mismatch')
            audit.append({'source_id': source_id, 'attempt': index, 'role': call['role'],
                'delegation': call['delegation'], 'turn': call['turn'],
                'public_messages_sha256': sha_text(json.dumps(call['public_messages'], ensure_ascii=False, sort_keys=True)),
                'canonical_action_sha256': sha_text(call['canonical_action']),
                'action': call['action']['action'], 'teacher_trimmed': call['teacher_trim']['trimmed']})
        cases.append({'source_id': source_id, 'genre': row['genre'], 'question': row['question'],
            'original': row['text'], 'items': plan['items'], 'attempt': attempt})
    return cases, audit


def audit_sample(sample, tok):
    messages = sample['messages']
    prompts.assert_messages(messages[:-1], sample['role'])
    prefix = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    whole = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids, labels = sample['input_ids'], sample['labels']
    n = len(prefix)
    if (ids != whole or ids[:n] != prefix or len(ids) != len(labels)
            or any(label != -100 for label in labels[:n]) or labels[n:] != ids[n:]
            or not labels[n:] or n > 7168 or len(ids) > 8192):
        raise ValueError('Exact inference prefix/action-only mask/context audit failed')
    if ids[-1] != tok.eos_token_id or labels[-1] != tok.eos_token_id:
        raise ValueError('End-of-turn token must remain an action loss target')
    parse_json(messages[-1]['content'])
    return {'prefix_tokens': n, 'total_tokens': len(ids), 'target_tokens': len(ids)-n,
            'end_turn_token': ids[-1]}


def export(*, root=ROOT, source_root=B2, count=200):
    with collection_lock(root / 'export.lock'):
        if (root / 'data/manifest.json').exists():
            return verify_manifest(root=root)
        source = freeze(root=root, source_root=source_root, count=count)
        tok = tokenizer()
        cases, calls = cases_for(source)
        base_export = frozen_export(cases, tok, output_root=root / 'data')
        result = {'contract': {'base_revision': REVISION,
            'source_manifest_sha256': file_sha(root / 'data/source_manifest.json'),
            'prompt_manifest_sha256': file_sha(FROZEN), 'prompt_version': prompts.VERSION,
            'tokenizer_sha256': file_sha(Path(tok.name_or_path) / 'tokenizer.json'),
            'chat_template_sha256': sha_text(tok.chat_template), 'context_limit': 8192,
            'generation_reserve': 1024, 'truncation': False, 'packing': False,
            'loss': 'canonical current action plus end-of-turn; all observations/tool outputs/notices masked',
            'raw_teacher_preserved': True, 'invalid_action_targets_included': False,
            'B3_included': False, 'new_API_calls': 0}, 'roles': {}, 'base_export': base_export,
            'split': source['split'], 'action_audit': [], 'training': False}
        lookup = {(c['source_id'], c['attempt'], c['role'], c['delegation'], c['turn']): c for c in calls}
        for role in ROLES:
            rows = [json.loads(line) for line in Path(base_export[role]['path']).read_text().splitlines()]
            if {r['source_id'] for r in rows} != set(source['roles'][role]):
                raise ValueError('Every selected role source must have exported accepted action targets')
            result['roles'][role] = {}
            for part in ('train', 'validation'):
                accepted_sources = set(source['split']['roles'][role][part+'_sources'])
                path = root / 'data' / f'{role}.{part}.jsonl.gz'
                sizes, targets, prefixes, actions = [], [], [], Counter()
                # mtime=0 makes interrupted re-export deterministic, including gzip bytes.
                with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                    for row in rows:
                        if row['source_id'] not in accepted_sources:
                            continue
                        stats = audit_sample(row, tok)
                        key = (row['source_id'], row['attempt'], role, row['delegation'], row['turn'])
                        original = lookup[key]
                        if sha_text(row['messages'][-1]['content']) != original['canonical_action_sha256']:
                            raise ValueError('Exported action changed from frozen executed teacher target')
                        row.update(partition=part, **stats)
                        stream.write((json.dumps(row, ensure_ascii=False)+'\n').encode('utf-8'))
                        sizes.append(stats['total_tokens']);targets.append(stats['target_tokens'])
                        prefixes.append(stats['prefix_tokens']);actions[original['action']]+=1
                        result['action_audit'].append({**original, 'partition': part, **stats})
                result['roles'][role][part] = {'path': str(path), 'sha256': file_sha(path),
                    'sources': len(accepted_sources), 'trajectories': len(accepted_sources), 'turns': len(sizes),
                    'tokens': sum(sizes), 'target_tokens': sum(targets), 'lengths': lengths(sizes),
                    'target_lengths': lengths(targets), 'prefix_lengths': lengths(prefixes), 'actions': dict(actions)}
        if len(result['action_audit']) != len(calls):
            raise ValueError('Action target loss/export audit did not cover every accepted call')
        atomic_new(root / 'data/manifest.json', result)
        return result
