"""Frozen, source-grouped action-only warm-start exports."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path

from ..common import file_sha, load_config, read_json, sha_text, write_json
from ..phase2 import read_jsonl
from .formatting import formatted_turns, lengths
from .teacher_bulk import atomic_new

REVISION = 'c963a5f4f6496c749f94064a20b33028b0db9f19'
PHASE = 'phase7_sft'
ROLES = ('global', 'korean')


def config_for():
    config = load_config()
    config['paths'][PHASE + '_output'] = config['paths']['repo'] / 'verak/v3/outputs' / PHASE
    if config['policy']['base_revision'] != REVISION:
        raise ValueError('SFT requires the accepted pinned policy revision')
    if (config['policy']['context_limit'], config['policy']['generation_reserve']) != (8192, 1024):
        raise ValueError('Inference and training must use 8192/1024')
    if config['env']['mode'] != 'two_stage' or config['env']['enable_check']:
        raise ValueError('Keep two_stage with no CHECK')
    return config


def source_split(selections, corpus, *, seed=71, fraction=.05):
    """Exact rounded-up 5% source groups per role, with no cross-role leakage."""
    sources = {r: {corpus[i]['source_id'] for i in selections[r]} for r in ROLES}
    target = {r: math.ceil(fraction * len(sources[r])) for r in ROLES}
    common = sources['global'] & sources['korean']
    only = {r: sources[r] - common for r in ROLES}
    low = max(0, *(target[r] - len(only[r]) for r in ROLES))
    high = min(len(common), *target.values())
    if low > high:
        raise ValueError('Cannot form exact shared source-group holdout')
    common_n = max(low, min(high, math.ceil(fraction * len(common))))
    def ordered(values):
        return sorted(values, key=lambda v: (sha_text(f'{seed}:{v}'), v))
    held = set(ordered(common)[:common_n])
    for role in ROLES:
        held.update(ordered(only[role])[:target[role] - common_n])
    result = {'seed': seed, 'fraction': fraction, 'group_key': 'source_id',
              'rounding': 'ceil(0.05 * unique source essays per role)', 'roles': {}}
    for role in ROLES:
        validation = sources[role] & held
        assert len(validation) == target[role]
        result['roles'][role] = {
            'train_sources': sorted(sources[role] - held), 'validation_sources': sorted(validation),
            'train_ids': sorted(i for i in selections[role] if corpus[i]['source_id'] not in held),
            'validation_ids': sorted(i for i in selections[role] if corpus[i]['source_id'] in held)}
    train_union = set().union(*(set(result['roles'][r]['train_sources']) for r in ROLES))
    assert not train_union & held
    return result


def export(config):
    from transformers import AutoTokenizer
    root = config['paths'][PHASE + '_output'] / 'data'
    root.mkdir(parents=True, exist_ok=True)
    bulk = config['paths']['repo'] / 'verak/v3/outputs/teacher_bulk_two_stage'
    selection_path = bulk / 'best_role_trajectories.json'
    selected = read_json(selection_path)
    if {r: len(selected[r]) for r in ROLES} != {'global': 810, 'korean': 903}:
        raise ValueError('Use the accepted 810/903 best-attempt inventory exactly')
    corpus_path = config['paths']['active_corrupt'] / 'agent_train.jsonl'
    corpus = {r['episode_id']: r for r in read_jsonl(corpus_path)}
    dev = read_jsonl(config['paths']['active_corrupt'] / 'agent_dev.jsonl')
    if {r['source_id'] for r in corpus.values()} & {r['source_id'] for r in dev}:
        raise ValueError('Active train/dev source leakage')
    split = source_split(selected, corpus)
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    contract = {'selection_sha256': file_sha(selection_path), 'corpus_sha256': file_sha(corpus_path),
        'base_revision': REVISION, 'tokenizer_sha256': file_sha(config['paths']['policy_base'] / 'tokenizer.json'),
        'chat_template_sha256': sha_text(tokenizer.chat_template), 'context_limit': 8192,
        'generation_reserve': 1024, 'split': split,
        'loss': 'current assistant action content and end-of-turn only; all context masked',
        'teacher_handoffs': True, 'packing': False, 'truncation': False}
    if (root / 'manifest.json').exists():
        result = read_json(root / 'manifest.json')
        if result['contract'] != contract:
            raise ValueError('Existing SFT export contract changed')
        for role in ROLES:
            for part in ('train', 'validation'):
                meta = result['roles'][role][part]
                if file_sha(Path(meta['path'])) != meta['sha256']:
                    raise ValueError('Saved SFT data hash changed')
        return result
    result = {'contract': contract, 'roles': {}, 'source_trajectories': {}, 'checks': 0}
    for role in ROLES:
        result['roles'][role] = {}
        for part in ('train', 'validation'):
            ids = split['roles'][role][part + '_ids']
            path = root / f'{role}.{part}.jsonl.gz'
            sizes, target_sizes, counters = [], [], Counter()
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                for index, episode_id in enumerate(ids):
                    entry = selected[role][episode_id]
                    original = Path(entry['path'])
                    if file_sha(original) != entry['sha256']:
                        raise ValueError('Accepted trajectory hash changed')
                    row = read_json(original)
                    if row['corpus_episode_id'] != episode_id or row['source_id'] != corpus[episode_id]['source_id']:
                        raise ValueError('Trajectory identity mismatch')
                    result['source_trajectories'][role + ':' + episode_id] = entry
                    calls = [c for c in row['calls'] if c['role'] == role]
                    samples = list(formatted_turns(row, role, tokenizer, 8192))
                    if len(samples) != len(calls) or not samples:
                        raise ValueError('Every saved model turn must have one training target')
                    for sample, call in zip(samples, calls):
                        if sample['messages'][:-1] != call['messages']:
                            raise ValueError('SFT prefix differs from actual teacher inference context')
                        if sample['messages'][-1]['content'] != call['raw']:
                            raise ValueError('Action target changed')
                        n = sample['prompt_tokens']
                        if (any(x != -100 for x in sample['labels'][:n]) or
                                sample['labels'][n:] != sample['input_ids'][n:] or not sample['loss_tokens']):
                            raise ValueError('Incorrect action-only loss mask')
                        sample.update(source_id=row['source_id'], attempt=entry['attempt'],
                            source_path=str(original), source_sha256=entry['sha256'], partition=part)
                        stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        sizes.append(sample['total_tokens'])
                        target_sizes.append(sample['loss_tokens'])
                        counters['compacted_turns'] += bool(sample['history_compacted'])
                        counters['json_invalid_targets'] += not call['json_valid']
                        counters['protocol_invalid_targets'] += not call['valid_json_action']
                        result['checks'] += 5
                    if (index + 1) % 100 == 0:
                        print(json.dumps({'export': role, 'partition': part, 'trajectories': index + 1,
                                          'turns': len(sizes)}), flush=True)
            result['roles'][role][part] = {'path': str(path), 'sha256': file_sha(path),
                'trajectories': len(ids), 'sources': len(split['roles'][role][part + '_sources']),
                'turns': len(sizes), 'tokens': sum(sizes), 'target_tokens': sum(target_sizes),
                'lengths': lengths(sizes), 'target_lengths': lengths(target_sizes), **dict(counters)}
    atomic_new(root / 'manifest.json', result)
    return result


def load_samples(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [{k: value[k] for k in ('input_ids', 'labels')} for value in map(json.loads, stream)]
