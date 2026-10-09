"""Frozen source sampling with source-level exclusion of both SFT v1 cohorts."""
from collections import Counter
import json
from pathlib import Path
import random

from ..common import file_sha, read_json, sha_text, write_json
from ..corrupt.document import source_document
from ..corrupt.sources import select_sources
from .config import PHASE, OPERATORS, require_v2
from .operators import fusion_candidates, label_targets


class SourceTokensOnly:
    def seed(self, source):
        pass


def excluded_sources(config):
    path = config['paths']['repo'] / 'verak/v3/outputs/phase7_sft/evaluation_design.json'
    design = read_json(path)
    with (config['paths']['active_corrupt'] / 'agent_dev.jsonl').open() as stream:
        dev = {r['episode_id']: r for r in map(json.loads, stream)}
    excluded = {}
    for episode_id in design['contract']['dev_ids']:
        row = dev[episode_id]
        excluded[row['source_id']] = {'source_hash': row['source_hash'], 'cohort': 'sft_v1_dev'}
    for episode_id in design['contract']['real_ids']:
        row = read_json(design['reuse'][episode_id]['path'])
        text = ''.join(gap + ''.join(u['leading'] + u['text'] for u in p['units'])
                       for gap, p in zip(row['initial_layout']['gaps'], row['initial_layout']['paragraphs'])) + row['initial_layout']['tail']
        excluded[row['source_id']] = {'source_hash': sha_text(text), 'cohort': 'sft_v1_real'}
    if len(design['contract']['dev_ids']) != 100 or len(design['contract']['real_ids']) != 30:
        raise ValueError('The frozen SFT v1 cohort changed')
    return excluded, {'path': str(path), 'sha256': file_sha(path)}


def source_pools(config):
    require_v2(config)
    excluded, evidence = excluded_sources(config)
    excluded_hashes = {v['source_hash'] for v in excluded.values()}
    pools, metadata = {}, {}
    for split in ('agent_train', 'agent_dev'):
        examples, _ = select_sources(config, split)
        scores_path = config['paths']['phase3_output'] / f'sources_{split}.json'
        scores = read_json(scores_path)['rows']
        pool = []
        for example in examples:
            if example.id in excluded or example.essay_hash in excluded_hashes:
                continue
            saved = scores[example.id]
            if not saved['eligible']:
                continue
            if saved['essay_hash'] != example.essay_hash:
                raise ValueError('Frozen source score provenance changed')
            source = source_document(config, example, SourceTokensOnly())
            pool.append((example, source, saved['score']))
        pools[split] = pool
        metadata[split] = {'eligible_source_essays': len(pool), 'source_score_path': str(scores_path),
                           'source_score_sha256': file_sha(scores_path)}
    return pools, {'excluded_sources': excluded, 'v1_design': evidence, 'pools': metadata}


def prepare(config):
    require_v2(config)
    root = config['paths'][PHASE + '_output']
    pools, audit = source_pools(config)
    plans, stats = {}, {}
    for op_index, op in enumerate(OPERATORS):
        plans[op], stats[op] = {}, {}
        for split_index, split in enumerate(('agent_train', 'agent_dev')):
            requested = config[PHASE]['source_counts'][split]
            pool = list(pools[split])
            seed = config[PHASE]['seed'] + 1000 * op_index + 100 * split_index
            random.Random(seed).shuffle(pool)
            selected, counts = [], Counter()
            for example, source, score in pool:
                targets, candidates = label_targets(source), []
                if op == 'G_DEL_LINK':
                    if len(source.units) <= 1 or not targets:
                        counts['no_deletion_site'] += 1
                        continue
                else:
                    candidates, excluded = fusion_candidates(source)
                    counts.update(excluded)
                    if not candidates:
                        counts['no_fusion_window'] += 1
                        continue
                selected.append({'source_id': example.id, 'source_hash': example.essay_hash,
                    'question_hash': example.question_hash, 'genre': example.genre,
                    'label_targets': targets if op == 'G_DEL_LINK' else None,
                    'fusion_windows': [dict(paragraph=c.paragraph, position=c.position,
                        sids=list(c.sids), source_classes=list(c.source_classes)) for c in candidates],
                    'source_score': score['mean']})
                if len(selected) == requested:
                    break
            if len(selected) != requested:
                raise ValueError(f'Only {len(selected)}/{requested} feasible {split} sources for {op}')
            plans[op][split] = selected
            stats[op][split] = {'selected': len(selected), 'seed': seed, 'screening': dict(counts),
                'genres': dict(Counter(r['genre'] for r in selected))}
    result = {'version': 'v2', 'source_policy': 'same frozen valid-only, genre q75, 500–2500-character source pool as v1 corruption',
        'operators': list(OPERATORS), 'records_per_essay': 1, 'plans': plans, 'sampling': stats,
        'exclusion_audit': audit, 'budget_usd': config[PHASE]['max_cost_usd'],
        'paid_calls': 0, 'new_scorer_calls': 0, 'gpu_used': False,
        'operator_source_overlap_permitted': True, 'train_dev_sources_disjoint': True}
    train_ids = {r['source_id'] for op in OPERATORS for r in plans[op]['agent_train']}
    dev_ids = {r['source_id'] for op in OPERATORS for r in plans[op]['agent_dev']}
    if train_ids & dev_ids or (train_ids | dev_ids) & audit['excluded_sources'].keys():
        raise ValueError('Source-level split or v1 evaluation leakage')
    path = root / 'source_plan.json'
    if path.exists() and read_json(path) != result:
        raise ValueError('Frozen v2 source sampling changed; inspect before creating paid data')
    write_json(path, result)
    return result


if __name__ == '__main__':
    from .config import config_for
    result = prepare(config_for(version='v2'))
    print(json.dumps({'sampling': result['sampling'], 'pools': result['exclusion_audit']['pools'],
                     'paid_calls': 0, 'gpu_used': False}, ensure_ascii=False, indent=2))
