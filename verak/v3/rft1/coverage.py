"""Read-only best-of-four coverage from a frozen list of saved rollout files."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..common import file_sha, read_json, write_json
from ..phase2 import read_jsonl


def recovery(row, record, *, combined=False):
    rewards = row.get('reward') or {}
    if combined:
        reward, field = rewards.get('combined'), 'recovery'
    else:
        role = 'global' if record['level'] == 'GLOBAL' else 'korean'
        reward = rewards.get(role)
        if reward is None and role == 'global':
            reward = row.get('global_only_reward')
        field = 'main'
    return next((r[field] for r in (reward or {}).get('per_record', [])
                 if r['record_id'] == record['record_id']), None)


def coverage(corpus, rows):
    groups = defaultdict(dict)
    for row in rows:
        eid, sample = row['corpus_episode_id'], row['rft1']['sample']
        if eid not in corpus or sample not in (1, 2, 3, 4) or sample in groups[eid]:
            raise ValueError('Unknown essay, invalid sample, or duplicate saved slot')
        groups[eid][sample] = row
    complete = {eid: samples for eid, samples in groups.items() if len(samples) == 4}
    operators = {}
    for op in sorted({r['op'] for e in corpus.values() for r in e['records']}):
        own = {eid: [r for r in corpus[eid]['records'] if r['op'] == op]
               for eid in complete if any(r['op'] == op for r in corpus[eid]['records'])}
        result = {'essays_with_four_saved_samples': len(own), 'records': sum(map(len, own.values()))}
        for combined in (False, True):
            success, possible, record_success, unknown_records = 0, 0, 0, 0
            details = []
            for eid, records in own.items():
                scores = [[recovery(complete[eid][sample], r, combined=combined) for r in records]
                          for sample in (1, 2, 3, 4)]
                full = [i + 1 for i, values in enumerate(scores)
                        if all(v is not None and v >= 1 - 1e-12 for v in values)]
                unresolved = any(all(v is None or v >= 1 - 1e-12 for v in values)
                                 and any(v is None for v in values) for values in scores)
                success += bool(full)
                possible += bool(full) or unresolved
                for column in zip(*scores):
                    recovered = any(v is not None and v >= 1 - 1e-12 for v in column)
                    record_success += recovered
                    unknown_records += not recovered and any(v is None for v in column)
                details.append({'episode_id': eid, 'record_ids': [r['record_id'] for r in records],
                                'scores_by_sample': scores, 'successful_samples': full,
                                'unresolved': not bool(full) and unresolved})
            result['combined_full_record' if combined else 'role_main'] = {
                'successful_essays': success, 'share': success / len(own) if own else None,
                'upper_share_if_unknown_success': possible / len(own) if own else None,
                'unresolved_essays': possible - success,
                'records_recovered_in_any_sample': record_success,
                'record_share': record_success / result['records'] if result['records'] else None,
                'unresolved_records': unknown_records, 'details': details}
        operators[op] = result
    return {'saved_samples': len(rows), 'essays_with_any_saved_sample': len(groups),
            'essays_with_four_saved_samples': len(complete),
            'partial_essays_excluded': {eid: sorted(samples) for eid, samples in groups.items() if len(samples) < 4},
            'levels_four_samples': dict(Counter(corpus[eid]['level'] for eid in complete)),
            'sample_runtime_errors': dict(Counter((r.get('runtime_error') or {}).get('type')
                                                   for r in rows if r.get('runtime_error'))),
            'definition': 'At least one of the four saved samples fully recovers every record of that operator in the essay; no STOP or R selection filter.',
            'main_definition': 'GLOBAL stage-1 main; KOREAN final main. Combined full-record recovery additionally requires any coupled repairs.',
            'unknown_policy': 'Unobserved rewards are unknown; denominator retains saved error attempts and bounds are reported.',
            'operators': operators}


def snapshot(root, corpus_path, destination):
    root, corpus_path, destination = map(Path, (root, corpus_path, destination))
    files = sorted(root.glob('rollouts/sample_*/episodes/*.json'))
    corpus = {r['episode_id']: r for r in read_jsonl(corpus_path)}
    rows, provenance = [], []
    for path in files:
        row = read_json(path)
        rows.append(row)
        provenance.append({'path': str(path), 'sha256': file_sha(path),
                           'episode_id': row['corpus_episode_id'], 'sample': row['rft1']['sample']})
    result = coverage(corpus, rows)
    result.update(at=datetime.now(timezone.utc).isoformat(), corpus_sha256=file_sha(corpus_path),
                  files=provenance, new_calls=0, gpu_used=False,
                  caveat='Early stratified rollout prefix, not the completed 1,430-essay result or held-out evaluation.')
    if destination.exists():
        raise FileExistsError('Keep prior snapshot immutable: ' + str(destination))
    write_json(destination, result)
    return result
