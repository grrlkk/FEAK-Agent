"""Offline, annotated L_CONN cleanup; never mutate a historical corpus or target."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

from .common import REPO, ref, read_json, sha_text, file_sha
from .check_judge import freeze
from verak.v3.ko.coarse import CLOSED_CONJUNCTIONS

ROOT = REPO / 'verak/v5/outputs/cooperate1/lconn_cleanup'
SOURCE = REPO / 'verak/v3/data/corrupt_scope_v3'
TRAIN = REPO / 'verak/v4/outputs/train44/data'


def render(layout):
    return ''.join(gap + ''.join(u['leading'] + u['text'] for u in p['units'])
                   for gap, p in zip(layout['gaps'], layout['paragraphs'])) + layout['tail']


def clean_task(task, labels):
    """Restore just the malformed units. Other record damage and IDs are preserved.

    This is a preparation input, NOT a reward-ready v1 corpus: old scores/views and
    reverse-application snapshots become stale after removing a corruption.
    """
    result = deepcopy(task)
    dropped, relabeled, uncertain = [], [], []
    units = {u['sid']: u for p in result['corrupted_layout']['paragraphs'] for u in p['units']}
    records = []
    for record in task['records']:
        if record['op'] != 'L_CONN':
            records.append(deepcopy(record)); continue
        annotation = labels[record['record_id']]
        label = annotation['label']
        if label.startswith('ill_formed_'):
            for sid in record['sids']:
                if any(r['record_id'] != record['record_id'] and sid in r['sids'] for r in task['records']):
                    raise ValueError('Overlapping record needs explicit replay: ' + record['record_id'])
                if units[sid]['text'] != record['corrupted_text'][sid]:
                    raise ValueError('Current sentence differs from audited corruption')
                units[sid]['text'] = record['original_text'][sid]
            dropped.append(record['record_id']); continue
        if label.startswith('uncertain'):
            uncertain.append(record['record_id'])
        rec = deepcopy(record)
        if annotation.get('conjunction_misclassified_as_EC'):
            sid = record['sids'][0]
            form = annotation.get('conjunction_form', '하지만')
            start = len(record['original_text'][sid]) - len(record['original_text'][sid].lstrip())
            if not record['original_text'][sid][start:].startswith(form):
                raise ValueError('Relabeled conjunction not sentence initial')
            coarse = CLOSED_CONJUNCTIONS.get(form, annotation.get('conjunction_class'))
            if coarse is None:
                raise ValueError('An out-of-set conjunction needs an explicit audit class')
            rec.update(op='L_CONJ', level='SENTENCE', cleanup_previous_operator='L_CONN')
            rec['recovery_target'].update(channel='initial_conjunction', coarse_class=coarse,
                                          site=[start, start+len(form)])
            previous = deepcopy(rec['params'])
            end = start+len(form)+len(previous['new'])-len(previous['old'])
            rec['params'].update(coarse_before=coarse, span=[start, start+len(form)],
                                 original_conjunction=form, old=form,
                                 new=record['corrupted_text'][sid][start:end],
                                 previous_EC_params=previous)
            rec['params'].pop('token_id', None)
            # Do not pretend a malformed replacement belongs to a conjunction class.
            rec['params']['coarse_after'] = None
            rec['cleanup'] = {'original_record_preserved_in': record['record_id'],
                              'corrupted_surface_unchanged': True,
                              'closed_set_original': form in CLOSED_CONJUNCTIONS}
            relabeled.append(record['record_id'])
        records.append(rec)
    result['records'] = records
    if dropped or relabeled:
        result['corrupted_text'] = render(result['corrupted_layout'])
        result['corrupted_hash'] = sha_text(result['corrupted_text'])
        invalidated = []
        for key in ('compact_view', 'compact_tokens', 'analysis', 'q_corrupted', 'corrupted_score',
                    'scorer_fingerprint', 'scorer_average_k', 'llm_judgment', 'record_kept',
                    'essay_kept', 'balance_selected', 'human_ok', 'judgment_origin'):
            if key in result:
                invalidated.append(key); result.pop(key)
        for record in result['records']:
            # Reverse snapshots contain the dropped corruption; archive is authoritative.
            for key in ('inverse', 'before_sha256', 'after_sha256'):
                record.pop(key, None)
        result['schema_version'] = 'v5_lconn_cleanup_preparation_input_1'
        result['cleanup'] = {'dropped_record_ids': dropped, 'relabeled_record_ids': relabeled,
            'uncertain_retained_ids': uncertain, 'invalidated_fields': invalidated,
            'reward_ready': False, 'training_ready': False,
            'required_before_new_rollouts': ['regenerate profile/view and record snapshots',
                                            'GPU reference score of cleaned input'],
            'old_demonstrations_compatible': not dropped and not relabeled}
    return result, {'dropped': dropped, 'relabeled': relabeled, 'uncertain': uncertain}


def episode_id(sample):
    return sample.get('corpus_episode_id') or sample.get('provenance', {}).get('episode_id')


def excluded(sample, dropped_episodes, relabeled_episodes):
    eid = episode_id(sample)
    if eid in dropped_episodes:
        return 'input_contains_dropped_ill_formed_record'
    if eid in relabeled_episodes:
        return 'obsolete_L_CONN_task_or_reward_after_reclassification'
    return None


def write_jsonl_new(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Refusing to overwrite cleanup artifact: '+str(path))
    else:
        with path.open('x') as stream:
            stream.write(text)


def run():
    if (ROOT/'complete.json').exists():
        return read_json(ROOT/'complete.json')
    started = time.monotonic(); annotations = read_json(ROOT/'annotations.json')
    by_split, dropped_episodes, relabeled_episodes, changes = {}, set(), set(), []
    for split in ('agent_train', 'agent_dev'):
        path = SOURCE/(split+'.jsonl')
        if file_sha(path) != annotations['inputs'][split]['sha256']:
            raise ValueError('Audit source changed')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        output = []; counts = Counter(); before_ops = Counter(); after_ops = Counter()
        for task in rows:
            result, delta = clean_task(task, annotations['records'])
            output.append(result)
            before_ops.update(r['op'] for r in task['records']); after_ops.update(r['op'] for r in result['records'])
            counts.update({'tasks': 1, 'records_before': len(task['records']), 'records_after': len(result['records']),
                'dropped_records': len(delta['dropped']), 'relabeled_records': len(delta['relabeled']),
                'uncertain_retained': len(delta['uncertain']), 'zero_record_tasks': not result['records'],
                'tasks_restored': bool(delta['dropped']), 'tasks_relabeled': bool(delta['relabeled'])})
            if delta['dropped']: dropped_episodes.add(task['episode_id'])
            if delta['relabeled']: relabeled_episodes.add(task['episode_id'])
            if any(delta.values()):
                changes.append({'split': split, 'episode_id': task['episode_id'], 'source_id': task['source_id'],
                    **delta, 'before_sha256': task['corrupted_hash'], 'after_sha256': result['corrupted_hash'],
                    'remaining_operators': [r['op'] for r in result['records']]})
        out = ROOT/'corpus'/(split+'.jsonl'); write_jsonl_new(out, output)
        by_split[split] = {**counts, 'operators_before': dict(before_ops), 'operators_after': dict(after_ops),
                           'input': ref(path), 'output': ref(out)}
    freeze(ROOT/'changes.json', changes)
    freeze(ROOT/'exclusions.json', {'dropped_episodes': sorted(dropped_episodes),
        'relabeled_episodes': sorted(relabeled_episodes),
        'scope': 'All roles/targets on affected corruption episodes, not other practice episodes of the same source.',
        'reason': 'Saved observations contain obsolete damage or task labels; filtering only the action would leak it.'})
    training = {}; exclusions = []
    manifest = read_json(TRAIN/'manifest.json')
    for role, partitions in manifest['roles'].items():
        for partition in ('train', 'validation'):
            source = Path(partitions[partition]['path'])
            if file_sha(source) != partitions[partition]['sha256']:
                raise ValueError('Frozen SFT export changed')
            dest = ROOT/'sft_without_bad_conn'/source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            totals = Counter(); seen = set(); retained = set(); reasons = Counter()
            # Input lines copied byte-for-byte; loss masks/tokens/prompts are unchanged.
            with gzip.open(source, 'rt') as inp, gzip.open(dest, 'wt') as out:
                for index, line in enumerate(inp, 1):
                    sample = json.loads(line); eid = episode_id(sample)
                    identity = (sample['source_id'], eid, sample.get('attempt'), sample.get('delegation'))
                    seen.add(identity); totals['before_turns'] += 1
                    reason = excluded(sample, dropped_episodes, relabeled_episodes)
                    if reason:
                        reasons[reason] += 1
                        exclusions.append({'role': role, 'partition': partition, 'input_line': index,
                            'episode_id': eid, 'reason': reason})
                    else:
                        out.write(line); retained.add(identity); totals['kept_turns'] += 1
            training[role+'/'+partition] = {**totals, 'excluded_turns': sum(reasons.values()),
                'before_trajectory_keys': len(seen), 'kept_trajectory_keys': len(retained),
                'exclusions_by_reason': dict(reasons), 'input': ref(source), 'output': ref(dest)}
    freeze(ROOT/'excluded_training_turns.json', exclusions)
    result = {'corpora': by_split, 'SFT_exports': training, 'wall_seconds': time.monotonic()-started,
        'annotation': ref(ROOT/'annotations.json'), 'new_API_calls': 0, 'GPU_calls': 0,
        'training_started': False, 'historical_artifacts_modified': False,
        'diagnostic_comparison_inputs_unchanged': True,
        'uncertainty': 'Codex saved-text linguistic audit, not independent human gold; uncertain forms retained.',
        'reuse_requirement': 'Any later stage-3 data derivation must start from these filtered exports or enforce exclusions.json.',
        'corpus_readiness': 'Changed corpus inputs need new preparation and GPU scores; stale fields removed, no v1 consumer changed.'}
    freeze(ROOT/'complete.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
