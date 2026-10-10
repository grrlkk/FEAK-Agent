"""Materialize only the explicitly approved frozen 20+10 diagnostic cohorts."""
from pathlib import Path
import json

from .dryrun_common import (ROOT, read_json, atomic_new, file_sha, sha_text,
                           reference, checked, write_json)
from .common import load_config, safe_id
from .policy_prompts_v43 import verify_frozen, FROZEN


def prepare(*, root=ROOT):
    path = root / 'evaluation/manifest.json'
    if path.exists():
        return read_json(path)
    index_path = root / 'audit/evaluation_cohorts.json'
    index = read_json(index_path)
    if index['counts'] != {'smoke20': 20, 'agent_dev10': 10} or not index['test100_untouched']:
        raise ValueError('Only the approved fixed 20+10 cohorts may materialize')
    smoke = checked(index['cohorts']['smoke20'])
    dev = checked(index['cohorts']['agent_dev10'])
    result = {'schema_version': 1, 'user_approved_cohorts': ['smoke20', 'agent_dev10'],
        'cohort_index': reference(index_path), 'prompt_manifest_sha256': file_sha(FROZEN),
        'quality_results': False, 'test100_untouched': True, 'new_API_calls': 0, 'cohorts': {}}
    # The peer freezes paths and hashes; never sample a replacement here.
    smoke_rows = smoke.get('source_files', smoke.get('rows'))
    if smoke_rows is None or len(smoke_rows) != 20:
        raise ValueError('Exactly twenty approved saved Sol-plan entries are required')
    cases = []
    for item in smoke_rows:
        row_ref = {'path': item['essay_path'], 'sha256': item['essay_sha256']}
        plan_ref = {'path': item['plan_path'], 'sha256': item['plan_sha256']}
        row, plan = checked(row_ref), checked(plan_ref)
        if row['source_id'] != item['source_id'] or plan['source_id'] != item['source_id']:
            raise ValueError('Approved B1 row/plan identity mismatch')
        cases.append({'case_id': row['source_id'], 'source_id': row['source_id'],
                      'row': row_ref, 'plan': plan_ref})
    result['cohorts']['smoke20'] = {'cases': cases, 'original_split': 'train',
        'diagnostic_only': True, 'not_agent_dev_or_test': True,
        'question_overlap_B2_essays': 17, 'question_overlap_B2_unique': '15/17',
        'source_text_overlap_B2': 0, 'plan_provenance': 'saved Sol B1 plans, unchanged'}
    from verak.v3.view_data import load_episode_examples
    from .scale3_corruption import prepare_corruption
    from .dryrun_common import local_analysis_environment
    local_analysis_environment()
    config = load_config()
    examples = {e.id: e for e in load_episode_examples(config, 'agent_dev')}
    if not examples:
        raise ValueError('agent_dev source inventory unavailable; no train fallback')
    corpus = Path(dev['corpus']['path'])
    if file_sha(corpus) != dev['corpus']['sha256']:
        raise ValueError('Frozen active dev corpus changed')
    lines = corpus.read_text().splitlines()
    cases = []
    for item in dev['rows']:
        text = lines[item['corpus_line_1based']-1]
        if sha_text(text) != item['corpus_line_sha256']:
            raise ValueError('Frozen dev corpus line changed')
        row = json.loads(text)
        if (row['source_id'] != item['source_id'] or row['episode_id'] != item['episode_id']
                or sha_text(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
                != item['corpus_row_canonical_sha256']):
            raise ValueError('Frozen dev row identity changed')
        prepared, prepared_path = prepare_corruption(row,
            root=root/'evaluation/dev10_prepared', config=config, examples=examples)
        policy_path = root/'evaluation/dev10_rows'/f'{safe_id(row["episode_id"])}.json'
        plan_path = root/'evaluation/dev10_plans'/f'{safe_id(row["episode_id"])}.json'
        for destination, value in ((policy_path, prepared['policy_row']), (plan_path, prepared['plan'])):
            if destination.exists():
                if read_json(destination) != value:
                    raise ValueError('Frozen prepared dev row/plan changed')
            else:
                atomic_new(destination, value)
        cases.append({'case_id': row['episode_id'], 'source_id': row['source_id'],
            'row': reference(policy_path), 'plan': reference(plan_path), 'prepared': reference(prepared_path),
            'level': item['level'], 'operators': item['operators']})
        write_json(root/'evaluation/preparation_status.json', {'agent_dev_saved': len(cases), 'planned': 10})
    result['cohorts']['agent_dev10'] = {'cases': cases, 'active_split': 'agent_dev',
        'diagnostic_only': True, 'level_counts': dev['level_counts'],
        'record_counts_by_operator': dev['record_counts_by_operator'],
        'B2_overlap': dev['B2_overlap'], 'plan_provenance': 'unchanged exact B3 record-derived delegates'}
    verify_frozen()
    atomic_new(path, result)
    return result
