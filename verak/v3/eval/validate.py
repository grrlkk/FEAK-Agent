"""Final Phase 6 integrity checks, using saved artifacts only."""
from collections import Counter

from ..agent.runner import system_prompt
from ..common import read_json, write_json, file_sha
from ..phase2 import read_jsonl
from .design import prepare
from .filter import filter_rows


def validate(config, api):
    design, corpus, examples = prepare(config)
    output = config['paths']['phase6_output']
    checks, failures = [], []
    def check(name, ok, detail=None):
        item = {'check': name, 'passed': bool(ok), 'detail': detail}
        checks.append(item)
        if not ok:
            failures.append(name)
    judgments = read_jsonl(output/'filter/judgments.jsonl')
    index = {r['key']: r['judgment'] for r in judgments}
    check('filter_judgment_keys_unique', len(index) == len(judgments))
    original_hashes = read_json(output/'filter/manifest.json')['source_hashes']
    for split in ('agent_train', 'agent_dev'):
        path = config['paths']['metadata']/'corrupt'/(split+'.jsonl')
        check(split+'_original_unchanged', file_sha(path) == original_hashes[split])
        original = read_jsonl(path)
        expected, _ = filter_rows(original, index)
        actual = read_jsonl(config['paths']['active_corrupt']/(split+'.jsonl'))
        check(split+'_filtered_exactly_no_replacement', actual == expected)
    requested = {'two_stage': design['paired_ids'], 'single': design['paired_ids'],
                 'check_once': design['check_ids'], 'real': design['real_ids'],
                 'rewrite_sol': design['paired_ids'], 'rewrite_kanana': design['paired_ids']}
    role_summary = Counter()
    for condition, ids in requested.items():
        rows = [read_json(p) for p in (output/condition/'episodes').glob('*.json')]
        check(condition+'_exact_sample', len(rows) == len(ids) and {r['corpus_episode_id'] for r in rows} == set(ids))
        check(condition+'_complete', all(r['completed'] and not r.get('runtime_error') and not r.get('measurement_error') for r in rows),
              {'completed': sum(r['completed'] for r in rows), 'requested': len(ids)})
        check(condition+'_measurements_present', all('measurement' in r and
              ('quality_measurement' in r if condition == 'real' else r.get('reward') is not None) for r in rows))
        if condition.startswith('rewrite_'):
            check(condition+'_record_blind_alignment', all(r.get('alignment', {}).get('method') == 'corrupted_only_hungarian_lexical' for r in rows))
            continue
        variant = 'check_once' if condition == 'check_once' else 'default'
        for row in rows:
            ok = all(history[0] == {'role': 'system', 'content': system_prompt(role, variant)}
                     for role, history in row['messages_by_role'].items())
            check(condition+':'+row['corpus_episode_id']+':prompts', ok)
            prefixes = {}
            caching_ok = True
            for call in row['calls']:
                role = call['role']
                if role in prefixes and call['messages'][:len(prefixes[role])] != prefixes[role]:
                    caching_ok = False
                prefixes[role] = call['messages']
            check(condition+':'+row['corpus_episode_id']+':append_only_history', caching_ok)
            for role, actions in row['actions_by_role'].items():
                for action in actions:
                    if action['action'] == 'STOP' and action['valid']:
                        role_summary[role+'_stops'] += 1
                        role_summary[role+'_nonempty_summaries'] += bool(action['args'].get('summary', '').strip())
            if condition == 'real':
                check('real:'+row['source_id']+':no_reward', row['reward'] is None)
    real_judgments = [read_json(p) for p in (output/'real_judgments').glob('valid_*.json')]
    check('real_separate_judgments', len(real_judgments) == 30 and {r['source_id'] for r in real_judgments} == set(design['real_ids']))
    local = [read_json(p) for p in (output/'local_link/essays').glob('*.json')]
    check('local_exact_100_sources', len(local) == 100 and {r['source_id'] for r in local} == set(design['local_source_ids']))
    check('local_operators_independent', all(r['record']['before_sha256'] == examples[row['source_id']].essay_hash
          for row in local for r in row['operators'] if r['applied']))
    accounting = api.accounting()
    check('confirmed_cost_within_35', accounting['confirmed_usd'] <= 35, accounting)
    check('no_pending_api_calls', accounting['pending'] == 0)
    check('filter_within_260_calls', accounting['by_stage']['filter']['calls'] <= 260)
    with api.db() as db:
        calls = db.execute("SELECT id,stage,status,created,finished,path FROM calls WHERE status!='blocked_before_send'").fetchall()
    boundaries = [(start, 1) for _, _, _, start, _, _ in calls]+[(finish, -1) for _, _, _, _, finish, _ in calls if finish]
    current = maximum = 0
    for _, delta in sorted(boundaries):
        current += delta
        maximum = max(current, maximum)
    check('at_most_four_API_requests', maximum <= 4, maximum)
    completed_keys = []
    for _, stage, status, _, _, path in calls:
        if status != 'completed':
            continue
        record = read_json(path)
        check('api_model_effort:'+str(record['phase_call']), record['model'] == 'gpt-6.1-sol' and
              record['reasoning_effort'] == ('high' if stage in {'filter', 'content_judge'} else 'low'))
        completed_keys.append(record['fingerprint'])
    check('completed_requests_not_regenerated', len(set(completed_keys)) == len(completed_keys))
    result = {'passed': not failures, 'checks': len(checks), 'failures': failures,
              'stop_summaries': dict(role_summary), 'details': checks, 'maximum_API_concurrency': maximum}
    write_json(output/'validation.json', result)
    if failures:
        raise ValueError('Phase 6 validation failed: '+', '.join(failures[:10]))
    return result
