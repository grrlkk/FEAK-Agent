"""Fixed paired teacher comparison; no bulk or training entry point."""
from collections import Counter
import json

from ..common import read_json, write_json, file_sha, sha_text
from ..phase2 import read_jsonl, write_jsonl
from ..view_data import load_episode_examples
from ..agent.runner import system_prompt, run_episode
from ..env import RevisionEnv
from ..eval.api import Phase6Teacher
from ..eval.resources import Resources
from .pilot import safe_id
from .pilot2_data import inventory, bounded_map
from .formatting import structural_action

PHASE = 'phase7_teacher'


def without_drops(rows):
    return [r for r in rows if not any(x['op'] == 'L_CONJ_DROP' for x in r['records'])]


def prepare(config):
    root = config['paths'][PHASE+'_output']
    root.mkdir(parents=True, exist_ok=True)
    old_root = config['paths']['phase7_pilot2_output']
    previous = read_json(old_root/'corpus_manifest.json')
    manifest = {}
    for split in ('agent_train', 'agent_dev'):
        old_path = config['paths']['metadata']/'corrupt_scope_v2'/f'{split}.jsonl'
        if file_sha(old_path) != previous[split]['final_sha256']:
            raise ValueError('Archived Pilot-2 corpus changed')
        rows = read_jsonl(old_path)
        retained = without_drops(rows)
        removed = [r['episode_id'] for r in rows if r not in retained]
        if len(removed) != 1 or any(x['op'] in config['corruption']['disabled_operators']
                                  for r in retained for x in r['records']):
            raise ValueError('Unexpected scope removal or disabled active operator')
        target = config['paths']['active_corrupt']/f'{split}.jsonl'
        if target.exists() and read_jsonl(target) != retained:
            raise ValueError('Refuse to overwrite a different active corpus')
        if not target.exists():
            write_jsonl(target, retained)
        manifest[split] = {'before': inventory(rows), 'after': inventory(retained),
            'removed_ids': removed, 'previous_path': str(old_path), 'previous_sha256': file_sha(old_path),
            'path': str(target), 'sha256': file_sha(target), 'retained_rows_unchanged': True}
    write_json(root/'corpus_manifest.json', manifest)
    old = read_json(old_root/'design.json')
    corpus = {r['episode_id']: r for r in read_jsonl(config['paths']['active_corrupt']/'agent_train.jsonl')}
    ids = old['retained_pilot_ids']
    if len(ids) != 92 or not set(ids) <= corpus.keys():
        raise ValueError('Exactly the same 92 paired Pilot-2 essays are required')
    pinned = read_json(config['paths']['phase4_output']/'models.json')['luna_model']
    if config[PHASE]['model'] != pinned or config['cheap_model']['model'] != pinned:
        raise ValueError('Teacher comparison must use the Phase-4 pinned Luna ID')
    hashes = {p: file_sha(config['paths']['repo']/'verak/v3'/p) for p in old['code_sha256']}
    prompts = {r: sha_text(system_prompt(r)) for r in ('global', 'korean')}
    if hashes != old['code_sha256'] or prompts != old['prompt_sha256']:
        raise ValueError('Keep the Pilot-2 environment, rewards, prompts, and context unchanged')
    if config['policy']['context_limit'] != 8192 or config['policy']['generation_reserve'] != 1024:
        raise ValueError('Keep the accepted 8192/1024 context scheme')
    design = {'pilot_ids': ids, 'levels': dict(Counter(corpus[i]['level'] for i in ids)),
        'model': pinned, 'dated_snapshot_available_in_phase4': False, 'efforts': config[PHASE]['efforts'],
        'ordering': 'Pilot-2 retained order; alternate low and medium for every essay',
        'code_sha256': hashes, 'prompt_sha256': prompts,
        'contract_sha256': sha_text(json.dumps({k: config[k] for k in ('env', 'reward', 'policy', PHASE)}, sort_keys=True)),
        'corpus_sha256': manifest['agent_train']['sha256'], 'comparison_n': len(ids),
        'thresholds': {'global': .80, 'korean': .80},
        'best_luna_rule': 'highest projected total kept GLOBAL+KOREAN; tie: higher combined R, then lower cost'}
    if (root/'design.json').exists() and read_json(root/'design.json') != design:
        raise ValueError('Frozen comparison contract changed')
    write_json(root/'design.json', design)
    return design, corpus


def absolute_selection(row, corpus_row):
    has_global = any(r['level'] == 'GLOBAL' for r in corpus_row['records'])
    reward = row.get('reward') if row.get('completed') else None
    global_reward = reward['global'] if reward else row.get('global_only_reward')
    if global_reward is None:
        return {'global': False, 'korean': False, 'global_rule': 'incomplete_global_stage'}
    if has_global:
        keep_global = global_reward['R'] >= .80
        rule = 'GLOBAL_R_ge_0.80'
    else:
        keep_global = (row['termination'].get('global') == 'STOP' and row['steps']['global'] <= 2
            and global_reward['R_over'] == 0
            and not any(structural_action(a) for a in row['actions_by_role'].get('global', [])))
        rule = 'no_GLOBAL_STOP_rule'
    return {'global': keep_global, 'korean': bool(reward and reward['korean']['R'] >= .80), 'global_rule': rule}


def execute(config, api):
    design, corpus = prepare(config)
    root = config['paths'][PHASE+'_output']
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    if not all(corpus[i]['source_id'] in examples for i in design['pilot_ids']):
        raise ValueError('Wrong split or view-ineligible comparison source')
    resources = Resources(config, examples, output_key=PHASE+'_output')
    def one(item):
        id, effort = item
        condition = 'luna_'+effort
        path = root/condition/'episodes'/(safe_id(id)+'.json')
        # Completed and failed attempts are immutable. Never silently re-judge.
        if path.exists():
            return
        row = corpus[id]
        state = resources.worker()
        episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre', 'level',
            'records', 'corrupted_score', 'preexisting_spell_spans')}
        episode.update(source=resources.source(row['source_id']), document=resources.corrupted(row))
        env = RevisionEnv(config, mode='two_stage', analysis=state.analysis, scorer=resources,
            similarity=resources.similarity, tokenizer=state.tokenizer)
        result = run_episode(env, episode, Phase6Teacher(api, condition, max_output=1024, effort=effort),
            event_path=root/condition/'events'/(safe_id(id)+'.jsonl'))
        result['comparison_condition'] = condition
        # Include usage of incomplete/API-error responses which cannot become actions.
        with api.db() as db:
            paths = db.execute('SELECT path FROM calls WHERE stage=? AND item_id LIKE ?',
                (condition, result['episode_id']+':%')).fetchall()
        requests = [read_json(p[0]) for p in paths if p[0]]
        result['confirmed_episode_cost'] = sum(r.get('cost', {}).get('confirmed_usd', 0) for r in requests)
        result['api_request_count'] = len(requests)
        result['api_noncompleted'] = [{k: r.get(k) for k in ('phase_call', 'status', 'error_type', 'http_status')}
            for r in requests if r['status'] != 'completed']
        write_json(path, result)
        print(json.dumps({'condition': condition, 'id': id, 'completed': result['completed'],
            'termination': result['termination'], 'steps': result['steps'], 'error': result['runtime_error'],
            'cost': result['confirmed_episode_cost'], 'budget': api.accounting()}, ensure_ascii=False), flush=True)
        if result['runtime_error'] and result['runtime_error']['type'] == 'CallBudgetExceeded':
            raise RuntimeError('Shared $5/API-call budget reached; stop dispatch')
    errors = []
    try:
        tasks = [(id, e) for id in design['pilot_ids'] for e in design['efforts']]
        errors = bounded_map(tasks, one)
    finally:
        resources.close()
        write_json(root/'run_status.json', {'requested_per_setting': len(design['pilot_ids']),
            'errors': errors, 'budget': api.accounting()})
    return errors
