"""Paired rerun plus twenty new drop episodes; no bulk-generation path."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from statistics import mean

from ..common import read_json, write_json, file_sha, sha_text
from ..phase2 import read_jsonl, write_jsonl
from ..view_data import load_episode_examples
from ..agent.runner import system_prompt, run_episode
from ..env import RevisionEnv
from ..eval.api import Phase6Teacher
from ..eval.resources import Resources
from ..corrupt.document import Document, Paragraph, Unit
from ..reward.overedit import order_distance
from .pilot import safe_id, stratified_order
from .pilot2_data import PHASE, bounded_map


def prepare(config):
    root = config['paths'][PHASE+'_output']
    manifest = read_json(root/'corpus_manifest.json')
    if config['policy']['context_limit'] != 8192 or config['policy']['generation_reserve'] != 1024:
        raise ValueError('Teacher, SFT and policy must share 8192/1024')
    if config['env']['enable_check'] or config['env']['mode'] != 'two_stage':
        raise ValueError('Pilot 2 requires two_stage without CHECK')
    old = read_json(config['paths']['phase7_pilot_output']/'design.json')
    path = config['paths']['active_corrupt']/'agent_train.jsonl'
    if str(path) != manifest['agent_train']['final_path'] or file_sha(path) != manifest['agent_train']['final_sha256']:
        raise ValueError('Active corpus is not the finalized scope-v2 corpus')
    corpus = {r['episode_id']: r for r in read_jsonl(path)}
    retained = [id for id in old['pilot_ids'] if id in corpus]
    removed = [id for id in old['pilot_ids'] if id not in corpus]
    if set(removed) - set(manifest['agent_train']['removed_ids']):
        raise ValueError('An old pilot essay disappeared for another reason')
    old_sources = {corpus[id]['source_id'] for id in retained}
    new = [r for r in corpus.values() if r['episode_id'].startswith('phase7drop:')]
    chosen = stratified_order(new, config[PHASE]['seed'])[:20]
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    for row in corpus.values():
        if row['source_id'] not in examples or row['split'] != 'agent_train':
            raise ValueError('Wrong-split or view-ineligible source')
        if any(r['op'] == 'G_DELETE_SUPPORT' for r in row['records']):
            raise ValueError('Deleted support remains active')
    code = ('env/revision.py', 'agent/runner.py', 'reward/overedit.py', 'reward/total.py', 'reward/recovery.py')
    design = {'retained_pilot_ids': retained, 'removed_pilot_ids': removed,
        'new_drop_ids': chosen, 'pilot_ids': retained+chosen,
        'new_sampling_seed': config[PHASE]['seed'],
        'new_requested': 20, 'new_shortfall': 20-len(chosen),
        'shortfall_decision': 'User approved only passing cases; report shortfall, no extra candidates',
        'new_source_overlap_with_paired': sum(corpus[id]['source_id'] in old_sources for id in chosen),
        'counts_by_level': dict(Counter(corpus[id]['level'] for id in retained+chosen)),
        'corpus_sha256': file_sha(path), 'corpus_count': len(corpus), 'old_design_sha256': file_sha(config['paths']['phase7_pilot_output']/'design.json'),
        'prompt_sha256': {role: sha_text(system_prompt(role)) for role in ('global', 'korean')},
        'code_sha256': {p: file_sha(config['paths']['repo']/'verak/v3'/p) for p in code},
        'contract_sha256': sha_text(json.dumps({k: config[k] for k in ('env', 'reward', 'policy', PHASE)}, sort_keys=True)),
        'teacher': config['role_teacher'], 'context_limit': 8192, 'generation_reserve': 1024}
    target = root/'design.json'
    if target.exists() and read_json(target) != design:
        raise ValueError('Frozen pilot 2 design changed; do not reuse mismatched results')
    write_json(target, design)
    return design, corpus, examples


def execute(config, api):
    design, corpus, examples = prepare(config)
    root = config['paths'][PHASE+'_output']
    resources = Resources(config, examples, output_key=PHASE+'_output')
    def one(id):
        path = root/'episodes'/(safe_id(id)+'.json')
        if path.exists() and read_json(path).get('completed'):
            return
        row = corpus[id]
        state = resources.worker()
        episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre', 'level',
            'records', 'corrupted_score', 'preexisting_spell_spans')}
        episode.update(source=resources.source(row['source_id']), document=resources.corrupted(row))
        env = RevisionEnv(config, mode='two_stage', analysis=state.analysis, scorer=resources,
                          similarity=resources.similarity, tokenizer=state.tokenizer)
        result = run_episode(env, episode, Phase6Teacher(api, 'teacher_pilot2', max_output=1024),
            event_path=root/'events'/(safe_id(id)+'.jsonl'))
        write_json(path, result)
        print(json.dumps({'episode': id, 'completed': result['completed'], 'steps': result['steps'],
            'cost': result['cost_usd'], 'error': result['runtime_error'], 'budget': api.accounting()}, ensure_ascii=False), flush=True)
        if not result['completed']:
            raise RuntimeError(str(result['runtime_error']))
    errors = []
    try:
        errors = bounded_map(design['pilot_ids'], one)
    finally:
        resources.close()
        write_json(root/'run_status.json', {'requested': len(design['pilot_ids']),
            'errors': errors, 'budget': api.accounting()})
    if errors:
        raise RuntimeError('Pilot 2 stopped; inspect saved errors and budget before resuming')


def layout_document(snapshot):
    # Order-only computation never requires new morphological analysis.
    return Document([Paragraph(p['pid'], [Unit(u['sid'], u['text'], [], u['leading']) for u in p['units']])
        for p in snapshot['paragraphs']], snapshot['gaps'], snapshot['tail'])


def recompute_old(row, corpus):
    result = deepcopy(row)
    source = layout_document(corpus['source_layout'])
    initial, middle, final = [layout_document(row[k]) for k in ('initial_layout', 'stage1_layout', 'final_layout')]
    for role, before, after in [('global', initial, middle), ('korean', middle, final), ('combined', source, final)]:
        reward = result['reward'][role]
        morpheme = reward['R_over']
        order = order_distance(source, before, after, corpus['records'],
            actions=row['actions_by_role'][role] if role != 'combined' else None)
        updated = .5*morpheme + .5*order['value']
        weight = -reward['weighted_components']['overedit']/morpheme if morpheme else .5
        reward['R'] += weight*(morpheme-updated)
        reward['R_over'] = updated
        reward['weighted_components']['overedit'] = -weight*updated
        reward['overedit'].update(value=updated, morpheme=morpheme, order=order['value'], order_details=order)
    result['recomputed_order_reward_only'] = True
    return result


def diagnostic(config):
    from ..eval.ending_diagnostic import run
    return run(config)


def report(config):
    from .pilot2_report import summarize
    return summarize(config)
