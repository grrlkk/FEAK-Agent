"""Existing recovery/over-edit definitions with pilot-specific role costs."""
from copy import deepcopy
from statistics import mean

from ..reward.recovery import recover_records
from ..reward.overedit import overedit
from ..reward.total import rewards
from .environment import WRITES


def reward_actions(actions):
    result = []
    for a in actions:
        if a['action'] not in WRITES:
            continue
        b = deepcopy(a)
        if b['action'] in {'INSERT', 'DELETE'}:
            b['action'] = 'EDIT'
            if a['action'] == 'DELETE':
                b['args']['new_text'] = ''
        result.append(b)
    return result


def editor_reward(env, role, before, scope_ids, actions, config):
    ep = env.episode
    if 'records' not in ep:
        return None
    records = deepcopy(ep['records'])
    if role == 'cohesion':
        for r in records:
            r['coupled_changes'] = [c for c in r.get('coupled_changes', []) if c['sid'] in scope_ids]
    recovered = recover_records(ep['source'], env.document, records, corrupted=env.corrupted,
                                tau=config['similarity']['tau'], coupled_weight=config['reward']['dependents_weight'])
    if role == 'composition':
        terms = {r['record_id'] + ':main': r['main'] for r in recovered if r['level'] == 'GLOBAL'}
    else:
        eligible = {r['record_id'] for r in records if set(r['sids']) & scope_ids}
        terms = {r['record_id'] + ':main': r['main'] for r in recovered if r['level'] != 'GLOBAL' and r['record_id'] in eligible}
        terms.update({r['record_id'] + ':coupled': r['coupled'] for r in recovered if r['level'] == 'GLOBAL' and r['coupled'] is not None})
    writes = reward_actions(actions)
    over = overedit(ep['source'], before, env.document, ep['records'], actions=writes,
                    preexisting_spell_spans=ep.get('preexisting_spell_spans', ()))
    rec = mean(terms.values()) if terms else 0.
    return {'R': rec - .5 * over['value'] - .01 * len(writes), 'R_rec': rec,
            'R_over': over['value'], 'R_step': len(writes), 'recovery_terms': terms, 'overedit': over}


def final_rewards(env, sequences, config):
    ep = env.episode
    if 'records' not in ep:
        return None
    combined = rewards(ep['source'], env.corrupted, env.document, ep['records'], genre=ep['genre'],
        q_corrupted=ep['corrupted_score']['mean'], q_final=env.score()['mean'], config=config['reward'],
        actions=reward_actions(env.actions), tau=config['similarity']['tau'],
        preexisting_spell_spans=ep.get('preexisting_spell_spans', ()))['combined']
    result = {'combined': combined, 'orchestrator': {'R': combined['R'] - .02 * env.delegations,
                                                    'final_combined_R': combined['R'], 'delegations': env.delegations}}
    for role in ('composition', 'cohesion'):
        values = [s['reward'] for s in sequences if s['role'] == role and s.get('reward') is not None]
        # The latest observed recovery for each record/term, and the sum of costs
        # from disjoint editor invocations. Never charge inherited role changes.
        terms = {}
        for value in values:
            terms.update(value['recovery_terms'])
        rec = mean(terms.values()) if terms else 0.
        over, steps = sum(v['R_over'] for v in values), sum(v['R_step'] for v in values)
        result[role] = {'R': rec - .5 * over - .01 * steps, 'R_rec': rec, 'R_over': over,
                        'R_step': steps, 'recovery_terms': terms, 'invocations': len(values)}
    return result
