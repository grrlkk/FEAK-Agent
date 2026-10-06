"""Smoke summaries separate protocol completion from revision quality."""
from collections import Counter
from statistics import mean


def summarize(episodes, requested):
    actions = [a for e in episodes for values in e['actions_by_role'].values() for a in values]
    calls = [c for e in episodes for c in e['calls']]
    completed = [e for e in episodes if e['completed']]
    def reward_stats(rows):
        result = {}
        for role in ('global', 'korean', 'combined'):
            values = [e['reward'][role] for e in rows if e.get('reward') and e['reward'][role] is not None]
            result[role] = {k: mean(v[k] for v in values) for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')} if values else None
        return result
    roles = sorted({role for e in episodes for role in e['steps']})
    tokens = {k: sum(e['usage'].get(k, 0) for e in episodes) for k in ('input', 'output', 'reasoning', 'cache_read', 'cache_write')}
    return {'requested': requested, 'attempted': len(episodes), 'completed': len(completed),
        'completion_rate': len(completed)/requested,
        'completion_definition': 'all stages terminated and terminal reward computed; errors reason reported separately',
        'roles': {role: {'episodes': sum(role in e['steps'] for e in episodes),
            'mean_steps': mean(e['steps'].get(role, 0) for e in episodes),
            'mean_checks': mean(e['checks'].get(role, 0) for e in episodes),
            'termination': dict(Counter(e['termination'].get(role, 'not_completed') for e in episodes))} for role in roles},
        'actions': len(actions), 'invalid_actions': sum(not a['valid'] for a in actions),
        'invalid_action_rate': sum(not a['valid'] for a in actions)/max(1, len(actions)),
        'role_violations': sum(a['error_code'] == 'role_forbidden' for a in actions),
        'error_codes': dict(Counter(a['error_code'] for a in actions if not a['valid'])),
        'model_turns': len(calls), 'json_valid_share': sum(c['json_valid'] for c in calls)/max(1, len(calls)),
        'valid_json_action_share': sum(c['valid_json_action'] for c in calls)/max(1, len(calls)),
        'history_compactions': sum(c['history_compacted'] for c in calls),
        'rewards': reward_stats(completed),
        'by_level': {level: {'n': sum(e.get('level') == level for e in completed),
            'rewards': reward_stats([e for e in completed if e.get('level') == level])} for level in ('L1', 'L2', 'L3', 'L4')},
        'recovery_by_linguistic_level': {level: {'records': len(v), 'main': mean(v) if v else None}
            for level in ('GLOBAL', 'WORD', 'SENTENCE', 'TEXT')
            for v in [[r['main'] for e in completed if e.get('reward')
                       for r in e['reward']['combined']['per_record'] if r['level'] == level]]},
        'tokens': tokens, 'cached_token_share': tokens['cache_read']/max(1, tokens['input']),
        'cache_write_share': tokens['cache_write']/max(1, tokens['input']),
        'cost_usd': sum(e['cost_usd'] for e in episodes),
        'cost_per_episode_usd': mean(e['cost_usd'] for e in episodes) if episodes else None,
        'mean_elapsed_s': mean(e['elapsed_s'] for e in episodes) if episodes else None,
        'runtime_errors': [{'episode_id': e['episode_id'], **e['runtime_error']} for e in episodes if e.get('runtime_error')]}
