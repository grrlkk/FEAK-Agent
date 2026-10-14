"""GLOBAL eligibility and source caps for the approved diverse-data continuation."""
from collections import defaultdict
from copy import deepcopy

from ..env.protocol import parse_action


def eligible(row):
    reward = (row.get('reward') or {}).get('global') or row.get('global_only_reward')
    actions = row.get('actions_by_role', {}).get('global', [])
    if reward is None or reward['R'] < .80:
        return False
    return (row.get('termination', {}).get('global') == 'STOP' and bool(actions)
        and actions[-1]['action'] == 'STOP' and actions[-1].get('valid', False)
        and sum(not a.get('valid', False) for a in actions) <= 1)


def source_cap(selected, *, cap=4):
    groups = defaultdict(list)
    for eid, entry in selected.items():
        if entry.get('score_source') != 'gpu_reference':
            raise ValueError('Source-cap ranking must use GPU-reference rewards exclusively')
        groups[(entry['source_id'], entry['operator'])].append((eid, entry))
    kept, excluded = {}, []
    for (source, operator), values in sorted(groups.items()):
        values.sort(key=lambda pair: (-pair[1]['R'], -pair[1].get('R_rec', 0),
                    pair[1].get('R_over', 0), pair[0], pair[1]['attempt']))
        kept.update(values[:cap])
        excluded.extend({'episode_id': eid, 'source_id': source, 'operator': operator,
                         'R': entry['R'], 'reason': 'max4_practices_per_source_operator'}
                        for eid,entry in values[cap:])
    return kept, excluded


def rescue_action_targets(row):
    """Keep failed format attempts as context, never as Sol action-loss targets."""
    result=deepcopy(row)
    calls=[];masked=[]
    for index,call in enumerate(result['calls']):
        if call.get('role')=='global':
            try:
                parse_action(call['raw'])
            except ValueError:
                masked.append({'call_index':index,'turn':call['turn'],'reason':'not_action_JSON'})
                continue
        calls.append(call)
    result['calls']=calls
    result['action_only_export']={'policy':'sol_global_action_json_targets_v1',
        'scope':'GLOBAL malformed parse attempts excluded as targets; retained in subsequent context',
        'masked_non_action_targets':masked,
        'historical_messages_unchanged':True,'actions_and_rewards_unchanged':True}
    return result
