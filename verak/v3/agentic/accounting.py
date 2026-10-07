"""Terminal reward accounting, independent of every policy observation."""
from copy import deepcopy

FREE_ACTIONS = {'SCORE', 'QUERY', 'AUDIT', 'PREVIEW', 'PLAN', 'PROGRESS'}


def combined_steps(actions):
    """Keep the existing all-action cost except the task's explicit exemptions.

    Editor-specific rewards charge writes only. Combined reporting additionally
    includes delegation/report/finish control actions, as the v2 task exempts
    only read tools and ledgers from the established R_step definition.
    """
    return sum(a['action'] not in FREE_ACTIONS for a in actions)


def audit_reward(row):
    result = deepcopy(row)
    if not result.get('reward'):
        return result
    combined = result['reward']['combined']
    expected = combined_steps(result['actions'])
    old = combined['R_step']
    if old != expected:
        delta = -.01 * (expected - old)
        result['reward_step_audit'] = {'original_steps': old, 'corrected_steps': expected, 'R_delta': delta,
                                       'exempt_actions': sorted(FREE_ACTIONS), 'policy_inputs_affected': False}
        combined['R_step'] = expected
        combined['R'] += delta
        combined['weighted_components']['steps'] = -.01 * expected
        result['reward']['orchestrator']['R'] += delta
        result['reward']['orchestrator']['final_combined_R'] = combined['R']
    return result
