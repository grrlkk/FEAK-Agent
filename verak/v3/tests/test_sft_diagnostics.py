"""Saved-evidence diagnostics: no models, GPUs, or API calls."""
from verak.v3.train.sft_composition import composition_summary, global_inaction, trajectory_details


def test_global_inaction_keeps_failed_global_decisions_and_unknown_denominators():
    corpus = {str(i): {'records': [{'level': 'GLOBAL'}]} for i in range(5)}
    corpus['local'] = {'records': [{'level': 'WORD'}]}
    def row(i, actions=(), termination='STOP', completed=True):
        return {'corpus_episode_id': str(i), 'completed': completed,
            'termination': {'global': termination},
            'actions_by_role': {'global': list(actions), 'korean': [{'action': 'MOVE', 'valid': True}]}}
    rows = [row(0), row(1, [{'action': 'MOVE', 'valid': False}]),
            row(2, [{'action': 'EDIT', 'args': {'target': 'S2', 'new_text': ''}, 'valid': True}]),
            row(3, termination='max_steps'), row('local')]
    result = global_inaction(rows, corpus, list(corpus))
    assert result['intended_with_global_records'] == 5 and result['unknown'] == 1
    attempted = result['STOP_without_structural_attempt']
    accepted = result['STOP_without_accepted_structural_action']
    assert attempted['n'] == 1 and attempted['rate'] is None
    assert (attempted['lower_bound'], attempted['upper_bound']) == (.2, .4)
    assert accepted['n'] == 2 and accepted['rate'] is None
    rows.append(row(4, completed=False))  # KOREAN failed after an observed GLOBAL STOP.
    result = global_inaction(rows, corpus, list(corpus))
    assert result['unknown'] == 0
    assert result['STOP_without_structural_attempt']['rate'] == .4
    assert result['STOP_without_accepted_structural_action']['rate'] == .6


def test_composition_counts_trajectories_separately_from_records_and_partial_recovery():
    records = [{'record_id': 'a', 'level': 'GLOBAL', 'op': 'G_PARA_SWAP'},
               {'record_id': 'b', 'level': 'GLOBAL', 'op': 'G_PARA_SWAP'},
               {'record_id': 'c', 'level': 'GLOBAL', 'op': 'G_SENT_MOVE'}]
    reward = {'per_record': [{'record_id': 'a', 'main': 1}, {'record_id': 'b', 'main': .5},
                             {'record_id': 'c', 'main': 1}]}
    row = {'corpus_episode_id': 'one', 'source_id': 'essay', 'completed': True,
           'termination': {'global': 'STOP'}, 'actions_by_role': {'global': [
               {'action': 'STOP', 'valid': False}, {'action': 'STOP', 'valid': True}]}}
    detail = trajectory_details(row, 'global', records, reward)
    result = composition_summary([detail], 'global')
    assert result['STOP_only'] == 1 and result['single_valid_STOP_only'] == 0
    assert result['STOP_only_with_global_records'] == 1
    op = result['operators']['G_PARA_SWAP']
    assert op['selected_trajectories_with_operator'] == 1 and op['records'] == 2
    assert op['at_least_one_fully_recovered'] == 1 and op['all_fully_recovered'] == 0
    assert op['fully_recovered_records'] == 1 and op['partially_recovered_records'] == 1
    assert result['operators']['G_SENT_MOVE']['all_fully_recovered'] == 1
