"""Saved reporting distinguishes executed edits, budget returns and evidence gaps."""
from copy import deepcopy

import pytest

from verak.v3.agentic.audit import audit_v3_row, audit_snapshots, expected_protection
from verak.v3.agentic.report import deletion_summary, orchestration, render, summary, routing_summary, paired_steps, paired_interpretation
from verak.v3.common import sha_text
from verak.v3.agentic.redelegation import summarize_cases


def test_deletion_denominators_include_failures_and_do_not_refund_undo():
    actions = [{'action': 'DELETE', 'args': {'target': 'S2'}, 'valid': True},
               {'action': 'UNDO', 'args': {}, 'valid': True},
               {'action': 'DELETE', 'args': {'target': 'S1-S4'}, 'valid': False}]
    rows = [{'episode_id': 'a', 'completed': True, 'actions': actions},
            {'episode_id': 'b', 'completed': False, 'actions': []}]
    new = deletion_summary(rows, 3)
    assert new['valid_deletions'] == new['rejected_deletion_attempts'] == 1
    assert new['per_attempted_episode'] == .5
    assert new['per_completed_episode'] == 1
    assert new['per_intended_episode'] == pytest.approx(1 / 3)
    baseline = deepcopy(rows)
    for row in baseline:
        row['actions_by_role'] = {'global': row.pop('actions')}
        for action in row['actions_by_role']['global']:
            if action['action'] == 'DELETE':
                action['action'] = 'EDIT'
                action['args']['new_text'] = ''
    old = deletion_summary(baseline, 3, baseline=True)
    for field in ('valid_deletions', 'rejected_deletion_attempts', 'per_attempted_episode', 'per_completed_episode'):
        assert new[field] == old[field]
    assert summary(baseline, 3, baseline=True)['termination'] == {'completed': 1, 'unfinished': 1}


def test_editor_return_rates_keep_failed_turns_out_of_success_counts():
    def seq(role, delegation, terminal, notice, auto=False):
        return {'role': role, 'delegation': delegation, 'terminal': terminal,
                'final_notice_sent': notice, 'auto_report': auto}
    rows = [{'episode_id': 'a', 'actions': [], 'delegations': 3, 'sequences': [
                seq('composition', 1, 'AUTO_REPORT', True, True),
                seq('composition', 2, 'REPORT', True), seq('cohesion', 3, 'REPORT', False)]},
            {'episode_id': 'b', 'actions': [], 'delegations': 1, 'sequences': [seq('composition', 1, None, True)]}]
    result = orchestration(rows)
    comp = result['editor_returns']['composition']
    assert comp['delegations'] == 3 and comp['auto_reports'] == 1 and comp['unfinished'] == 1
    assert comp['auto_report_rate_per_delegation'] == pytest.approx(1 / 3)
    assert comp['explicit_report_rate_after_final_notice'] == pytest.approx(1 / 3)
    assert result['editor_returns']['all_editors']['auto_report_rate_per_delegation'] == .25
    assert result['auto_report_episodes'] == ['a']


def test_preview_coverage_requires_ids_and_order_not_only_identical_surface_text():
    rows = [{'sid': sid, 'paragraph': 'P1', 'text': '같다.' if sid != 'S3' else '끝.'}
            for sid in ('S1', 'S2', 'S3')]
    moved = [rows[1], rows[0], rows[2]]
    target = {'target': 'S1', 'position': 'after:S3'}
    common = {'role': 'composition', 'valid': True, 'delegation': 1, 'before_hash': 'same-surface'}
    actions = [{**common, 'action': 'PREVIEW', 'args': {'action': {'action': 'MOVE', 'args': target}}, 'before_rows': rows},
               {**common, 'action': 'MOVE', 'args': {'target': 'S1', 'position': 'after:S2'}, 'before_rows': rows},
               {**common, 'action': 'MOVE', 'args': target, 'before_rows': moved},
               {'role': 'orchestrator', 'action': 'DELEGATE', 'args': {}, 'valid': False, 'error': 'delegation cap'}]
    result = orchestration([{'episode_id': 'a', 'actions': actions, 'sequences': []}])
    assert result['preview_rate'] == 0
    assert result['tool_outcomes']['DELEGATE'] == {'attempted': 1, 'executed': 0, 'rejected': 1}
    assert result['tool_outcomes']['MOVE'] == {'attempted': 2, 'executed': 2, 'rejected': 0}


def test_saved_audit_rejects_deletion_without_named_target_or_matching_state():
    before = [{'sid': 'S1', 'paragraph': 'P1', 'text': '하나.'}, {'sid': 'S2', 'paragraph': 'P1', 'text': '둘.'}]
    initial = {'off_topic_candidate': ['S2'], 'sentences': [
        {'id': 'S1', 'predecessor': None}, {'id': 'S2', 'predecessor': 'S1'}]}
    delegate_args = {'agent': 'composition', 'task': 'S2 삭제', 'scope': 'all'}
    delete_args = {'target': 'S2'}
    report = {'agent': 'composition', 'origin': 'agent', 'status': 'done', 'summary': '완료'}
    def action(name, args, result, delegation=1, role='composition', after=None):
        return {'action': name, 'args': args, 'valid': True, 'role': role, 'delegation': delegation,
                'before_rows': deepcopy(before), 'after_rows': deepcopy(before if after is None else after), 'result': result}
    row = {'episode_id': 'x', 'version': 3, 'score_calls': 0, 'delegations': 1,
           'calls': [{'role': 'composition', 'delegation': 1, 'turn': n} for n in range(1, 4)], 'runtime_error': None,
           'sequences': [{'role': 'composition', 'delegation': 1, 'task': 'S2 삭제', 'scope': 'all',
                          'terminal': 'REPORT', 'auto_report': False, 'final_notice_sent': False, 'report': report}],
           'actions': [action('DELEGATE', delegate_args, {'disturbed_markers': []}, 0, 'orchestrator'),
                       action('PREVIEW', {'action': {'action': 'DELETE', 'args': delete_args}}, {}),
                       action('DELETE', delete_args, {'marker_changes': [], 'off_topic_candidate': []}, after=before[:1]),
                       action('REPORT', {'status': 'done', 'summary': '완료'}, report)]}
    def failures(value):
        bad = []
        audit_v3_row(value, initial, lambda passed, name, item: bad.append(name) if not passed else None)
        return bad
    assert failures(row) == []
    corrupt = deepcopy(row)
    corrupt['sequences'][0]['task'] = corrupt['actions'][0]['args']['task'] = 'S1 확인'
    corrupt['actions'][1]['before_rows'][0]['text'] = '예전 내용.'
    assert {'delete_target_named_in_task', 'delete_exact_preview_in_same_state'} <= set(failures(corrupt))


def test_protection_audit_traces_union_children_only_two_hops():
    def edge(source, target, label):
        return {'source': source, 'target': target, 'label': label}
    left = {'sentence_edges': [edge('S1', 'Q', 'addresses'), edge('S2', 'S1', 'supports'),
                               edge('S4', 'S3', 'contrasts'), edge('S1', 'S5', 'example_of')],
            'off_topic': ['S1', 'S2', 'S3', 'S4', 'S5']}
    right = {'sentence_edges': [edge('S3', 'S2', 'example_of')], 'off_topic': ['S2', 'S3', 'S4', 'S5']}
    result = expected_protection(left, right, ['S1', 'S2', 'S3', 'S4', 'S5'])
    assert result == {'seeds': ['S1'], 'protected_ids': ['S1', 'S2', 'S3'],
                      'depths': {'S1': 0, 'S2': 1, 'S3': 2}, 'overridden_off_topic': ['S2', 'S3']}


def test_snapshot_audit_checks_action_prefix_and_fresh_audit():
    layout = {'gaps': [''], 'tail': '', 'analyzer_version': 'test',
              'paragraphs': [{'pid': 'P1', 'units': [{'sid': 'S1', 'leading': '', 'text': '글.'}]}]}
    def action(name, role, delegation):
        return {'action': name, 'valid': True, 'role': role, 'delegation': delegation,
                'before_hash': sha_text('글.'), 'after_hash': sha_text('글.')}
    row = {'episode_id': 'x', 'runtime_error': None, 'actions': [action('DELEGATE', 'orchestrator', 0),
        action('REPORT', 'composition', 1), action('AUDIT', 'orchestrator', 1),
        action('DELEGATE', 'orchestrator', 1), action('REPORT', 'cohesion', 2)],
        'sequences': [{'role': role, 'delegation': n, 'terminal': 'REPORT', 'before_layout': layout,
            'after_layout': layout, 'before_action_count': before, 'after_action_count': after,
            'audit_before_redelegation': bool(audits), 'redelegation_audit_action_indices': audits}
            for role, n, before, after, audits in [('composition', 1, 0, 2, []), ('cohesion', 2, 3, 5, [2])]]}
    def failures(value):
        errors = []
        audit_snapshots(value, lambda passed, label, item: errors.append(label) if not passed else None)
        return errors
    assert failures(row) == []
    corrupted = deepcopy(row)
    corrupted['sequences'][1]['before_action_count'] = 4
    assert 'snapshot_before_excludes_delegate' in failures(corrupted)
    corrupted = deepcopy(row)
    corrupted['sequences'][1]['redelegation_audit_action_indices'] = []
    assert 'fresh_audit_redelegation_trigger' in failures(corrupted)


def test_routing_and_steps_keep_failures_rejections_and_editor_identity_separate():
    def action(name, valid=True, agent=None):
        return {'action': name, 'valid': valid, 'args': {'agent': agent}, 'before_rows': [], 'result': {}}
    rows = [{'episode_id': 'a', 'completed': True, 'actions': [action('AUDIT'), action('DELEGATE', False), action('FINISH')]},
            {'episode_id': 'b', 'completed': False, 'actions': [action('DELEGATE', agent='composition')]},
            {'episode_id': 'c', 'completed': True, 'actions': [action('DELEGATE', agent='composition'), action('REPORT'),
                action('AUDIT'), action('DELEGATE', agent='cohesion'), action('REPORT'), action('AUDIT'),
                action('DELEGATE', agent='composition'), action('REPORT'), action('FINISH')],
             'sequences': [{'role': role, 'delegation': n, 'terminal': 'REPORT', 'audit_before_redelegation': n > 1}
                           for n, role in enumerate(('composition', 'cohesion', 'composition'), 1)]}]
    result = routing_summary(rows, 4)
    assert result['completed'] == 2 and result['attempted'] == 3
    assert result['completed_routing']['skipped_composition'] == {'count': 1, 'share': .5}
    assert result['completed_routing']['finish_after_audit_zero_delegation']['count'] == 1
    assert result['redelegations_after_audit'] == 2 and result['repeated_same_editor'] == result['first_use_of_other_editor'] == 1
    assert result['all_observed_routing']['no_composition_delegation'] == {
        'count': 1, 'attempted_denominator': 3, 'intended_denominator': 4,
        'share_attempted': pytest.approx(1 / 3), 'share_intended': .25}
    assert result['all_observed_routing']['failed_without_cohesion']['count'] == 1
    baseline = [{'episode_id': 'a', 'completed': True, 'actions_by_role': {'global': [action('STOP')], 'korean': [action('STOP')]}}]
    steps = paired_steps(rows, baseline)
    assert steps['n'] == 1 and steps['raw_actions'] == {'new': 3, 'baseline': 2, 'delta': 1}
    assert steps['charged_steps'] == {'new': 2, 'baseline': 2, 'delta': 0}


def test_redelegation_reward_summary_never_counts_unavailable_as_zero():
    common = {'role': 'composition', 'editor_use': 'repeated_same_editor', 'level': 'L2'}
    rows = [{**common, 'episode_id': 'a', 'episode_completed': True, 'status': 'available',
             'delta': {'R': .4, 'R_rec': .5, 'R_q': 0, 'R_over': 0, 'R_step': 10}},
            {**common, 'episode_id': 'b', 'episode_completed': False, 'status': 'unavailable'}]
    result = summarize_cases(rows)
    assert result['by_level']['L2']['eligible_redelegations'] == 2
    assert result['by_level']['L2']['evaluated'] == result['by_level']['L2']['unavailable'] == 1
    assert result['by_level']['L2']['mean_delta']['R'] == .4
    assert result['by_level']['L1']['mean_delta']['R'] is None
    assert result['completed_episodes_only']['eligible_redelegations'] == 1


def test_interpretation_counts_executed_deletions_and_selects_failure_loops_from_data():
    layout = {'paragraphs': [{'units': [{'sid': sid, 'text': sid} for sid in ('S1', 'S2', 'S3')]}]}
    records = [{'record_id': key, 'op': op, 'level': level, 'sids': [sid]}
               for key, op, level, sid in [('one', 'L_REGISTER', 'TEXT', 'S1'), ('two', 'L_SPACING', 'WORD', 'S2')]]
    def reward(recovery, steps):
        return {'combined': {'R': recovery - .01 * steps, 'weighted_components': {
            'recovery': recovery, 'quality': 0, 'overedit': 0, 'steps': -.01 * steps},
            'per_record': [{**r, 'main': recovery, 'recovery': recovery} for r in records]}}
    actions = [{'action': name, 'valid': True, 'changed_sids': ['S3'] if name == 'DELETE' else []}
               for name in ('DELETE', 'UNDO', 'DELETE', 'REPORT')]
    completed = {'episode_id': 'paired', 'completed': True, 'reward': reward(0, 4), 'actions': actions,
                 'sequences': [{'role': 'composition'}], 'final_layout': layout}
    baseline = {'episode_id': 'paired', 'completed': True, 'reward': reward(1, 2),
                'actions_by_role': {'global': [{'action': 'STOP'}], 'korean': [{'action': 'STOP'}]}}
    failure = {'episode_id': 'automatically_selected', 'completed': False, 'termination': 'orchestrator_budget',
        'sequences': [{'role': 'composition', 'delegation': 1, 'terminal': 'REPORT'}],
        'actions': [{'role': 'composition', 'delegation': 1, 'valid': False, 'action': 'PREVIEW',
                     'args': {'action': {'action': 'DELETE', 'args': {'target': 'S1-S3'}}}, 'error': 'invalid target'} for _ in range(3)] +
                   [{'role': 'composition', 'delegation': 1, 'valid': True, 'action': 'REPORT'}] +
                   [{'role': 'orchestrator', 'delegation': 1, 'valid': True, 'action': 'AUDIT', 'result': {}, 'after_hash': 'same'} for _ in range(3)]}
    corpus = {eid: {'records': records if eid == 'paired' else [], 'source_layout': layout, 'corrupted_layout': layout}
              for eid in ('paired', 'automatically_selected')}
    result = paired_interpretation([completed, failure], [baseline], corpus)
    assert result['weighted_component_deltas']['recovery'] == -1
    assert sum(v['contribution_to_mean_R_rec_delta'] for v in result['paired_operators'].values()) == -1
    assert result['local_coverage']['fully_recovered_baseline'] == 2
    assert result['executed_deletion_categories_all_saved']['unreferenced_source'] == 2
    assert result['failures_with_all_editors_returned'] == result['failures_ending_on_audit'] == 1
    assert all(v['episode_id'] == 'automatically_selected' and v['count'] == 3 for v in result['representative_loops'].values())


@pytest.mark.parametrize('version,protected', [(2, False), (3, False), (3, True)])
def test_report_uses_versioned_contract_and_destination(tmp_path, version, protected):
    root = tmp_path / 'verak/v3/outputs' / ('agentic_pilot_v3_protected' if protected else 'agentic_pilot_v3' if version == 3 else 'agentic_pilot')
    (tmp_path / 'imple/reports').mkdir(parents=True)
    root.mkdir(parents=True)
    config = {'paths': {'repo': tmp_path, 'agentic_pilot_output': root},
              'agentic_pilot': {'version': version, 'max_cost_usd': 7 if protected else 5 if version == 3 else 6}}
    m = {key: {} for key in ('question_audit', 'graph_quality', 'graph_status', 'prompt_tokens', 'off_topic_query_association',
          'low_use_tools', 'by_level', 'markers', 'baseline_marker_metrics', 'relevance', 'relevance_paired', 'evaluation_stages',
          'export', 'validation', 'environment_preflight', 'protocol_preflight')}
    m.update(recommendation='current_two_stage', orchestration=orchestration([]), drop_tool_candidates=[],
             paired={}, budget={'confirmed_usd': 0, 'reserved_usd': 0, 'pending': 0},
             baseline_hashes_unchanged=True, all_observations_have_question=True,
             runtime_files_unchanged=True, test_result='focused tests passed')
    m['new'] = {c: summary([], n) for c, n in [('corrupted', 92), ('real', 30)]}
    m['baseline'] = {c: summary([], n, baseline=True) for c, n in [('corrupted', 92), ('real', 30)]}
    if protected:
        m.update(initial_run={'confirmed_usd': 1.4}, budget_runs={}, graph_reuse={}, relevance_protection={'graphs': 226})
    render(config, m, [])
    filename = 'V3_AGENTIC_PILOT_V3.md' if version == 3 else 'V3_AGENTIC_PILOT.md'
    text = (tmp_path / 'imple/reports' / filename).read_text()
    assert f'# V3 Agentic Pilot (v{version})' in text
    assert 'DELETE / completed essay' in text
    if version == 3:
        assert 'done (budget)' in text and 'at most three accepted DELEGATE' in text
        assert 'has no unsupported list' in text and 'same state and delegation' in text
        assert '12 started essays' not in text and 'Pre-intersection validation failures were preserved' not in text
        assert 'No bulk generation, SFT or RFT was run.' not in text
    if protected:
        assert 'The $7 A cap includes both runs' in text
        assert 'at most two hops' in text and 'Ordinary sentence and paragraph relations contain only the intersection' in text
        assert 'Reward change across re-delegations' in text
