"""Saved reporting distinguishes executed edits, budget returns and evidence gaps."""
from copy import deepcopy

import pytest

from verak.v3.agentic.audit import audit_v3_row
from verak.v3.agentic.report import deletion_summary, orchestration, render, summary


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


@pytest.mark.parametrize('version', [2, 3])
def test_report_uses_versioned_contract_and_destination(tmp_path, version):
    root = tmp_path / 'verak/v3/outputs' / ('agentic_pilot_v3' if version == 3 else 'agentic_pilot')
    (tmp_path / 'imple/reports').mkdir(parents=True)
    root.mkdir(parents=True)
    config = {'paths': {'repo': tmp_path, 'agentic_pilot_output': root},
              'agentic_pilot': {'version': version, 'max_cost_usd': 5 if version == 3 else 6}}
    m = {key: {} for key in ('question_audit', 'graph_quality', 'graph_status', 'prompt_tokens', 'off_topic_query_association',
          'low_use_tools', 'by_level', 'markers', 'baseline_marker_metrics', 'relevance', 'relevance_paired', 'evaluation_stages',
          'export', 'validation', 'environment_preflight', 'protocol_preflight')}
    m.update(recommendation='current_two_stage', orchestration=orchestration([]), drop_tool_candidates=[],
             paired={}, budget={'confirmed_usd': 0, 'reserved_usd': 0, 'pending': 0},
             baseline_hashes_unchanged=True, all_observations_have_question=True,
             runtime_files_unchanged=True, test_result='focused tests passed')
    m['new'] = {c: summary([], n) for c, n in [('corrupted', 92), ('real', 30)]}
    m['baseline'] = {c: summary([], n, baseline=True) for c, n in [('corrupted', 92), ('real', 30)]}
    render(config, m, [])
    filename = 'V3_AGENTIC_PILOT_V3.md' if version == 3 else 'V3_AGENTIC_PILOT.md'
    text = (tmp_path / 'imple/reports' / filename).read_text()
    assert f'# V3 Agentic Pilot (v{version})' in text
    assert 'DELETE / completed essay' in text
    if version == 3:
        assert 'done (budget)' in text and 'at most three DELEGATE' in text
        assert 'has no unsupported list' in text and 'same state and delegation' in text
        assert '12 started essays' not in text and 'Pre-intersection validation failures were preserved' not in text
        assert 'No bulk generation, SFT or RFT was run.' not in text
