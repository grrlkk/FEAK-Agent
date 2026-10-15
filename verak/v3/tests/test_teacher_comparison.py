from copy import deepcopy
from types import SimpleNamespace

import pytest

from verak.v3.common import load_config
from verak.v3.eval.api import Phase6API, Phase6Teacher
from verak.v3.train.teacher_comparison import without_drops, absolute_selection
from verak.v3.train.teacher_diagnostics import visibility, disturbance
from verak.v3.train.pilot2_data import build, judge, finalize
from feak_tc.runtime.openai import CallBudgetExceeded


def test_removed_essays_are_not_replaced_or_modified():
    rows = [{'episode_id': 'a', 'records': [{'op': 'L_CONN'}]},
            {'episode_id': 'b', 'records': [{'op': 'L_CONJ_DROP'}, {'op': 'L_REGISTER'}]}]
    old = deepcopy(rows)
    assert without_drops(rows) == [rows[0]] and rows == old


def test_disabled_operator_cannot_generate_judge_or_readd():
    config = load_config()
    for function, args in [(build, (config,)), (judge, (config, None)), (finalize, (config,))]:
        with pytest.raises(ValueError, match='disabled'):
            function(*args)


def test_absolute_selection_and_no_global_attempt_rule():
    row = {'completed': True, 'reward': {'global': {'R': .80, 'R_over': 0}, 'korean': {'R': .799}},
        'steps': {'global': 1}, 'termination': {'global': 'STOP'}, 'actions_by_role': {'global': []}}
    assert absolute_selection(row, {'records': [{'level': 'GLOBAL'}]})['global']
    assert not absolute_selection(row, {'records': [{'level': 'GLOBAL'}]})['korean']
    row['reward']['global']['R'] = -.01
    row['reward']['korean']['R'] = .80
    assert absolute_selection(row, {'records': []})['global']
    assert absolute_selection(row, {'records': []})['korean']
    row['actions_by_role']['global'] = [{'action': 'MOVE', 'args': {}, 'valid': False}]
    assert not absolute_selection(row, {'records': []})['global']
    row['completed'] = False
    assert not absolute_selection(row, {'records': []})['korean']


def test_luna_settings_share_budget_and_pass_exact_effort(tmp_path, monkeypatch):
    from verak.v3.eval import api as module
    calls = []
    class Adapter:
        def __init__(self, config):
            self._client = SimpleNamespace(responses=SimpleNamespace(create=self.create))
        def _load(self):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            usage = {'input_tokens': 100, 'output_tokens': 20,
                'input_tokens_details': {}, 'output_tokens_details': {'reasoning_tokens': 10}}
            return SimpleNamespace(output_text='{}', status='completed', id='fake', model='gpt-6-luna',
                usage=SimpleNamespace(model_dump=lambda: usage))
        def close(self):
            pass
    monkeypatch.setattr(module.socket, 'getaddrinfo', lambda *a: [])
    config = load_config()
    config['paths']['phase7_teacher_output'] = tmp_path
    api = Phase6API(config, 100, phase='phase7_teacher', adapter_factory=Adapter)
    for effort in ('low', 'medium'):
        teacher = Phase6Teacher(api, 'luna_'+effort, effort=effort)
        assert teacher.model == 'gpt-6-luna'
        teacher.generate([{'role': 'user', 'content': 'sample'}], episode_id='a', role='global', turn='1:0')
    assert [c['reasoning']['effort'] for c in calls] == ['low', 'medium']
    assert all(c['max_output_tokens'] == 1024 and c['model'] == 'gpt-6-luna' for c in calls)
    assert api.accounting()['confirmed_usd'] == pytest.approx(.00004)
    with api.db() as db:
        db.execute('UPDATE calls SET confirmed=2.4999')
    with pytest.raises(CallBudgetExceeded):
        api.reserve('luna_medium', 'b', 'b', .001)


def test_full_profile_visible_does_not_imply_latest_target_visible():
    full = '[전체 갱신]\n[글]\nS1 | 문장.\n[Korean document profile: 설명]\nS1 | 문체:한다'
    update = '[부분 갱신]\n[글]\nS1 | 고친 문장.\n[Korean document profile: 설명]\nS1 | 문체:합니다'
    call = {'messages': [{'role': 'user', 'content': full}, {'role': 'user', 'content': 'recent notice'}]}
    raw = [{'role': 'user', 'content': full}, {'role': 'user', 'content': update}]
    v = visibility(call, 'S1', raw)
    assert v['full_profile_in_context'] and not v['immediate_observation_full']
    assert not v['target_profile_current'] and not v['target_text_current']


def test_finished_global_stage_remains_eligible_after_korean_transport_failure():
    row = {'completed': False, 'reward': None, 'global_only_reward': {'R': .85, 'R_over': 0},
           'steps': {'global': 2}, 'termination': {'global': 'STOP'}, 'actions_by_role': {'global': []}}
    selected = absolute_selection(row, {'records': [{'level': 'GLOBAL'}]})
    assert selected['global'] and not selected['korean']


def test_notice_counts_deduplicate_types_and_distinguish_attempts():
    fact = {'sid': 'S1', 'type': 'dependency_change', 'message': 'fact'}
    action = {'action': 'MOVE', 'args': {'target': 'S2'}, 't': 1, 'valid': True,
              'cohesion_changes': [fact, fact]}
    row = {'corpus_episode_id': 'x', 'actions_by_role': {'global': [action,
        {**action, 't': 2, 'cohesion_changes': []}, {**action, 't': 3, 'valid': False, 'cohesion_changes': []}],
        'korean': [{'action': 'EDIT', 'valid': True, 'changed_public_sids': ['S1']}]}}
    result = disturbance([row])
    assert result['accepted']['actions'] == 2 and result['rejected_structural_actions'] == 1
    assert result['accepted']['notice_share'] == .5
    assert result['by_type']['dependency_change']['action_sentence_pairs'] == 1
    assert result['accepted']['later_edit_share_among_notice_actions'] == 1


def test_projection_uses_attempted_denominator_and_actual_corpus_level_sizes():
    from verak.v3.train.teacher_report import projection
    levels = {'L1': {'attempted': 10, 'cost_per_episode': .01, 'kept': {'global': 2, 'korean': 3},
        'global_kept_by_rule': {'no_GLOBAL_STOP_rule': 2}},
        'L3': {'attempted': 5, 'cost_per_episode': .02, 'kept': {'global': 4, 'korean': 1},
        'global_kept_by_rule': {'GLOBAL_R_ge_0.80': 4}}}
    result = projection(levels, {'L1': 100, 'L3': 50})
    assert result == {'cost_usd': 2., 'global': 60., 'korean': 40.,
                      'global_record_threshold': 40., 'global_no_record_stop': 20.}
    levels['L1']['attempted'] = 0
    assert projection(levels, {'L1': 100}) is None
