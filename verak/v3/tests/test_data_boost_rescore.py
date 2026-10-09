from copy import deepcopy
from pathlib import Path

import pytest

from verak.v3.common import file_sha, pair_key, read_json, write_json
from verak.v3.data_boost.rescore import assert_slot, boundary_plan, ready, authorize_after_evaluation
from verak.v3.rft1.selection import extra_global, rebalance, counts


def configuration(tmp_path):
    config = {'paths': {'repo': tmp_path, 'phase8_rft1_output': tmp_path / 'rft'}}
    root = tmp_path / 'verak/v3/outputs/data_boost'
    write_json(root / 'task_contract.json', {'gpu_reference_required': True})
    return config, root


def manifest(root, name='global'):
    path = root / name / 'gpu_rescore_manifest.json'
    write_json(path, {'schema_version': 1, 'component': name, 'reference_gpu_fingerprint': 'frozen',
        'episodes': [], 'requests': [{'key': pair_key('question', 'essay'), 'question': 'question', 'text': 'essay'}]})
    write_json(root / name / 'cpu_ready.json', {'status': 'ready', 'manifest_path': str(path),
        'manifest_sha256': file_sha(path), 'no_live_paid_calls': True,
        'teacher_collection_finished': True, 'cpu_measurements_finished': True})
    return path


def test_readiness_frozen_per_component_and_late_data_waits_for_evaluation(tmp_path):
    config, root = configuration(tmp_path)
    manifest(root)
    plan = boundary_plan(config, 'pre_rft_training')
    assert set(plan['components']) == {'global'} and plan['deferred'] == ['insertion']
    manifest(root, 'insertion')
    assert boundary_plan(config, 'pre_rft_training') == plan
    assert authorize_after_evaluation(config, 'insertion') is None
    write_json(config['paths']['phase8_rft1_output'] / 'a_complete.json', {'completed': True})
    later = authorize_after_evaluation(config, 'insertion')
    assert later['slot'] == 'post_rft_evaluation'


def test_gpu_slot_rejects_active_rollouts_started_training_and_changed_inputs(tmp_path):
    config, root = configuration(tmp_path)
    path = manifest(root)
    boundary_plan(config, 'pre_rft_training')
    rft = config['paths']['phase8_rft1_output']
    write_json(rft / 'rollout_status.json', {'sampling_complete': False, 'saved': 5000, 'errors': []})
    with pytest.raises(RuntimeError, match='rollouts still own'):
        assert_slot(config, 'global', 'pre_rft_training', file_sha(path))
    write_json(rft / 'rollout_status.json', {'sampling_complete': True, 'saved': 5720, 'errors': []})
    assert_slot(config, 'global', 'pre_rft_training', file_sha(path))
    write_json(rft / 'adapters/global/recipe.json', {})
    with pytest.raises(RuntimeError, match='gap.*closed'):
        assert_slot(config, 'global', 'pre_rft_training', file_sha(path))
    write_json(path, {**read_json(path), 'requests': []})
    with pytest.raises(ValueError, match='changed'):
        ready(config, 'global')


def test_global_duplicate_threshold_uses_merged_unique_recoveries_not_weighted_count():
    entries = [{'episode_id': str(i), 'STOP_only': False, 'R': .9, 'origin': 'extra_teacher',
        'operators': {'G_PARA_SWAP': {'all_fully_recovered': True},
                      'G_SENT_MOVE': {'all_fully_recovered': i < 199}}} for i in range(200)]
    selected, _ = rebalance(entries, 'global')
    assert all('G_PARA_SWAP' not in e['duplicated_for'] for e in selected)
    assert sum('G_SENT_MOVE' in e['duplicated_for'] for e in selected) == 199
    assert sum(e['weight'] for e in selected) == 399
    c = counts(selected)
    assert c['operators']['G_PARA_SWAP']['by_source']['extra_teacher']['unique_trajectories'] == 200
    korean = [{**e, 'operators': {'L_CONJ': {'all_fully_recovered': True}}} for e in entries]
    assert all(e['weight'] == 2 for e in rebalance(korean, 'korean')[0])


def extra_fixture(tmp_path):
    config, root = configuration(tmp_path)
    row = {'completed': True, 'termination': {'global': 'STOP'}, 'steps': {'global': 2},
        'score_source': 'gpu_reference', 'reward': None,
        'global_only_reward': {'R': .9, 'R_over': 0., 'per_record': [{'record_id': 'r', 'main': 1.}]},
        'actions_by_role': {'global': [{'action': 'MOVE', 'args': {}, 'valid': True},
                                       {'action': 'STOP', 'valid': True}]}}
    path = root / 'global/gpu_episode.json'
    write_json(path, row)
    selection = root / 'global/gpu_selection.json'
    candidate = {'source_id': 'unused-source', 'split': 'agent_train',
                 'records': [{'record_id': 'r', 'op': 'G_SENT_MOVE', 'level': 'GLOBAL'}]}
    write_json(selection, {'version': 'v1', 'role': 'global', 'score_source': 'gpu_reference',
        'fingerprint': 'frozen', 'manifest_sha256': 'manifest', 'selected': {'new-practice': {
            'path': str(path), 'sha256': file_sha(path), 'attempt': 1, 'candidate': candidate}}})
    write_json(root / 'gpu_rescore/global_complete.json', {'slot': 'pre_rft_training',
        'fingerprint': 'frozen', 'manifest_sha256': 'manifest'})
    merge = root / 'rft1_extra_merge.json'
    write_json(merge, {'include_extra_global': True, 'selection_path': str(selection),
                       'selection_sha256': file_sha(selection)})
    return config, root, selection, merge, row


def test_only_gpu_v1_global_teacher_enters_merge_and_source_holdout_is_enforced(tmp_path):
    config, root, path, merge, row = extra_fixture(tmp_path)
    active = {'old': {'source_id': 'old-source'}}
    selected, provenance = extra_global(config, active, set())
    assert len(selected) == 1 and selected[0]['origin'] == 'extra_teacher'
    assert selected[0]['score_source'] == 'gpu_reference'
    assert provenance['included']
    held, info = extra_global(config, active, {'unused-source'})
    assert held == [] and info['excluded'] == {'held_out_source': 1}
    original = read_json(path)
    for invalid in ({**original, 'version': 'v2'}, {**original, 'score_source': 'cpu'}):
        write_json(path, invalid)
        write_json(merge, {**read_json(merge), 'selection_sha256': file_sha(path)})
        with pytest.raises(ValueError, match='Only GPU-reference v1'):
            extra_global(config, active, set())


def test_extra_teacher_still_requires_rft_stop_and_rejection_gate(tmp_path):
    config, root, path, merge, row = extra_fixture(tmp_path)
    row['termination']['global'] = 'step_limit'
    trajectory = root / 'global/gpu_episode.json'
    write_json(trajectory, row)
    selection = read_json(path)
    selection['selected']['new-practice']['sha256'] = file_sha(trajectory)
    write_json(path, selection)
    write_json(merge, {**read_json(merge), 'selection_sha256': file_sha(path)})
    entries, info = extra_global(config, {}, set())
    assert entries == [] and info['excluded'] == {'role_did_not_STOP': 1}
