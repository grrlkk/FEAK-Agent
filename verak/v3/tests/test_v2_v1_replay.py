"""The requested full saved-data v1 regression: no GPU or external service."""
import os

import pytest

from verak.v3.common import load_config
from verak.v3.v2_ops.replay_v1 import replay


def test_v1_replays_all_92_saved_luna_pilot_trajectories_with_identical_rewards(tmp_path):
    config = load_config()
    if not (config['paths']['phase7_teacher_output'] / 'design.json').exists():
        pytest.skip('The archived local pilot is not distributed with source code')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        pytest.skip('Run the archived replay with CUDA_VISIBLE_DEVICES= for explicit CPU isolation')
    result = replay(config, output=tmp_path / 'v1_replay.json')
    assert result['passed'] and result['trajectories'] == 92
    assert result['completed_full_reward_matches'] == 91
    assert result['historical_failure_preserved'] == 1
    assert result['paid_calls'] == result['new_bareun_calls'] == result['new_scorer_calls'] == 0
    assert not result['gpu_used']
