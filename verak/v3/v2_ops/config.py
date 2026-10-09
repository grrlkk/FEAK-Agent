"""Opt-in configuration and guards for the independent v2 preparation task."""
from copy import deepcopy
import os
from pathlib import Path

import yaml

from ..common import load_config, read_json

PHASE = 'v2_ops'
OPERATORS = ('G_DEL_LINK', 'L_FUSE')


def config_for(*, version, base_config=None, overlay=None):
    if version != 'v2':
        raise ValueError('New operators require the explicit config v2')
    config = deepcopy(base_config) if base_config is not None else load_config()
    path = Path(overlay) if overlay else Path(__file__).parents[1] / 'config_v2.yaml'
    addition = yaml.safe_load(path.read_text(encoding='utf-8'))
    if addition.get('method_version') != 'v2':
        raise ValueError('Expected a v2 overlay')
    config.update(addition)
    root = config['paths']['repo'] / 'verak/v3/outputs/v2_ops'
    config['paths']['v2_ops_output'] = root
    config['paths']['v2_corpus'] = config['paths']['repo'] / 'verak/v3/data/corrupt_ops_v2'
    config[PHASE]['model'] = read_json(config['paths']['phase4_output'] / 'models.json')['luna_model']
    config['env']['mode'] = 'two_stage'
    config['env']['enable_check'] = False
    config['policy']['context_limit'] = 8192
    config['policy']['generation_reserve'] = 1024
    return config


def require_v2(config):
    if config.get('method_version') != 'v2':
        raise ValueError('INSERT, SPLIT and new operator rewards require config v2')


def require_scorer_slot(config):
    """Physical GPU0 is never visible to v2; GPU1 is queued behind SFT v1."""
    require_v2(config)
    state = read_json(config['paths']['repo'] / 'verak/v3/outputs/phase7_sft/continuation_status.json')
    if state['current']['stage'] != 'completed_awaiting_final_audit':
        raise RuntimeError('GPU1 scoring is queued behind the complete SFT v1 evaluation')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('The v2 scorer process requires CUDA_VISIBLE_DEVICES=1 (physical GPU1 only)')
    return 0  # Physical GPU1 is visible as local device 0 in this isolated process.
