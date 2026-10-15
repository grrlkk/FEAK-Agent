"""Explicit new experiment: prior v1/v2 settings and artifacts stay immutable."""
from ..v2_ops.config import PHASE
from ..v2_ops.retry import config_for_retry


def config_for():
    config = config_for_retry()
    shared = config['paths']['repo'] / 'verak/v3/outputs/data_boost'
    config['paths'][PHASE + '_output'] = shared / 'insertion'
    config['paths']['data_boost_shared'] = shared
    config['paths']['retry_output'] = config['paths']['repo'] / 'verak/v3/outputs/v2_ops_retry'
    config[PHASE]['max_cost_usd'] = 6.0
    config[PHASE]['max_concurrent_requests'] = 2
    config[PHASE]['gpu_policy'] = 'cpu_api_only_no_gpu'
    config['insertion_boost'] = {'enabled': True, 'qc_denominator': 'judged_records',
        'minimum_qc_pass_rate': .30, 'selected_roles': ['global'],
        'prior_train_passes': 78, 'remaining_feasible_train_sources': 200,
        'cpu_threads': 16, 'training': False, 'gpu_used': False}
    return config
