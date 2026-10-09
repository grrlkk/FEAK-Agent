from pathlib import Path

from ..common import load_config, read_json

PHASE = 'data_boost_global'
OPERATORS = ('G_PARA_SWAP', 'G_SENT_MOVE')


def config_for():
    config = load_config()
    config[PHASE] = {
        'model': read_json(config['paths']['phase4_output'] / 'models.json')['luna_model'],
        'sol_model': 'gpt-6.1-sol', 'max_cost_usd': 12., 'max_concurrent_requests': 2,
        'phase_api_ceiling': 20000, 'seed': 97, 'ordering_seeds': [71, 72],
        'per_operator': 400, 'attempts': 2,
    }
    config['paths'][PHASE + '_output'] = config['paths']['repo'] / 'verak/v3/outputs/data_boost/global'
    if config['env']['mode'] != 'two_stage' or config['env']['enable_check']:
        raise ValueError('Keep v1 two-stage/no-CHECK')
    if (config['policy']['context_limit'], config['policy']['generation_reserve']) != (8192, 1024):
        raise ValueError('Keep v1 8192/1024 context')
    return config


def constrain_cpu():
    """Called before importing torch: no visible CUDA, small disjoint CPU allocation."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    allowed = sorted(os.sched_getaffinity(0))
    chosen = sorted(set(allowed) & set(range(104, 112)))
    os.sched_setaffinity(0, chosen or allowed[-4:])
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))
