from pathlib import Path

from ..common import file_sha, read_json
from ..train.sft_data import config_for as sft_config, REVISION

PHASE = 'phase8_rft1'
ROLES = ('global', 'korean')
WORKTREE = Path(__file__).resolve().parents[3]


def config_for():
    config = sft_config()
    root = config['paths']['repo'] / 'verak/v3/outputs' / PHASE
    config['paths'][PHASE + '_output'] = root
    config[PHASE] = {'samples': 4, 'temperature': .7, 'top_p': .95, 'seed': 83,
                     'workers': 4, 'role_threshold': .8, 'maximum_rejected_actions': 1,
                     'global_stop_only_cap': .35, 'duplicate_weight': 2,
                     'duplicate_operators': ['G_PARA_SWAP', 'G_SENT_MOVE', 'L_CONJ'],
                     'epochs': 1, 'learning_rate': 5e-5, 'world_size': 2}
    for role in ROLES:
        path = config['paths']['phase7_sft_output'] / 'adapters' / role / 'epoch_2'
        provenance = read_json(path / 'provenance.json')
        if provenance['base_revision'] != REVISION or provenance['adapter_sha256'] != file_sha(path / 'adapter_model.safetensors'):
            raise ValueError('Accepted SFT epoch-2 checkpoint changed')
    config['policy']['adapters'] = {role: f'sft-{role}-ep2' for role in ROLES}
    return config


def load_environment(config):
    from ..v2_ops.local import load_environment as load
    load(config)


def runtime_hashes(config):
    files = [p for directory in ('agent', 'env', 'reward', 'ko')
             for p in (WORKTREE / 'verak/v3' / directory).rglob('*')
             if p.suffix in ('.py', '.txt')]
    result = {}
    for path in files:
        relative = path.relative_to(WORKTREE)
        value = file_sha(path)
        if file_sha(config['paths']['repo'] / relative) != value:
            raise ValueError('v1 runtime differs from accepted SFT: ' + str(relative))
        result[str(relative)] = value
    return result
