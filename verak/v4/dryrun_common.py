"""Isolated v4.3 plumbing dry run; no API client or shared runtime mutation."""
from pathlib import Path
import os

from .common import REPO, read_json, write_json, file_sha, sha_text, atomic_new
from .policy_prompts_v43 import verify_frozen, FROZEN
from verak.v3.train.sft_data import REVISION

ROOT = REPO / 'verak/v4/outputs/dryrun'
B2 = REPO / 'verak/v4/outputs/scale3/B2'
ROLES = ('revision', 'korean')
SEED = 871
BASE = REPO / '.hf_cache/kanana-1.5-8b-instruct-2505'
TRAIN_PYTHON = Path('/home/chanwoo/anaconda3/envs/feak_agent/bin/python')
VLLM_PYTHON = Path('/home/chanwoo/anaconda3/envs/verak_vllm/bin/python')
PORT = 8043


def constrain_cpu(*, gpu=False):
    os.environ.update(CUDA_VISIBLE_DEVICES='1' if gpu else '', CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '4' if gpu else '1'
    available = set(os.sched_getaffinity(0))
    chosen = available & set(range(88, 96))
    if not chosen:
        raise RuntimeError('Reserved dry-run CPU cores 88–95 unavailable; do not take collector cores')
    os.sched_setaffinity(0, chosen)
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))


def pinned_base():
    if BASE.resolve().name != REVISION:
        raise ValueError('Dry run requires the accepted c963a5f snapshot')
    verify_frozen()
    return BASE


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(pinned_base()), local_files_only=True)


def local_analysis_environment():
    """Load only the existing local Bareun credential, never instantiate a paid client."""
    from dotenv import dotenv_values
    from .common import load_config
    config = load_config()
    name = config['bareun'].get('api_key_env', 'BAREUN_API_KEY')
    if not os.environ.get(name):
        env_file = Path(os.environ.get('FEAK_ENV_FILE', REPO / '.env')).expanduser()
        value = dotenv_values(env_file).get(name)
        if value:
            os.environ[name] = value
    if not os.environ.get(name):
        # Same credential source as the established teacher_bulk loader.
        value = dotenv_values(config['paths']['scorer_package'] / '.env').get(name)
        if value:
            os.environ[name] = value
    if not os.environ.get(name):
        raise ValueError('Existing local Bareun credential unavailable; no analysis dispatched')


def verify_manifest(*, root=ROOT):
    manifest = read_json(root / 'data/manifest.json')
    if manifest['contract']['base_revision'] != REVISION:
        raise ValueError('Base snapshot changed')
    if manifest['contract']['prompt_manifest_sha256'] != file_sha(FROZEN):
        raise ValueError('Frozen v4.3 contract changed')
    verify_frozen()
    for role in ROLES:
        for part in ('train', 'validation'):
            item = manifest['roles'][role][part]
            if file_sha(Path(item['path'])) != item['sha256']:
                raise ValueError('Dry-run export changed')
    source = root / 'data/source_manifest.json'
    if file_sha(source) != manifest['contract']['source_manifest_sha256']:
        raise ValueError('Frozen source inventory changed')
    return manifest


def require_gpu_approval(*, root=ROOT):
    """The root reviewer approves concrete export/preflight files, not a broad GPU action."""
    approval = read_json(root / 'gpu_approval.json')
    if (approval.get('approved') is not True or approval.get('physical_gpu') != 1
            or approval.get('data_manifest_sha256') != file_sha(root / 'data/manifest.json')
            or approval.get('preflight_sha256') != file_sha(root / 'preflight.json')):
        raise ValueError('A reviewed GPU1-only dry-run approval is required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Expose exactly physical GPU1; logical CUDA0 maps to GPU1')
    preflight = read_json(root / 'preflight.json')
    for name, digest in preflight['code_sha256'].items():
        if file_sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Dry-run implementation changed after preflight')
    if file_sha(Path(preflight['evaluation_manifest']['path'])) != preflight['evaluation_manifest']['sha256']:
        raise ValueError('Approved diagnostic cohort changed after preflight')
    verify_manifest(root=root)
    return approval


def reference(path):
    return {'path': str(path), 'sha256': file_sha(Path(path))}


def checked(item):
    if file_sha(Path(item['path'])) != item['sha256']:
        raise ValueError('Saved artifact changed: ' + item['path'])
    return read_json(Path(item['path']))
