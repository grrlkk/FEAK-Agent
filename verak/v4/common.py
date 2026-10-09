"""CPU-only preparation configuration and immutable local artifacts."""
from pathlib import Path
import os

from verak.v3.common import read_json, write_json, sha_text, file_sha, load_config as v3_config
from verak.v3.train.teacher_bulk import atomic_new, collection_lock

REPO = Path('/home/chanwoo/FEAK-Agent')
ROOT = REPO / 'verak/v4/outputs/prep'
GENRES = ('argumentative', 'explanatory', 'emotional')
GENRE_KO = dict(zip(GENRES, ('논증', '설명', '정서')))


def safe_id(value):
    return value.replace(':', '_')


def load_config():
    return v3_config()


def constrain_cpu():
    # Do this before any library that might initialize CUDA or a thread pool.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    allowed = set(os.sched_getaffinity(0))
    chosen = allowed & set(range(112, 120))
    os.sched_setaffinity(0, chosen or sorted(allowed)[-2:])
    if os.nice(0) < 19:
        os.nice(19-os.nice(0))


def rows_for(group):
    design = read_json(ROOT / 'design.json')
    for source_id in design[group]:
        path = ROOT / 'essays' / (safe_id(source_id)+'.json')
        row = read_json(path)
        if row['source_id'] != source_id or file_sha(path) != design['essay_files'][source_id]['sha256']:
            raise ValueError('Frozen v4 source artifact changed')
        yield row


def public_essay(row):
    return {key: row[key] for key in ('source_id', 'genre', 'question', 'paragraphs')}


def schema_object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def schema_array(items):
    return {'type': 'array', 'items': items}


def schema_enum(values):
    return {'type': 'string', 'enum': list(values)}


STRING = {'type': 'string'}
