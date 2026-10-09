"""No models in teacher workers; cached Bareun and the shared CPU scoring service."""
import json
from ..common import file_sha, pair_key, read_json, sha_text
from ..corrupt.document import Document, source_document
from ..v2_ops.retry_resources import priority_active
from ..view_data import load_episode_examples
from .config import PHASE


def reference_gpu_fingerprint(config):
    """Frozen Kanana cache identity, without loading or executing the model."""
    from ..score.kanana import SCORER_VERSION, SYSTEM_PROMPT
    paths = config['paths']
    values = {'version': SCORER_VERSION, 'system': SYSTEM_PROMPT,
        'settings': {k: v for k, v in config['scorer'].items() if k not in ('average_k', 'average_seed')},
        'base': str(paths['policy_base']), 'adapter': str(paths['scorer_adapter'])}
    for name, path in [('adapter_weights', paths['scorer_adapter'] / 'adapter_model.safetensors'),
                       ('adapter_config', paths['scorer_adapter'] / 'adapter_config.json'),
                       ('base_config', paths['policy_base'] / 'config.json')]:
        if path.is_file():
            values[name] = file_sha(path)
    return sha_text(json.dumps(values, sort_keys=True, ensure_ascii=False))


def score_gpu_reference(config, question, text):
    """Read root-owned, deferred GPU results; this component starts no GPU work."""
    path = config['paths']['repo'] / 'verak/v3/outputs/data_boost/gpu_rescore/responses' / (pair_key(question, text)+'.json')
    row = read_json(path)
    if row.get('execution_device') != 'gpu_reference' or row.get('error'):
        raise ValueError('Missing valid GPU-reference scoring provenance')
    if row['result']['cache_key'] != pair_key(question, text):
        raise ValueError('GPU-reference score input identity changed')
    return {**row['result'], 'execution_device': 'gpu_reference', 'scorer_fingerprint': row['fingerprint'],
            'gpu_result_path': str(path), 'gpu_slot': row.get('slot'), 'new_gpu_calls_by_component': 0}


def score_cpu(config, question, text):
    # The insertion worker owns one CPU-only model for both background tasks.
    from ..insertion_boost.cpu_score import cached_gpu_score, score_cpu as shared_score
    cached = cached_gpu_score(config, question, text)
    return cached if cached is not None else shared_score(config, question, text,
        requester='global_boost', timeout=86400)


class CPUResources:
    def __init__(self, config):
        from transformers import AutoTokenizer
        from ..insertion_boost.resources import BoostBank, BoostParagraphs
        from ..insertion_boost.cpu_score import shared_root
        self.config = config
        root = config['paths'].get('global_boost_shared_root', config['paths'][PHASE + '_output'])
        self.bank = BoostBank(config, cache_dir=root / 'bareun_units')
        self.analysis = BoostParagraphs(config, cache_dir=root / 'bareun_paragraphs')
        self.analysis.suspended = lambda: (
            (shared_root(config) / 'bareun_pause.json').exists() and priority_active(config))
        self.tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
        self.examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
        self.sources = {}

    def source(self, row):
        source_id = row['source_id']
        if source_id not in self.sources:
            self.sources[source_id] = source_document(self.config, self.examples[source_id], self.bank)
        source = self.sources[source_id].clone()
        if source.text != row['source_text']:
            raise ValueError('Source document does not match the frozen practice')
        self.bank.seed(source)
        return source

    def restore(self, layout):
        result = Document.restore(layout, self.bank)
        self.analysis.refresh(result, {p.pid for p in result.paragraphs})
        return result
