"""One GPU1 service thread; thread-local Bareun/tokenizer for four API workers."""
from concurrent.futures import ThreadPoolExecutor
from threading import local

from ..corrupt.document import BareunBank, Document, source_document
from ..env.analysis import ParagraphAnalyzer


class Resources:
    def __init__(self, config, examples, *, output_key='phase6_output'):
        # Transformers' lazy module loader is not safe on simultaneous first imports.
        from transformers import AutoTokenizer
        self.tokenizer_class = AutoTokenizer
        self.config, self.examples = config, examples
        self.output_key = output_key
        self.local = local()
        self.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix='scorer-gpu1')
        self.scorer = self.embedding = None

    def worker(self):
        if not hasattr(self.local, 'bank'):
            paths = self.config['paths']
            self.local.bank = BareunBank(self.config, cache_dir=paths[self.output_key]/'bareun_units',
                read_cache_dirs=tuple(paths[p]/'bareun_units' for p in ('phase6_output', 'phase5_output', 'phase3b_output', 'phase3_output') if p != self.output_key))
            self.local.analysis = ParagraphAnalyzer(self.config, cache_dir=paths[self.output_key]/'bareun_paragraphs')
            self.local.tokenizer = self.tokenizer_class.from_pretrained(str(paths['policy_base']), local_files_only=True)
            self.local.sources = {}
        return self.local

    def source(self, sid):
        state = self.worker()
        if sid not in state.sources:
            state.sources[sid] = source_document(self.config, self.examples[sid], state.bank)
        return state.sources[sid].clone()

    def corrupted(self, row):
        state = self.worker()
        self.source(row['source_id'])
        for r in row['records']:
            if r['op'] == 'G_OFFTOPIC':
                self.source(r['params']['donor']['source_id'])
        return Document.restore(row['corrupted_layout'], state.bank)

    def _score(self, question, text):
        if self.scorer is None:
            from ..score.kanana import KananaScorer
            self.scorer = KananaScorer(self.config)
        return self.scorer.score(question, text).to_dict()

    def score(self, question, text):
        return self.gpu.submit(self._score, question, text).result()

    def _similarity(self, left, right):
        if self.embedding is None:
            from ..reward.similarity import SentenceSimilarity
            self.embedding = SentenceSimilarity(self.config['similarity']['model'], device='cuda:1')
        return self.embedding(left, right)

    def similarity(self, left, right):
        return self.gpu.submit(self._similarity, left, right).result()

    def close(self):
        if self.scorer is not None:
            self.gpu.submit(self.scorer.close).result()
        self.gpu.shutdown()
