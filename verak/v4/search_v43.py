"""Local-only v4.3 SEARCH permissions and sentence-attached source provenance.

The editor environment owns steps, scopes, total INSERTs, and action parsing.
This module owns the narrower search permission and one sourced-INSERT cap.
It never calls a model, changes an essay, or modifies the frozen wiki index.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading


VERSION = 'v4.3_local_search_citations_20261010'
INDEX_ROOT = Path('/home/chanwoo/FEAK-Agent/verak/v4/outputs/scale2/search')
INDEX_SHA256 = '860cda61615d665a5282fecb143147771885042fcca353992d3e6414b528ca78'
SOURCE_KEYS = {'title', 'passage_id'}
PUBLIC_TASK_ID = re.compile(r'(?:D[1-9][0-9]*T[1-9][0-9]*|KT[1-9][0-9]*)\Z')
SEARCH_PROMPT_LINE = 'needs_search=yes 작업만 SEARCH(item_id,query); 검색 근거는 바꾸어 써서 1문장만 INSERT하고 source={title,passage_id}로 인용한다.'
_local = threading.local()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_index(root=INDEX_ROOT, *, verify_hashes=False):
    """Read-only preflight; full artifact hashing is requested once before a run."""
    root = Path(root)
    path = root / 'index_complete.json'
    if _sha(path) != INDEX_SHA256:
        raise ValueError('The frozen kowiki index manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('complete') is not True:
        raise ValueError('A complete local index is required')
    if (root / 'passages.sqlite').stat().st_size != manifest['database_bytes']:
        raise ValueError('The frozen passage database size changed')
    for name, expected in manifest['artifacts'].items():
        file = root / 'bm25' / name
        if file.stat().st_size != expected['bytes']:
            raise ValueError('The frozen BM25 artifact size changed: ' + name)
        if verify_hashes and _sha(file) != expected['sha256']:
            raise ValueError('The frozen BM25 artifact hash changed: ' + name)
    return {'version': VERSION, 'index_manifest_sha256': INDEX_SHA256,
            'passages': manifest['passages'], 'full_index_hashes_verified': verify_hashes,
            'search_top_k': 3, 'sourced_INSERT_cap': 1, 'gpu_used': False,
            'external_search_or_model_calls': 0}


def local_search(query, *, root=INDEX_ROOT):
    """Lazily reuse one read-only Kiwi/BM25 engine per worker thread."""
    key = str(Path(root).resolve())
    engines = getattr(_local, 'engines', None)
    if engines is None:
        engines = _local.engines = {}
    if key not in engines:
        verify_index(root)
        from .wiki_local import LocalWikiSearch
        engines[key] = LocalWikiSearch(root)
    return engines[key].search(query)


def _sentences(live_sentences):
    if isinstance(live_sentences, dict):
        return dict(live_sentences)
    result = {}
    for unit in live_sentences:
        if isinstance(unit, dict):
            sid = unit.get('id', unit.get('sentence_id', unit.get('sid')))
            text = unit['text']
        else:
            sid, text = unit.sid, unit.text
        if not isinstance(sid, str) or not isinstance(text, str) or sid in result:
            raise ValueError('Unique sentence IDs and current text are required')
        result[sid] = text
    return result


def citation_text(source):
    """An attribution display next to a sentence; never inserted into essay text."""
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        raise ValueError('A citation must contain only title and passage_id')
    return f'[출처: {source["title"]} | {source["passage_id"]}]'


class SearchSession:
    def __init__(self, backend=None, *, index_root=INDEX_ROOT):
        self.backend = backend
        self.index_root = Path(index_root)
        self.tasks = {}
        self.delegation = None
        self.history = []
        self.citations = {}
        self.successful_sourced_inserts = 0

    def activate(self, public_tasks, delegation):
        tasks = {}
        for item in public_tasks:
            alias = item['item_id']
            if not isinstance(alias, str) or not PUBLIC_TASK_ID.fullmatch(alias) or alias in tasks:
                raise ValueError('SEARCH requires a unique public task alias, not a private item ID')
            needs = item.get('needs_search', 'no')
            if needs not in {'yes', 'no'}:
                raise ValueError('needs_search must be yes or no')
            if needs == 'yes' and item.get('owner', 'revision') != 'revision':
                raise ValueError('Only Revision tasks may require SEARCH')
            tasks[alias] = deepcopy(item)
        self.tasks, self.delegation = tasks, delegation

    def search(self, action, role='revision'):
        if role != 'revision':
            raise ValueError('KOREAN cannot SEARCH')
        if not isinstance(action, dict) or set(action) != {'action', 'item_id', 'query'} or action['action'] != 'SEARCH':
            raise ValueError('SEARCH requires action, item_id, and query only')
        item = self.tasks.get(action['item_id'])
        if item is None or item.get('needs_search', 'no') != 'yes':
            raise ValueError('SEARCH is allowed only for an active needs_search=yes task')
        query = action['query']
        if not isinstance(query, str) or not query.strip():
            raise ValueError('SEARCH needs a nonempty query')
        backend = self.backend
        passages = (backend.search(query) if hasattr(backend, 'search') else backend(query)) if backend is not None else local_search(query, root=self.index_root)
        if not isinstance(passages, list) or len(passages) > 3:
            raise ValueError('The local SEARCH backend must return at most three passages')
        seen = set()
        for passage in passages:
            fields = ('title', 'section', 'text', 'passage_id', 'url', 'license', 'license_url')
            if not isinstance(passage, dict) or any(not isinstance(passage.get(k), str) or not passage[k] for k in fields):
                raise ValueError('Local SEARCH lost required paragraph or attribution metadata')
            if passage['passage_id'] in seen:
                raise ValueError('Duplicate retrieved passage ID')
            seen.add(passage['passage_id'])
        result = {'delegation': self.delegation, 'item_id': action['item_id'], 'query': query,
                  'status': 'ok' if passages else 'no_results', 'passages': deepcopy(passages)}
        self.history.append(result)
        return deepcopy(result)

    def _retrieved(self, source, role='revision'):
        if role != 'revision':
            raise ValueError('KOREAN cannot make sourced INSERTs')
        if not isinstance(source, dict) or set(source) != SOURCE_KEYS or any(not isinstance(source[k], str) or not source[k] for k in SOURCE_KEYS):
            raise ValueError('source must contain exactly the retrieved title and passage_id; no URL or arbitrary ID')
        for index in range(len(self.history)-1, -1, -1):
            result = self.history[index]
            task = self.tasks.get(result['item_id'])
            if result['delegation'] != self.delegation or task is None or task.get('needs_search', 'no') != 'yes':
                continue
            for passage in result['passages']:
                if all(source[k] == passage[k] for k in SOURCE_KEYS):
                    return deepcopy(passage), {'history_index': index, 'delegation': result['delegation'],
                        'item_id': result['item_id'], 'query': result['query']}
        raise ValueError('The source was not retrieved for a currently delegated needs_search task')

    def validate_source(self, source, role='revision'):
        if source is None:
            return None
        if self.successful_sourced_inserts >= 1:
            raise ValueError('Only one successful search-grounded INSERT is allowed per essay; UNDO does not refund it')
        return self._retrieved(source, role)[0]

    def commit_insert(self, sid, source, text):
        if source is None:
            return
        self.validate_source(source)
        if not isinstance(sid, str) or sid in self.citations or not isinstance(text, str) or not text.strip():
            raise ValueError('A successful INSERT needs a new sentence ID and text')
        passage, retrieval = self._retrieved(source)
        self.citations[sid] = {'inserted_text': text, 'source': deepcopy(source),
                              'passage': passage, 'retrieval': retrieval}
        self.successful_sourced_inserts += 1

    def snapshot(self):
        return deepcopy(self.citations)

    def restore(self, snapshot):
        # UNDO restores sentence metadata; it does not undo searches or count limits.
        self.citations = deepcopy(snapshot)

    def prune(self, live_sids):
        live = set(live_sids)
        self.citations = {sid: data for sid, data in self.citations.items() if sid in live}

    def inherit(self, parent_sid, child_sids):
        if parent_sid in self.citations:
            for sid in child_sids:
                if sid != parent_sid:
                    self.citations[sid] = {**deepcopy(self.citations[parent_sid]), 'derived_from_sid': parent_sid}

    def observation(self, live_sentences):
        live = _sentences(live_sentences)
        citations = [{'sentence_id': sid, 'text': live[sid], **deepcopy(data),
                      'citation_text': citation_text(data['source'])}
                     for sid, data in self.citations.items() if sid in live]
        return {'search_version': VERSION, 'search_history': deepcopy(self.history),
                'successful_sourced_inserts': self.successful_sourced_inserts,
                'sourced_inserts_remaining': max(0, 1-self.successful_sourced_inserts),
                'source_citations': citations}

    def export(self, live_sentences):
        return self.observation(live_sentences)


def verify_citation_proof(attempt):
    """Check saved final citations against the exact recorded retrieval, offline."""
    citations = attempt.get('source_citations', [])
    history = attempt.get('search_history', [])
    if not isinstance(citations, list) or not isinstance(history, list):
        raise ValueError('Citation and retrieval arrays are required')
    live = _sentences(s for p in attempt.get('final_paragraphs', []) for s in p['sentences'])
    seen = set()
    for citation in citations:
        sid, source, passage = citation['sentence_id'], citation['source'], citation['passage']
        if sid in seen or not isinstance(sid, str):
            raise ValueError('Duplicate or invalid cited sentence ID')
        seen.add(sid)
        if sid not in live or live[sid] != citation['text']:
            raise ValueError('A cited sentence must match the final essay after Korean edits')
        if set(source) != SOURCE_KEYS or any(source[k] != passage[k] for k in SOURCE_KEYS):
            raise ValueError('Citation title/ID differs from its stored passage')
        retrieval = citation['retrieval']
        index = retrieval['history_index']
        if type(index) is not int or not 0 <= index < len(history):
            raise ValueError('Citation refers to a missing retrieval')
        result = history[index]
        if any(retrieval[k] != result[k] for k in ('delegation', 'item_id', 'query')):
            raise ValueError('Citation retrieval metadata differs from the action log')
        if passage not in result['passages']:
            raise ValueError('Cited passage is not the exact locally retrieved payload')
    return deepcopy(citations)
