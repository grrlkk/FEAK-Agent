"""Bareun paragraph cache; frozen Phase 2c annotation remains unchanged."""
from dataclasses import replace
import copy
import json

from ..common import read_json, sha_text, write_json
from ..phase2 import restore_profile
from ..ko.annotation import KoreanStructure


class ParagraphAnalyzer:
    def __init__(self, config, *, analyzer=None, cache_dir=None):
        self.config = config
        self.analyzer = analyzer
        self.cache_dir = cache_dir or config['paths']['phase5_output'] / 'bareun_paragraphs'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.hits = 0

    def profile(self, text, neighbors=()):
        key = sha_text(json.dumps([text, sha_text(json.dumps(neighbors, ensure_ascii=False))], ensure_ascii=False))
        path = self.cache_dir / (key + '.json')
        if path.exists():
            record = read_json(path)
            if record['text'] != text:
                raise ValueError('Paragraph cache mismatch')
            self.hits += 1
            return restore_profile(record['profile'])
        if self.analyzer is None:
            self.analyzer = KoreanStructure.from_config(self.config).analyzer
        profile = self.analyzer.profile(text)
        self.calls.append({'text_hash': sha_text(text), 'context_hash': sha_text(str(neighbors))})
        write_json(path, {'text': text, 'neighbors': list(neighbors), 'profile': profile.to_dict()})
        self.analyzer.cache.clear()
        return profile

    def pieces(self, text):
        if '\n' in text or '\r' in text:
            raise ValueError('EDIT new_text must stay within one paragraph')
        profile = self.profile(text)
        pieces, end = [], 0
        for sentence in profile.sentences:
            if text[end:sentence.start].strip():
                raise ValueError('Unanalyzed edit text')
            pieces.append((sentence.text,
                [replace(t, start=t.start-sentence.start, end=t.end-sentence.start) for t in sentence.tokens],
                text[end:sentence.start]))
            end = sentence.end
        if text[end:].strip() or not pieces:
            raise ValueError('No complete analyzed unit in edit')
        return pieces

    def refresh(self, document, paragraph_ids):
        for index, paragraph in enumerate(document.paragraphs):
            if paragraph.pid not in paragraph_ids or not paragraph.units:
                continue
            text = ''.join(u.leading + u.text for u in paragraph.units)
            before = [u.text for p in document.paragraphs[:index] for u in p.units][-2:]
            after = [u.text for p in document.paragraphs[index+1:] for u in p.units][:2]
            profile = self.profile(text, before + after)
            tokens = [t for sentence in profile.sentences for t in sentence.tokens]
            offset = 0
            for unit in paragraph.units:
                offset += len(unit.leading)
                end = offset + len(unit.text)
                if any(t.start < offset < t.end or t.start < end < t.end for t in tokens):
                    raise ValueError('Bareun token crosses a stable sentence boundary')
                unit.tokens = [replace(t, start=t.start-offset, end=t.end-offset)
                               for t in tokens if offset <= t.start and t.end <= end]
                offset = end
            document.analyzer_version = profile.analyzer_version


def alias_structure(structure, sentence_ids, paragraph_ids):
    result = copy.deepcopy(structure)
    for ann in result.annotations:
        ann.sid = sentence_ids[ann.sid]
        ann.paragraph = paragraph_ids[ann.paragraph]
        if ann.predecessor_id is not None:
            ann.predecessor_id = sentence_ids[ann.predecessor_id]
    for edge in result.edges:
        edge['src'] = sentence_ids[edge['src']]
        if edge['dst'] is not None:
            edge['dst'] = sentence_ids[edge['dst']]
    return result
