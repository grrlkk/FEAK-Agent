"""Lossless surface units with stable IDs; no dependency identities or runtime actions."""

from dataclasses import dataclass, field, replace
import copy
import re
from functools import lru_cache

from verak.src.schemas import Profile, Sentence, Token
from ..common import read_json, sha_text, write_json
from ..phase2 import restore_profile
from ..ko.annotation import KoreanStructure, read_lexicons
from ..ko.structural import annotate_structural

MARKER = re.compile(r"#@[^#\r\n]+#")


@lru_cache(maxsize=1)
def frozen_lexicons():
    return read_lexicons()


def protected(text, start, end):
    return any((start < m.end() and end > m.start()) or
               (start == end and m.start() < start < m.end()) for m in MARKER.finditer(text))


@dataclass
class Unit:
    sid: str
    text: str
    tokens: list
    leading: str = ""


@dataclass
class Paragraph:
    pid: str
    units: list[Unit]


@dataclass
class Document:
    paragraphs: list[Paragraph]
    gaps: list[str]
    tail: str = ""
    analyzer_version: str = "recorded-bareun"

    @classmethod
    def from_profile(cls, text, profile):
        paragraphs, gaps, previous, last_para = [], [], 0, None
        for i, sentence in enumerate(profile.sentences, 1):
            if text[sentence.start:sentence.end] != sentence.text:
                raise ValueError("Source offsets do not match")
            gap = text[previous:sentence.start]
            if gap.strip():
                raise ValueError("Unanalyzed non-whitespace source material")
            if sentence.paragraph != last_para:
                paragraphs.append(Paragraph(f"P{sentence.paragraph}", []))
                gaps.append(gap)
                gap = ""
            paragraphs[-1].units.append(Unit(f"S{i}", sentence.text,
                [replace(t, start=t.start-sentence.start, end=t.end-sentence.start) for t in sentence.tokens], gap))
            previous, last_para = sentence.end, sentence.paragraph
        doc = cls(paragraphs, gaps, text[previous:], profile.analyzer_version)
        if doc.text != text or doc.tail.strip():
            raise ValueError("Document must round-trip exactly")
        return doc

    @property
    def text(self):
        return "".join(gap + "".join(u.leading + u.text for u in p.units)
                       for gap, p in zip(self.gaps, self.paragraphs)) + self.tail

    @property
    def units(self):
        return [u for p in self.paragraphs for u in p.units]

    def locate(self, sid):
        for pi, p in enumerate(self.paragraphs):
            for si, unit in enumerate(p.units):
                if unit.sid == sid:
                    return pi, si, unit
        raise KeyError(sid)

    def clone(self):
        return copy.deepcopy(self)

    def structure(self):
        sentences, ids, offset = [], [], 0
        for gap, paragraph in zip(self.gaps, self.paragraphs):
            offset += len(gap)
            for unit in paragraph.units:
                offset += len(unit.leading)
                sentences.append(Sentence(unit.sid, int(paragraph.pid[1:]), offset,
                    offset+len(unit.text), unit.text,
                    [replace(t, start=t.start+offset, end=t.end+offset) for t in unit.tokens]))
                ids.append(unit.sid)
                offset += len(unit.text)
        return annotate_structural(self.text, Profile(sentences, [], "bareun", self.analyzer_version),
                                   frozen_lexicons(), sentence_ids=ids)

    def snapshot(self):
        return {"gaps": list(self.gaps), "tail": self.tail, "analyzer_version": self.analyzer_version,
                "paragraphs": [{"pid": p.pid, "units": [
                    {"sid": u.sid, "text": u.text, "leading": u.leading} for u in p.units]}
                    for p in self.paragraphs]}

    @classmethod
    def restore(cls, state, bank):
        return cls([Paragraph(p["pid"], [Unit(u["sid"], u["text"], bank.tokens(u["text"]), u["leading"])
                    for u in p["units"]]) for p in state["paragraphs"]],
                   list(state["gaps"]), state["tail"], state["analyzer_version"])


class BareunBank:
    """Seed with immutable source observations; re-analyze every new surface string.

    Stable unit boundaries come from corruption records, not a post-move sentence
    alignment heuristic. Offsets are recomputed before frozen 2c annotation.
    """
    def __init__(self, config, *, analyzer=None):
        self.config, self.analyzer = config, analyzer
        self.cache = config["paths"]["phase3_output"] / "bareun_units"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.memory = {}

    def seed(self, document):
        for unit in document.units:
            self.memory.setdefault(unit.text, unit.tokens)

    def tokens(self, text):
        if text in self.memory:
            return copy.deepcopy(self.memory[text])
        path = self.cache / (sha_text(text) + ".json")
        if path.exists():
            saved = read_json(path)
            if saved["text"] != text:
                raise ValueError("Bareun cache mismatch")
            profile = restore_profile(saved["profile"])
        else:
            if self.analyzer is None:
                self.analyzer = KoreanStructure.from_config(self.config).analyzer
            profile = self.analyzer.profile(text)
            write_json(path, {"text": text, "profile": profile.to_dict()})
            self.analyzer.cache.clear()
        if len(profile.sentences) != 1 or profile.sentences[0].text != text:
            raise ValueError("Changed unit must remain exactly one Bareun sentence")
        sentence = profile.sentences[0]
        tokens = [replace(t, start=t.start-sentence.start, end=t.end-sentence.start) for t in sentence.tokens]
        self.memory[text] = tokens
        return copy.deepcopy(tokens)


def source_document(config, example, bank):
    path = config["paths"]["phase2_output"] / "bareun_profiles" / f"{example.source_line}.json"
    saved = read_json(path)
    if saved["essay_hash"] != example.essay_hash:
        raise ValueError("Source profile hash mismatch")
    doc = Document.from_profile(example.text, restore_profile(saved["profile"]))
    bank.seed(doc)
    return doc


def positional_changes(before, after):
    old = {a.sid: a for a in before.annotations}
    changes = []
    for current in after.annotations:
        source = old.get(current.sid)
        if source is None or source.predecessor_id == current.predecessor_id:
            continue
        for kind, applies in (("CONJ", bool(source.initial_conj and source.initial_conj["eligible"])),
                              ("DEP", source.subject_omitted)):
            if applies:
                changes.append({"kind": kind, "level": "SENTENCE", "sid": current.sid,
                    "previous_before": source.predecessor_id, "previous_after": current.predecessor_id,
                    "cross_paragraph_before": source.cross_paragraph,
                    "cross_paragraph_after": current.cross_paragraph,
                    "interpretation": "positional_hint" if kind == "DEP" else "conjunction_position",
                    "recovery": "predecessor_restored_or_explicit_subject" if kind == "DEP" else
                                "predecessor_restored_or_fitting_conjunction",
                    "relative_weight": .5 if kind == "DEP" else 1.0})
                if kind == "CONJ":
                    changes[-1]["coarse_class_before"] = source.initial_conj["coarse_class"]
    return changes
