"""Active Phase 2c structure. Antecedent identity exists only in optional debug data.

The old annotation API remains a frozen Phase 2/2b reproduction interface. New
datasets and downstream consumers use this schema and its DEP/coarse contracts.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
import re

from .annotation import KoreanStructure, annotate, read_lexicons
from .coarse import coarse_connectives, coarse_conjunction
from .final_style import final_style, quote_spans
from .patterns import predicate_features
from .word_text import focus_particles

NOMINAL = {"NNG", "NNP", "NNB", "NP", "NR", "SL", "SH", "SN", "XSN"}
MARKER = re.compile(r"#@[^#\n]+#")
FIELD_LEVELS = {
    **dict.fromkeys(("sid", "paragraph", "text", "start", "end", "tokens", "subject_omitted",
                    "subject_evidence", "omission_uncertain", "predecessor_id", "cross_paragraph",
                    "initial_conj", "multi_unit"), "SENTENCE"),
    **dict.fromkeys(("connectives", "polarity", "modality", "focus_particles"), "WORD"),
    **dict.fromkeys(("style", "final_ending", "final_endings", "off_style"), "TEXT"),
}
CHANGE_LEVELS = {"dependency_change": "SENTENCE", "subject_omission_change": "SENTENCE",
                 "conjunction_change": "SENTENCE", "ec_relation_change": "WORD",
                 "polarity_change": "WORD", "modality_change": "WORD", "focus_change": "WORD",
                 "style_change": "TEXT"}


def marked_np_display(sentence, marker_index):
    """Lossless surface display only: never decide omission or NP eligibility here."""
    tokens, i = sentence.tokens, marker_index - 1
    while i and tokens[i - 1].tag in NOMINAL | {"XPN"} and tokens[i - 1].end == tokens[i].start:
        i -= 1
    start, end = tokens[i].start, tokens[marker_index].end
    entity = None
    for match in reversed(list(MARKER.finditer(sentence.text))):
        a, b = sentence.start + match.start(), sentence.start + match.end()
        if a >= end:
            continue
        if b <= start:
            between = [t for t in tokens if b <= t.start < start]
            if all(t.tag in NOMINAL | {"XPN", "JKG", "MMD", "MMN"} for t in between):
                start = a
        if start <= a and b <= end:
            entity = f"{match.group()}@{a}"
        break
    return {"surface": sentence.text[start - sentence.start:end - sentence.start],
            "span": [start, end], **({"entity_id": entity} if entity else {})}


def subject_observation(sentence):
    """Conservative surface evidence; deliberately no ETM/relative-clause filter.

    Any marked NP can make the main-clause reading uncertain. Abstain with false,
    rather than deleting that NP and inventing a subject omission.
    """
    tokens, evidence = sentence.tokens, []
    for i, token in enumerate(tokens):
        if not (token.tag == "JKS" and token.form in {"이", "가"} or
                token.tag == "JX" and token.form in {"은", "는", "도"}):
            continue
        if i and tokens[i - 1].tag in NOMINAL:
            evidence.append({"token_id": f"M{i + 1}", **marked_np_display(sentence, i),
                             "marker": token.form, "tag": token.tag, "level": "SENTENCE"})
    # Bareun may split #@...# into several SW tokens. Preserve each occurrence.
    for match in MARKER.finditer(sentence.text):
        suffix = sentence.text[match.end():]
        particle = re.match(r"\s*(은|는|이|가|도)(?=$|\s|[,.!?])", suffix)
        if particle:
            evidence.append({"token_id": None, "surface": match.group() + particle[1],
                             "marker": particle[1], "tag": "MARKER_SUBJECT",
                             "entity_id": f"{match.group()}@{sentence.start + match.start()}", "level": "SENTENCE"})
    if evidence:
        return False, evidence, False
    # Unmarked pronouns/quantified NPs and fragments are uncertain, not omissions.
    uncertain = not any(t.tag == "EF" for t in tokens)
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if token.tag in {"NP", "NR"} and (following is None or not following.tag.startswith("J")):
            uncertain = True
        if token.tag in NOMINAL and following and following.form in {"모두", "다"}:
            uncertain = True
    if len([t for t in tokens if t.tag == "EF"]) >= 2:
        # Several units can have different subject realization. Do not assert
        # that this entire Bareun sentence lacks a subject.
        uncertain = True
    return not uncertain, evidence, uncertain


@dataclass
class StructuralAnnotation:
    sid: str
    paragraph: str
    text: str
    start: int
    end: int
    tokens: list
    subject_omitted: bool
    subject_evidence: list
    omission_uncertain: bool
    predecessor_id: str | None
    cross_paragraph: bool
    connectives: list
    initial_conj: dict | None
    polarity: str
    modality: str | None
    focus_particles: list
    style: str
    final_ending: dict | None
    final_endings: list
    multi_unit: bool
    off_style: bool = False
    uncertain: list = field(default_factory=list)


@dataclass
class StructuralDocument:
    text: str
    annotations: list[StructuralAnnotation]
    edges: list[dict]
    dominant_style: str
    analyzer: str
    analyzer_version: str
    debug: dict | None = None
    schema_version: str = "phase2c_structural"

    def to_dict(self, *, include_debug=False):
        result = asdict(self)
        result.pop("debug")
        for ann in result["annotations"]:
            ann["field_levels"] = {**FIELD_LEVELS, "uncertain": {
                value: FIELD_LEVELS.get(value, "WORD") for value in ann["uncertain"]}}
        result["field_levels"] = {"dominant_style": "TEXT", "edges": "SENTENCE"}
        result["cohesion_change_levels"] = dict(CHANGE_LEVELS)
        if include_debug and self.debug is not None:
            result["debug"] = self.debug
        return result


def annotate_structural(text, profile, lexicons=None, *, sentence_ids=None, include_debug=False):
    if profile.analyzer != "bareun":
        raise ValueError("Bareun is the only supported analyzer")
    lexicons = lexicons or read_lexicons()
    ids = sentence_ids if sentence_ids is not None else [f"S{i + 1}" for i in range(len(profile.sentences))]
    if len(ids) != len(profile.sentences) or len(set(ids)) != len(ids):
        raise ValueError("Provide one unique stable ID per sentence")
    annotations, edges = [], []
    spans = quote_spans(text)
    for sid, sentence in zip(ids, profile.sentences):
        if text[sentence.start:sentence.end] != sentence.text:
            raise ValueError("Profile and source text disagree")
        previous = annotations[-1] if annotations else None
        style, final, endings = final_style(sentence, lexicons["style"], previous.style if previous else None, spans)
        omitted, evidence, omission_uncertain = subject_observation(sentence)
        polarity, modality, uncertain = predicate_features(sentence.tokens)
        ann = StructuralAnnotation(sid, f"P{sentence.paragraph}", sentence.text, sentence.start, sentence.end,
            sentence.tokens, omitted, evidence, omission_uncertain, previous.sid if previous else None,
            previous is not None and previous.paragraph != f"P{sentence.paragraph}",
            coarse_connectives(sentence, lexicons["connective"]),
            coarse_conjunction(sentence.text, lexicons["conjunction"]), polarity, modality,
            focus_particles(sentence.tokens), style, final, endings, len(endings) >= 2,
            uncertain=uncertain)
        if omission_uncertain:
            ann.uncertain.append("subject_omitted")
        if style == "unknown":
            ann.uncertain.append("style")
        if omitted:
            edges.append({"kind": "DEP", "src": sid, "dst": ann.predecessor_id,
                          "cross_paragraph": ann.cross_paragraph, "level": "SENTENCE"})
        if ann.initial_conj and ann.initial_conj["eligible"]:
            edges.append({"kind": "REL", "src": sid, "dst": ann.predecessor_id,
                          "coarse_class": ann.initial_conj["coarse_class"],
                          "cross_paragraph": ann.cross_paragraph, "level": "SENTENCE"})
        annotations.append(ann)
    counts = Counter(a.style for a in annotations if a.style != "unknown")
    winners = [k for k, v in counts.items() if v == max(counts.values())] if counts else []
    dominant = winners[0] if len(winners) == 1 else "unknown"
    for ann in annotations:
        ann.off_style = dominant != "unknown" and ann.style not in {"unknown", dominant}
    debug = {"phase2b_antecedents_debug_only": annotate(text, profile, lexicons).to_dict()} if include_debug else None
    return StructuralDocument(text, annotations, edges, dominant, profile.analyzer, profile.analyzer_version, debug)


class StructuralAnalyzer(KoreanStructure):
    def analyze(self, text, *, sentence_ids=None, include_debug=False):
        return annotate_structural(text, self.analyzer.profile(text), self.lexicons,
                                   sentence_ids=sentence_ids, include_debug=include_debug)


def dependency_changes(source, current):
    """Facts over stable IDs. No antecedent identity or claim of broken cohesion.

    Phase 4 will implement recovery (predecessor restored OR explicit subject).
    This function only describes currently observable structural changes.
    """
    if not isinstance(source, StructuralDocument) or not isinstance(current, StructuralDocument):
        raise TypeError("Use Phase 2c structures, never legacy REF/TOPIC identities")
    originals = {a.sid: a for a in source.annotations}
    result = []
    for ann in current.annotations:
        old = originals.get(ann.sid)
        if old and old.subject_omitted and ann.subject_omitted and old.predecessor_id != ann.predecessor_id:
            result.append({"type": "dependency_change", "level": "SENTENCE", "sid": ann.sid,
                           "before": old.predecessor_id, "after": ann.predecessor_id,
                           "message": f"{ann.sid} (subject omitted): preceding sentence changed "
                                      f"{old.predecessor_id or 'START'} → {ann.predecessor_id or 'START'}"})
    return result
