"""Build Section 6 annotations on existing Bareun profiles without changing v2."""

from collections import Counter
from pathlib import Path
import re

import yaml

from verak.src.analyzer import Analyzer, BareunBackend
from .patterns import predicate_features
from .types import Annotation, AnteInfo, ConjInfo, ConnInfo, Edge, Structure, SubjInfo

LEXICONS = Path(__file__).with_name("lexicons")
STYLES = {"해라체": "한다", "하십시오체": "합니다", "해요체": "해요", "해체": "해"}
RELATIONS = {"원인": "CAUSE", "순서": "SEQUENCE", "나열": "ADDITION", "대조": "CONTRAST",
             "양보": "CONCESSION", "조건": "CONDITION", "발견": "DISCOVERY",
             "배경": "BACKGROUND", "목적": "PURPOSE", "동시": "SIMULTANEOUS"}


def read_lexicons():
    return {name: yaml.safe_load((LEXICONS / f"{name}.yaml").read_text(encoding="utf-8"))
            for name in ("style", "connective", "conjunction")}


def initial_conjunction(text, lexicon):
    # Longest-first, whole phrase matching; 그래서인지 must not match 그래서.
    text = text.lstrip(' \t\"\'“‘([{')
    for form in sorted(lexicon, key=lambda value: (-len(value), value)):
        pattern = r"\s+".join(re.escape(part) for part in form.split())
        if re.match(pattern + r"(?=$|\s|[,，:;])", text):
            return ConjInfo(form, lexicon[form])
    return None


def subject_candidates(sentence):
    result = []
    for candidate in sentence.subjects:  # Reuse v2's JKS / topic NP boundaries.
        start, end = candidate["span"]
        # The legacy NP walk stops at XPN. Keep an attached lexical prefix:
        # 불/XPN + 평등/NNG must never be normalized to the opposite concept 평등.
        prefixes = [token for token in sentence.tokens if token.tag == "XPN" and token.end == start]
        if prefixes:
            start = prefixes[-1].start
        nouns = [token for token in sentence.tokens if start <= token.start < end and
                 token.tag in {"NNG", "NNP", "NNB", "NP", "NR", "XPN"}]
        if not nouns:
            continue
        surface = sentence.text[start - sentence.start:end - sentence.start]
        result.append(SubjInfo(True, surface,
                              "JKS" if candidate["kind"] == "subject_candidate" else "TOPIC",
                              "".join(token.form for token in nouns), [start, end]))
    # Explicit unmarked quantifier subjects are not ellipsis. This is a narrow
    # syntactic fallback, not a general inference from every sentence-initial noun.
    if not result:
        tokens = sentence.tokens
        if len(tokens) >= 3 and tokens[0].tag in {"NR", "NP"} and tokens[1].form in {"다", "모두"} and tokens[1].tag == "MAG":
            start, end = tokens[0].start, tokens[1].end
            result.append(SubjInfo(True, sentence.text[start - sentence.start:end - sentence.start],
                                  "NONE", tokens[0].form, [start, end]))
    return result


def annotate(text, profile, lexicons=None):
    if profile.analyzer != "bareun":
        # Recorded Bareun profiles are allowed in offline tests, not another analyzer.
        raise ValueError("Korean structure requires Bareun observations")
    lexicons = lexicons or read_lexicons()
    annotations = []
    for index, sentence in enumerate(profile.sentences, 1):
        if text[sentence.start:sentence.end] != sentence.text:
            raise ValueError("Profile and original source text disagree")
        uncertain = []
        endings = [token for token in sentence.tokens if token.tag == "EF"]
        styles = {STYLES[label] for label in lexicons["style"].get(endings[-1].form, [])} if endings else set()
        style = next(iter(styles)) if len(styles) == 1 else "mixed" if styles else "unknown"
        if style in {"unknown", "mixed"}:
            uncertain.append("style")
        connectives = [ConnInfo(f"M{i + 1}", token.form, [token.start, token.end],
                               [RELATIONS[label] for label in lexicons["connective"].get(token.form, [])])
                       for i, token in enumerate(sentence.tokens) if token.tag == "EC"]
        if any(len(conn.candidates) != 1 for conn in connectives):
            uncertain.append("connectives")
        candidates = subject_candidates(sentence)
        # Last JKS candidate is nearest the main predicate. Retain *all* candidates
        # and a flag when this cheap heuristic cannot establish the main subject.
        selected = next((value for value in reversed(candidates) if value.marker == "JKS"),
                        candidates[-1] if candidates else SubjInfo())
        if len(candidates) > 1:
            uncertain.append("subject")
        topics = [value.lemma for value in candidates if value.marker == "TOPIC"]
        if len(set(topics)) > 1:
            uncertain.append("topic")
        polarity, modality, feature_uncertain = predicate_features(sentence.tokens)
        uncertain.extend(feature_uncertain)
        annotations.append(Annotation(
            f"S{index}", f"P{sentence.paragraph}", style, connectives,
            initial_conjunction(sentence.text, lexicons["conjunction"]), selected,
            topics[-1] if topics else None,
            AnteInfo("realized") if selected.realized else AnteInfo(),
            polarity, modality, sorted(set(uncertain)), sentence.text, sentence.start, sentence.end,
            sentence.tokens, candidates))

    edges = []
    narrator = next((ann for ann in annotations if any(
        candidate.lemma in {"나", "저"} for candidate in ann.subject_candidates)), None)
    for index, ann in enumerate(annotations):
        previous = [value for value in annotations[max(0, index - 2):index]
                    if value.paragraph == ann.paragraph]
        if not ann.subject.realized:
            by_lemma = {}
            for earlier in previous:
                for candidate in earlier.subject_candidates:
                    # Repeated mentions of one NP count once; choose its nearest mention.
                    by_lemma[candidate.lemma] = (earlier, candidate)
            ann.antecedent = AnteInfo(
                "resolved" if len(by_lemma) == 1 else "ambiguous" if by_lemma else "none",
                list(dict.fromkeys(value[0].sid for value in by_lemma.values())),
                list(by_lemma))
            if len(by_lemma) != 1:
                ann.uncertain.append("antecedent")
            if not by_lemma and narrator is not None:
                ann.antecedent = AnteInfo("resolved", [narrator.sid], ["화자"])
            if ann.antecedent.status == "resolved":
                target = ann.antecedent.targets[0]
                label = ann.antecedent.candidates[0]
                edges.append(Edge("REF", ann.sid, target, label))
                if by_lemma:
                    earlier = next(iter(by_lemma.values()))[0]
                    if earlier.topic == label:
                        edges.append(Edge("TOPIC", ann.sid, target, label))
        if ann.topic:
            earlier = next((value for value in reversed(annotations[:index])
                            if value.paragraph == ann.paragraph and value.topic == ann.topic), None)
            if earlier:
                edges.append(Edge("TOPIC", ann.sid, earlier.sid, ann.topic))
        if ann.initial_conj:
            if index:
                edges.append(Edge("REL", ann.sid, annotations[index - 1].sid, ann.initial_conj.relation))
            else:
                ann.uncertain.append("relation_target")

    counts = Counter(ann.style for ann in annotations if ann.style not in {"mixed", "unknown"})
    winners = [style for style, count in counts.items() if count == max(counts.values())] if counts else []
    dominant = winners[0] if len(winners) == 1 else "mixed" if winners else "unknown"
    for ann in annotations:
        if dominant not in {"mixed", "unknown"} and ann.style not in {dominant, "unknown", "mixed"}:
            ann.uncertain.append("style_shift")
        ann.uncertain = sorted(set(ann.uncertain))
    return Structure(text, annotations, list(dict.fromkeys(edges)), dominant,
                     profile.analyzer, profile.analyzer_version)


class KoreanStructure:
    def __init__(self, analyzer):
        if analyzer.backend.name != "bareun":
            raise ValueError("Bareun is the only supported analyzer")
        self.analyzer = analyzer
        self.lexicons = read_lexicons()

    @classmethod
    def from_config(cls, config):
        return cls(Analyzer(BareunBackend(**config["bareun"])))

    def analyze(self, text):
        return annotate(text, self.analyzer.profile(text), self.lexicons)
