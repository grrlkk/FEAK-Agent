"""Build Section 6 annotations on existing Bareun profiles without changing v2."""

from collections import Counter
from pathlib import Path
import re

import yaml

from verak.src.analyzer import Analyzer, BareunBackend
from .patterns import predicate_features
from .types import Annotation, AnteInfo, ConjInfo, Edge, Structure, SubjInfo
from .subjects import subject_candidates
from .word_text import connectives as read_connectives, ending_styles, focus_particles

LEXICONS = Path(__file__).with_name("lexicons")


def read_lexicons():
    return {name: yaml.safe_load((LEXICONS / f"{name}.yaml").read_text(encoding="utf-8"))
            for name in ("style", "connective", "conjunction")}


def initial_conjunction(text, lexicon):
    # Longest-first, whole phrase matching; 그래서인지 must not match 그래서.
    text = text.lstrip(' \t\"\'“‘([{')
    for form in sorted(lexicon, key=lambda value: (-len(value), value)):
        pattern = r"\s*".join(re.escape(part) for part in form.split())
        if re.match(pattern + r"(?=$|\s|[,，:;])", text):
            return ConjInfo(form, lexicon[form])
    return None


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
        style, final_endings = ending_styles(sentence.tokens, lexicons["style"])
        if style in {"unknown", "mixed"}:
            uncertain.append("style")
        connectives = read_connectives(sentence.tokens, lexicons["connective"])
        if any(conn.classification == "AMBIGUOUS" for conn in connectives):
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
            sentence.tokens, candidates, len(final_endings) >= 2, final_endings,
            focus_particles(sentence.tokens)))

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
                earlier = next(value for value in annotations if value.sid == target)
                confidence = "HIGH" if (len(by_lemma) == 1 and not ann.multi_unit and
                                         not earlier.multi_unit and len(earlier.subject_candidates) == 1) else "LOW"
                edges.append(Edge("REF", ann.sid, target, label, confidence))
                if by_lemma:
                    earlier = next(iter(by_lemma.values()))[0]
                    if earlier.topic == label:
                        edges.append(Edge("TOPIC", ann.sid, target, label, confidence))
        if ann.topic:
            earlier = next((value for value in reversed(annotations[:index])
                            if value.paragraph == ann.paragraph and value.topic == ann.topic), None)
            if earlier:
                distinct = {candidate.lemma for value in previous for candidate in value.subject_candidates}
                confidence = "HIGH" if (earlier in previous and len(distinct) == 1 and
                    len(ann.subject_candidates) == 1 and not ann.multi_unit and not earlier.multi_unit) else "LOW"
                edges.append(Edge("TOPIC", ann.sid, earlier.sid, ann.topic, confidence))
        if ann.initial_conj:
            if index:
                edges.append(Edge("REL", ann.sid, annotations[index - 1].sid, ann.initial_conj.relation, "HIGH"))
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
