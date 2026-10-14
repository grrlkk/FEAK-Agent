"""Bareun observations with source offsets; linguistic readings stay candidates."""

from collections import Counter
from importlib.metadata import version
import os
from pathlib import Path
import re

import yaml

from .schemas import Profile, Sentence, Token


class BareunBackend:
    name = "bareun"

    def __init__(self, host="127.0.0.1", port=5656, api_key_env="BAREUN_API_KEY", timeout_s=30):
        from bareunpy import Tagger
        key = os.getenv(api_key_env)
        if not key:
            raise ValueError(f"Missing {api_key_env}; set the key locally")
        self.tagger = Tagger(key, host, port)
        self.timeout_s = timeout_s
        self.version = version("bareunpy")

    def analyze(self, text):
        from bareunpy._lang_service_client import pb, lpb
        from google.protobuf.json_format import MessageToDict
        request = pb.AnalyzeSyntaxRequest()
        request.document.content = text
        request.document.language = "ko_KR"
        request.encoding_type = lpb.EncodingType.UTF32
        request.auto_split_sentence = True
        request.auto_spacing = False
        request.auto_jointing = False
        client = self.tagger.client
        response = client.stub.AnalyzeSyntax(request, metadata=client.metadata, timeout=self.timeout_s)
        return MessageToDict(response, preserving_proto_field_name=True)


def read_lexicons(directory=None):
    directory = Path(directory or Path(__file__).resolve().parents[1] / "lexicons")
    return {name: yaml.safe_load((directory / f"{name}.yaml").read_text())
            for name in ("style", "connective", "focus")}


def _span(data, length):
    start = data.get("begin_offset", data.get("beginOffset", 0))
    end = start + data.get("length", len(data.get("content", "")))
    if not (0 <= start <= end <= length):
        raise ValueError("Analyzer returned an out-of-source offset")
    return start, end


class Analyzer:
    def __init__(self, backend, lexicons=None, context_window=2):
        if context_window != 2:
            raise ValueError("P1 antecedent candidate window is two sentences")
        self.backend = backend
        self.lexicons = lexicons if lexicons is not None else read_lexicons()
        self.context_window = context_window
        self.cache = {}

    def profile(self, text):
        if text in self.cache:
            return self.cache[text]
        sentences = []
        # Every newline starts a new paragraph; do not normalize CRLF/Unicode or whitespace.
        for paragraph, line in enumerate(re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", text), 1):
            content = line.group().rstrip("\r\n")
            if not content.strip():
                continue
            raw = self.backend.analyze(content)
            if not raw.get("sentences"):
                raise ValueError("Bareun returned no sentences for nonblank source")
            previous_end = 0
            for index, row in enumerate(raw["sentences"], 1):
                start, end = _span(row["text"], len(content))
                if start < previous_end or end <= start:
                    raise ValueError("Analyzer sentence spans overlap or are empty")
                if content[previous_end:start].strip():
                    raise ValueError("Analyzer omitted non-whitespace source text")
                # Source sentence text must remain exact even when token forms are restored.
                if row["text"]["content"] != content[start:end]:
                    raise ValueError("Analyzer sentence text does not match source offsets")
                previous_end = end
                tokens = []
                uncertain = []
                for word in row.get("tokens", []):
                    for morph in word.get("morphemes", []):
                        left, right = _span(morph["text"], len(content))
                        if not (start <= left <= right <= end):
                            raise ValueError("Morpheme span is outside its sentence")
                        tokens.append(Token(morph["text"]["content"], morph["tag"],
                                            line.start() + left, line.start() + right))
                        if morph["text"]["content"] != content[left:right]:
                            uncertain.append("restored_morpheme_form_differs_from_surface")
                sentence = Sentence(f"{paragraph}.{index}", paragraph, line.start() + start,
                                    line.start() + end, content[start:end], tokens,
                                    uncertain=sorted(set(uncertain)))
                self._annotate(sentence, sentences)
                sentences.append(sentence)
            if content[previous_end:].strip():
                raise ValueError("Analyzer omitted the end of a paragraph")
        counts = Counter(label for sentence in sentences for label in sentence.style_candidates)
        dominant = sorted(key for key, value in counts.items() if value == max(counts.values())) if counts else []
        result = Profile(sentences, dominant, self.backend.name, self.backend.version,
                         ["style_and_relations_are_candidates_not_constraints"])
        self.cache[text] = result
        return result

    def _annotate(self, sentence, previous):
        endings = [t for t in sentence.tokens if t.tag == "EF"]
        if endings:
            sentence.style_candidates = list(self.lexicons["style"].get(endings[-1].form, []))
            if not sentence.style_candidates:
                sentence.uncertain.append("unmapped_final_ending")
        else:
            sentence.uncertain.append("final_ending_not_detected")
        for i, token in enumerate(sentence.tokens):
            if token.tag == "EC":
                labels = self.lexicons["connective"].get(token.form, [])
                sentence.connectives.append({"form": token.form, "span": [token.start, token.end],
                                             "candidates": labels})
                if len(labels) != 1:
                    sentence.uncertain.append("connective_reading_unresolved")
            if token.tag == "JKS" or (token.tag == "JX" and token.form in {"은", "는"}):
                j = i - 1
                while j >= 0 and sentence.tokens[j].tag in {"NNG", "NNP", "NNB", "NP", "NR", "XSN"}:
                    j -= 1
                nouns = sentence.tokens[j + 1:i]
                if nouns:
                    left, right = nouns[0].start, token.end
                    sentence.subjects.append({"sentence_id": sentence.id, "span": [left, right],
                        "text": sentence.text[left - sentence.start:right - sentence.start],
                        "marker": token.form, "kind": "subject_candidate" if token.tag == "JKS" else "topic_candidate"})
        if not any(token.tag == "JKS" for token in sentence.tokens):
            sentence.uncertain.append("no_subject_marker_does_not_prove_ellipsis")
            sentence.antecedent_candidates = [dict(candidate) for earlier in previous[-self.context_window:]
                                               for candidate in earlier.subjects]
