"""Locate, validate and apply five edits without a whole-essay rewrite API."""
from collections import Counter
from difflib import SequenceMatcher
import re

from .models import Operation


def compact(text):
    return "".join(char for char in text if not char.isspace())


class Rejected(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def quote_spans(text, quote):
    if not quote or not compact(quote):
        return []
    indices = [i for i, char in enumerate(text) if not char.isspace()]
    haystack, needle = compact(text), compact(quote)
    result, start = [], 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return result
        result.append((indices[index], indices[index + len(needle) - 1] + 1))
        start = index + 1


def unique_span(text, quote):
    matches = quote_spans(text, quote)
    if len(matches) != 1:
        raise Rejected("quote_not_unique", f"인용은 현재 글에서 한 번 나와야 합니다. 발견 횟수: {len(matches)} / 인용: {quote!r}")
    return matches[0]


def paragraph_spans(text):
    spans, start = [], 0
    for match in [*re.finditer(r"\n[ \t]*\n", text), None]:
        end = match.start() if match else len(text)
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append((a, b))
        start = match.end() if match else len(text)
    return spans


def changed_char_ratio(before, after):
    changes = sum(max(b - a, d - c) for kind, a, b, c, d in
                  SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if kind != "equal")
    return min(1.0, changes / max(len(before), len(after), 1))


class Executor:
    def __init__(self, *, sentence_spans, pos, marker_pattern, placeholder_pattern):
        self.sentence_spans = sentence_spans
        self.pos = pos
        self.marker_pattern, self.placeholder_pattern = marker_pattern, placeholder_pattern

    def _sentence(self, text, quote):
        a, b = unique_span(text, quote)
        for index, sentence in enumerate(self.sentence_spans(text), 1):
            if (sentence["start"], sentence["end"]) == (a, b):
                return a, b, index
        raise Rejected("not_whole_sentence", "move·merge·split·connect에는 문장 전체를 인용해야 합니다.")

    def _protect(self, text, touched, keep, insertions=()):
        for quote in keep:
            matches = quote_spans(text, quote)
            if not matches:
                raise Rejected("preserve_quote_missing", f"지킬 문장을 현재 글에서 찾지 못했습니다: {quote}")
            for a, b in matches:
                if any(start < b and a < end for start, end in touched) or any(a < point < b for point in insertions):
                    raise Rejected("preserve_violation", f"지킬 목록의 문장을 건드리는 연산입니다: {quote}")

    def _markers(self, before, after):
        if Counter(self.marker_pattern.findall(before)) != Counter(self.marker_pattern.findall(after)):
            raise Rejected("anonymization_violation", "비식별 표시의 종류와 개수를 그대로 보존해야 합니다.")
        if self.placeholder_pattern.search(after) or "__ANON_" in after:
            raise Rejected("anonymization_violation", "복원되지 않은 비식별 임시 표시가 남았습니다.")

    def _content(self, text):
        try:
            return {form for form, tag in self.pos(text)
                    if tag.startswith(("NN", "NP", "NR", "VV", "VA", "MAG", "MAJ", "XR", "SL", "SH", "SN"))}
        except Exception as exc:
            raise Rejected("merge_analysis_unavailable", f"Bareun 내용어 분석 실패: {exc}") from exc

    def apply(self, text, operation, *, keep=(), allowed_issue_ids=None):
        op = operation if isinstance(operation, Operation) else Operation.model_validate(operation)
        if not op.issue_ids or any(not identifier.strip() for identifier in op.issue_ids):
            raise Rejected("missing_feedback_issue", "이번 턴 피드백의 이슈 ID를 하나 이상 참조해야 합니다.")
        if allowed_issue_ids is not None and not set(op.issue_ids) <= set(allowed_issue_ids):
            raise Rejected("outside_feedback", "이번 턴 피드백에 없는 이슈의 연산입니다.")
        patches, hints, touched, points = [], [], [], []
        label, source = "", ""
        if op.op == "replace":
            a, b = unique_span(text, op.quote)
            if not op.replacement or not op.replacement.strip():
                raise Rejected("sentence_deletion", "빈 텍스트로 바꾸어 학생 문장을 지울 수 없습니다.")
            intersects = [s for s in self.sentence_spans(text) if s["start"] < b and a < s["end"]]
            if len(intersects) != 1 or (a == 0 and b == len(text) and len(intersects) > 1):
                raise Rejected("replace_scope", "replace는 한 문장 안의 구간으로 한정합니다. 여러 문장은 다른 연산으로 나누세요.")
            sentence = intersects[0]
            if a == sentence["start"] and b == sentence["end"] and not any(c.isalnum() for c in op.replacement):
                raise Rejected("sentence_deletion", "문장 전체를 문장부호만 남도록 바꿀 수 없습니다.")
            if "\n\n" in op.replacement:
                raise Rejected("replace_scope", "replace로 새 문단을 만들 수 없습니다.")
            patches = [(a, b, op.replacement)]
            hints = [op.replacement]
            touched = [(a, b)]
            source = text[a:b]
            label = f"표현 ‘{source}’을 ‘{op.replacement}’으로 바꿈"
        elif op.op == "move":
            a, b, number = self._sentence(text, op.quote)
            if op.position is None:
                raise Rejected("missing_position", "옮길 위치가 필요합니다.")
            position = op.position
            if position.kind in {"before", "after"}:
                c, d, target_number = self._sentence(text, position.anchor_quote)
                point = c if position.kind == "before" else d
                target_label = f"{target_number}번 문장 {'앞' if position.kind == 'before' else '뒤'}"
            else:
                paragraphs = paragraph_spans(text)
                if position.paragraph is None or not 1 <= position.paragraph <= len(paragraphs):
                    raise Rejected("invalid_paragraph", "현재 글에 존재하는 문단 번호를 지정하세요.")
                c, d = paragraphs[position.paragraph - 1]
                point = c if position.kind == "paragraph_start" else d
                target_label = f"{position.paragraph}문단 {'처음' if position.kind == 'paragraph_start' else '끝'}"
            if a <= point <= b:
                raise Rejected("self_move", "문장을 자기 위치 안으로 옮길 수 없습니다.")
            source = text[a:b]
            # Touch only sentence content; retain original paragraph boundaries.
            insertion = source + " " if point == c else " " + source
            patches = [(a, b, ""), (point, point, insertion)]
            touched, points, hints = [(a, b)], [point], [source]
            label = f"{number}번 문장을 {target_label}으로 옮김"
        elif op.op == "merge":
            a, b, n1 = self._sentence(text, op.sentence_a)
            c, d, n2 = self._sentence(text, op.sentence_b)
            if a == c:
                raise Rejected("same_sentence", "서로 다른 두 문장을 합쳐야 합니다.")
            if not op.replacement or len(self.sentence_spans(op.replacement)) != 1:
                raise Rejected("merge_shape", "merge 결과는 비어 있지 않은 한 문장이어야 합니다.")
            left, right = self._content(text[a:b]), self._content(text[c:d])
            if not left or not right:
                raise Rejected("merge_analysis_unavailable", "Bareun 내용어 분석을 확인할 수 없어 합치지 않았습니다.")
            missing = (left | right) - self._content(op.replacement)
            if missing:
                raise Rejected("merge_content_loss", "합친 문장에서 빠진 내용어: " + ", ".join(sorted(missing)))
            patches = [(a, b, op.replacement), (c, d, "")]
            touched, hints = [(a, b), (c, d)], [op.replacement]
            source = text[a:b] + " / " + text[c:d]
            label = f"{n1}번·{n2}번 문장의 내용어를 유지하며 한 문장으로 합침"
        elif op.op == "split":
            a, b, number = self._sentence(text, op.quote)
            if not op.sentences or len(op.sentences) < 2 or any(not s.strip() or len(self.sentence_spans(s)) != 1 for s in op.sentences):
                raise Rejected("split_shape", "split에는 두 개 이상의 완전한 문장이 필요합니다.")
            patches = [(a, b, " ".join(op.sentences))]
            touched, hints = [(a, b)], list(op.sentences)
            source = text[a:b]
            label = f"{number}번 문장을 {len(op.sentences)}개 문장으로 나눔"
        else:
            a, b, n1 = self._sentence(text, op.sentence_a)
            c, d, n2 = self._sentence(text, op.sentence_b)
            if n2 != n1 + 1:
                raise Rejected("connect_not_adjacent", "읽는 순서에서 바로 이웃한 두 문장을 연결하세요.")
            if not op.connector or not op.connector.strip() or len(op.connector) > 30 or re.search(r"[.!?。！？\n]", op.connector):
                raise Rejected("invalid_connector", "새 문장이 아닌 30자 이내의 연결 표현을 지정하세요.")
            connector = op.connector.rstrip() + " "
            patches = [(c, c, connector)]
            touched, hints = [(c, d)], [connector + text[c:d]]
            source = text[c:d]
            label = f"{n1}번 문장과 이어지도록 {n2}번 문장 앞에 ‘{op.connector}’을 넣음"

        self._protect(text, touched, keep, points)
        result = text
        for a, b, replacement in sorted(patches, key=lambda p: p[0], reverse=True):
            result = result[:a] + replacement + result[b:]
        if not result.strip() or result == text:
            raise Rejected("empty_or_no_change", "빈 글 또는 실제 변경이 없는 연산입니다.")
        self._markers(text, result)
        count_before, count_after = len(self.sentence_spans(text)), len(self.sentence_spans(result))
        expected_delta = -1 if op.op == "merge" else len(op.sentences) - 1 if op.op == "split" else 0
        if count_after != count_before + expected_delta:
            raise Rejected("sentence_deletion", "연산 종류에서 허용하지 않은 문장 추가·삭제가 발생했습니다.")
        preserve = []
        for hint in hints:
            for a, b in quote_spans(result, hint):
                preserve.extend(s["text"] for s in self.sentence_spans(result) if s["start"] <= a and b <= s["end"])
        return result, {"operation": op.model_dump(), "description": label, "before": source,
                        "preserve_quotes": list(dict.fromkeys(preserve)),
                        "text_before": text, "text_after": result}
