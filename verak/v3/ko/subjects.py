"""Conservative mention filtering and occurrence-specific anonymized entities."""

import re

from .types import SubjInfo

EXCLUDED = set("것 점 수 때 데 바 등 측 중 뿐 사실 요즘 현재 오늘날 물론 지금 이제 당시 오늘 어제 내일 최근 과거 미래 결국 처음 이번 다음 이전 이후 요새 올해 작년 내년 순간 때때로 먼저 마지막".split())
TIME_UNITS = {"년", "월", "일", "시", "분", "초", "주", "주일", "개월", "세기"}
NOUNS = {"NNG", "NNP", "NNB", "NP", "NR", "XPN"}
MARKER = re.compile(r"#@[^#\n]+#")


def relative_subject(tokens, marker_index):
    """Reject a local embedded subject before ETM + nominal head.

    A newer subject/topic marker starts a different subject's domain, so an
    outer topic in '학생은 내가 읽은 책을 ...' is retained. This is a heuristic,
    not a syntactic parse; it deliberately avoids certifying all such clauses.
    """
    for j in range(marker_index + 1, len(tokens)):
        token = tokens[j]
        if token.tag in {"EF", "SF"} or token.tag == "JKS" or (
                token.tag == "JX" and token.form in {"은", "는"}):
            return False
        if token.tag == "ETM":
            return any(t.tag in NOUNS for t in tokens[j + 1:j + 4])
        if token.tag == "EC" and token.form in {"면", "으면", "니까", "으니까", "지만", "므로"}:
            return False
    return False


def subject_candidates(sentence):
    tokens = sentence.tokens
    markers = [(m.start() + sentence.start, m.end() + sentence.start, m.group())
               for m in MARKER.finditer(sentence.text)]
    raw = [(c["span"][0], c["span"][1], "JKS" if c["kind"] == "subject_candidate" else "TOPIC")
           for c in sentence.subjects]
    # Bareun tokenizes #@...# into SW/NNG/SW. Restore the complete protected NP
    # when the following token is a subject/topic marker.
    for start, end, _ in markers:
        for t in tokens:
            if t.start == end and (t.tag == "JKS" or t.tag == "JX" and t.form in {"은", "는"}):
                raw.append((start, t.end, "JKS" if t.tag == "JKS" else "TOPIC"))
    result = []
    for start, end, kind in raw:
        marker_index = next((i for i, t in reversed(list(enumerate(tokens)))
                             if t.end == end and (t.tag == "JKS" or t.tag == "JX")), None)
        if marker_index is None or relative_subject(tokens, marker_index):
            continue
        prefixes = [t for t in tokens if t.tag == "XPN" and t.end == start]
        if prefixes:
            start = prefixes[-1].start
        # Include a nearby possessive anonymous prefix, including '#@이름#네 반'.
        # Stop across a predicate, case marker, punctuation, or another entity.
        for a, b, literal in reversed(markers):
            if a >= end:
                continue
            if b <= start:
                between = [t for t in tokens if b <= t.start < start]
                if any(t.tag not in NOUNS | {"XSN", "JKG", "MMD", "MMN"} for t in between):
                    break
                start = a
            break
        nouns = [t for t in tokens if start <= t.start < end and t.tag in NOUNS]
        contained = [(a, b, lit) for a, b, lit in markers if start <= a and b <= end]
        if not nouns and not contained:
            continue
        if not contained and (nouns[-1].form in EXCLUDED or
                             any(t.form in TIME_UNITS for t in nouns) and
                             any(t.tag in {"SN", "NR"} for t in tokens if start <= t.start < end)):
            continue
        surface = sentence.text[start - sentence.start:end - sentence.start]
        lemma = "".join(t.form for t in nouns)
        entity = None
        if contained:
            # Identical anonymization labels do not prove that two occurrences
            # denote the same person. Preserve each literal and occurrence span.
            a, b, literal = contained[-1]
            entity = f"{literal}@{a}:{b}"
            suffix = "".join(t.form for t in nouns if t.start >= b)
            lemma = entity + (":" + suffix if suffix else "")
        result.append(SubjInfo(True, surface, kind, lemma, [start, end], entity))
    if not result and len(tokens) >= 3 and tokens[0].tag in {"NR", "NP"} and tokens[1].form in {"다", "모두"} and tokens[1].tag == "MAG":
        start, end = tokens[0].start, tokens[1].end
        result.append(SubjInfo(True, sentence.text[start - sentence.start:end - sentence.start],
                              "NONE", tokens[0].form, [start, end]))
    return list({(x.lemma, tuple(x.span)): x for x in result}.values())
