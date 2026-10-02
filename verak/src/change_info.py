"""Sentence alignment, morpheme edits and source-preserving surface differences."""

from dataclasses import asdict
from difflib import SequenceMatcher
import math
import re

from .schemas import Unit

NOUNS = {"NNG", "NNP", "NNB", "NP", "NR"}
CASE = {"JKS", "JKO", "JKC", "JKG", "JKB", "JKV", "JKQ"}
NEGATION = {"안", "못", "않", "아니", "없", "말"}


def surface_diff(before, after, before_offset=0, after_offset=0):
    return [{"operation": tag, "before_span": [i + before_offset, j + before_offset],
             "after_span": [k + after_offset, l + after_offset],
             "before_text": before[i:j], "after_text": after[k:l]}
            for tag, i, j, k, l in SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
            if tag != "equal"]


def _similarity(left, right):
    # This normalization affects only alignment scores, never recorded offsets.
    a = re.sub(r"\s+", "", "".join(s.text for s in left))
    b = re.sub(r"\s+", "", "".join(s.text for s in right))
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def align(before_sentences, after_sentences):
    """Monotonic DP over 1:1, 1:2, 2:1, 0:1 and 1:0 moves."""
    n, m = len(before_sentences), len(after_sentences)
    cost = [[math.inf] * (m + 1) for _ in range(n + 1)]
    back = {}
    cost[0][0] = 0.0
    moves = ((1, 1), (1, 2), (2, 1), (0, 1), (1, 0))
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            options = []
            for a, b in moves:
                if i < a or j < b:
                    continue
                sim = _similarity(before_sentences[i-a:i], after_sentences[j-b:j]) if a and b else 0.0
                # Unrelated sentences can align as a deletion and an insertion.
                step = 2 * (1 - sim) + (0.18 if a + b == 3 else 0) if a and b else 0.85
                options.append((cost[i-a][j-b] + step, a, b, sim))
            options.sort(key=lambda item: item[0])
            best = options[0]
            cost[i][j] = best[0]
            ambiguous = len(options) > 1 and options[1][0] - best[0] < 0.08
            back[i, j] = (*best[1:], ambiguous)
    result = []
    i, j = n, m
    while i or j:
        a, b, sim, ambiguous = back[i, j]
        result.append((i-a, i, j-b, j, sim, ambiguous))
        i, j = i-a, j-b
    return list(reversed(result))


def _block_span(sentences, first, end, text_length):
    if first < end:
        return [sentences[first].start, sentences[end - 1].end]
    point = sentences[first].start if first < len(sentences) else text_length
    return [point, point]


def _morph_changes(left, right):
    a = [t for s in left for t in s.tokens]
    b = [t for s in right for t in s.tokens]
    changes = []
    for tag, i, j, k, l in SequenceMatcher(None, [(t.form, t.tag) for t in a],
                                          [(t.form, t.tag) for t in b], autojunk=False).get_opcodes():
        if tag != "equal":
            changes.append({"kind": "morpheme_edit", "operation": tag,
                            "before": [asdict(t) for t in a[i:j]], "after": [asdict(t) for t in b[k:l]]})
    return changes


def _types(changes, before, after, alignment):
    if before != after and re.sub(r"\s", "", before) == re.sub(r"\s", "", after):
        return ["spacing"]
    labels = set()
    for change in changes:
        left, right = change["before"], change["after"]
        changed = left + right
        tags = {t["tag"] for t in changed}
        if "EC" in tags: labels.add("connective")
        if "EF" in tags: labels.add("ender")
        if tags & CASE: labels.add("particle_case")
        if "JX" in tags: labels.add("particle_focus")
        if tags & NOUNS: labels.add("noun")
        if any(t["form"] in NEGATION and t["tag"] in {"MAG", "VX", "VA", "VCN", "VV"} for t in changed):
            labels.add("negation")
        known = NOUNS | CASE | {"JX", "EC", "EF", "XSN", "SF", "SP", "SS", "SSO", "SSC"}
        if any(t["tag"] not in known and t["form"] not in NEGATION for t in changed):
            labels.add("content")
        # These indicate a surface argument candidate, never a resolved ellipsis.
        if len([t for t in right if t["tag"] in NOUNS]) > len([t for t in left if t["tag"] in NOUNS]):
            if any(t["tag"] in {"JKS", "JKO", "JX"} for t in right): labels.add("arg_insert")
        if len([t for t in left if t["tag"] in NOUNS]) > len([t for t in right if t["tag"] in NOUNS]):
            if any(t["tag"] in {"JKS", "JKO", "JX"} for t in left): labels.add("arg_delete")
    if alignment == "1:2": labels.add("split")
    if alignment == "2:1": labels.add("merge")
    if not labels or alignment in {"0:1", "1:0"}: labels.add("content")
    return sorted(labels)


def extract(before, after, before_profile, after_profile, focus_lexicon=None):
    units = []
    left, right = before_profile.sentences, after_profile.sentences
    for i, j, k, l, similarity, ambiguous in align(left, right):
        bs, ats = _block_span(left, i, j, len(before)), _block_span(right, k, l, len(after))
        old, new = before[slice(*bs)], after[slice(*ats)]
        if old == new:
            continue
        alignment = f"{j-i}:{l-k}"
        observed = _morph_changes(left[i:j], right[k:l])
        observed.insert(0, {"kind": "sentence_alignment", "before_ids": [s.id for s in left[i:j]],
                           "after_ids": [s.id for s in right[k:l]], "similarity": similarity})
        labels = _types(observed[1:], old, new, alignment)
        interpretations = []
        for side, group in (("before", left[i:j]), ("after", right[k:l])):
            for sentence in group:
                interpretations.append({"side": side, "sentence_id": sentence.id,
                    "style_candidates": sentence.style_candidates, "connective_candidates": sentence.connectives,
                    "antecedent_candidates": sentence.antecedent_candidates,
                    "focus_candidates": [{"form": t.form, "span": [t.start, t.end],
                        "candidates": (focus_lexicon or {}).get(t.form, [])}
                        for t in sentence.tokens if t.tag == "JX"]})
        uncertain = sorted({u for s in left[i:j] + right[k:l] for u in s.uncertain})
        if ambiguous: uncertain.append("alignment_has_near_equal_alternative")
        if j > i and l > k and similarity < 0.6: uncertain.append("low_sentence_similarity")
        if "arg_insert" in labels or "arg_delete" in labels:
            uncertain.append("argument_reference_and_recoverability_not_resolved")
        units.append(Unit(f"U{len(units)+1:04d}", bs, ats, old, new, labels, observed,
                          interpretations, uncertain, alignment, surface_diff(old, new, bs[0], ats[0])))
    # Include whitespace/boundary edits outside sentence spans and any missed raw changes.
    for edit in surface_diff(before, after):
        def covered(unit):
            return (unit.before_span[0] <= edit["before_span"][0] <= edit["before_span"][1] <= unit.before_span[1]
                    and unit.after_span[0] <= edit["after_span"][0] <= edit["after_span"][1] <= unit.after_span[1])
        if not any(covered(unit) for unit in units):
            old, new = edit["before_text"], edit["after_text"]
            labels = ["spacing"] if re.sub(r"\s", "", old) == re.sub(r"\s", "", new) else ["content"]
            units.append(Unit(f"U{len(units)+1:04d}", edit["before_span"], edit["after_span"], old, new,
                              labels, [{"kind": "surface_only", **edit}], [],
                              ["not_fully_covered_by_sentence_alignment"], "surface", [edit]))
    return units


def changed_korean_info(before_profile, after_profile, diff):
    """Only morphology overlapping actual edits, plus affected-sentence style candidates.

    For a zero-width insertion/deletion anchor, include touching morphemes. The
    full source/revision texts supply discourse context; no full profile or
    unrelated antecedent lists are sent to the RV.
    """
    result = {}
    for side, profile in (("before", before_profile), ("after", after_profile)):
        spans = [edit[f"{side}_span"] for edit in diff]

        def touched(start, end):
            return any((start <= a <= end) if a == b else (start < b and a < end)
                       for a, b in spans)

        sentences = []
        for sentence in profile.sentences:
            if not touched(sentence.start, sentence.end):
                continue
            sentences.append({
                "id": sentence.id, "span": [sentence.start, sentence.end],
                "tokens": [asdict(t) for t in sentence.tokens if touched(t.start, t.end)],
                "style_candidates": sentence.style_candidates,
                "connectives": [c for c in sentence.connectives if touched(*c["span"])],
                "subjects": [s for s in sentence.subjects if touched(*s["span"])],
                "uncertain": sentence.uncertain,
            })
        result[side] = sentences
    return result
