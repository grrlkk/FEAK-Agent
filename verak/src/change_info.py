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


def _touches(span, start, end):
    a, b = span
    return start <= a <= end if a == b else start < b and a < end


def _word_span(text, span):
    """Expand character fragments to readable surface words, not restored forms."""
    a, b = span
    if a == b and not (0 < a < len(text) and not text[a-1].isspace() and not text[a].isspace()):
        return [a, b]
    while a > 0 and not text[a-1].isspace():
        a -= 1
    while b < len(text) and not text[b].isspace():
        b += 1
    return [a, b]


def _reconstruct(before, after, changes):
    current = before
    for change in reversed(changes):
        a, b = change['before_span']
        c, d = change['after_span']
        current = current[:a] + after[c:d] + current[b:]
    return current


def readable_edits(before, after, plan, after_profile=None):
    """Every edit covers contiguous source text; reconstruction must be exact.

    An ADD is anchored to the actual plan, avoiding difflib's relocation of a
    repeated '다.' into the preceding sentence. Rewrites expand character edits
    to words and merge overlapping spans. Whitespace is never discarded.
    """
    raw = surface_diff(before, after)
    point = plan.target.insertion_at
    if (plan.action == 'ADD' and point is not None and len(after) >= len(before)
            and after[:point] == before[:point]
            and after[point + len(after) - len(before):] == before[point:]):
        end = point + len(after) - len(before)
        boundaries = [point]
        if after_profile:
            boundaries += [s.end for s in after_profile.sentences if point < s.end < end]
        boundaries.append(end)
        changes = []
        for a, b in zip(boundaries, boundaries[1:]):
            if a == b:
                continue
            if not after[a:b].strip() and changes:
                changes[-1]['after_span'][1] = b
            else:
                changes.append({'before_span': [point, point], 'after_span': [a, b]})
    else:
        changes = []
        for item in raw:
            change = {'before_span': _word_span(before, item['before_span']),
                      'after_span': _word_span(after, item['after_span'])}
            if changes and any(change[side][0] < changes[-1][side][1]
                               for side in ('before_span', 'after_span')):
                for side in ('before_span', 'after_span'):
                    changes[-1][side][0] = min(changes[-1][side][0], change[side][0])
                    changes[-1][side][1] = max(changes[-1][side][1], change[side][1])
            else:
                changes.append(change)
        # Ambiguous alignment must not omit a change. Coalesce to one contiguous
        # source span when expanded boundaries cannot reproduce the exact patch.
        if _reconstruct(before, after, changes) != after:
            changes = ([{'before_span': [raw[0]['before_span'][0], raw[-1]['before_span'][1]],
                         'after_span': [raw[0]['after_span'][0], raw[-1]['after_span'][1]]}]
                       if raw else [])
    assert _reconstruct(before, after, changes) == after, 'Edit extraction lost source text'
    edits = []
    for change in changes:
        old = before[slice(*change['before_span'])]
        new = after[slice(*change['after_span'])]
        if old == new:
            continue
        edits.append({'id': f'E{len(edits)+1}', **change,
                      'operation': 'replace' if old and new else 'delete' if old else 'insert',
                      'before_text': old, 'after_text': new})
    return edits


def _context(profile, span, text):
    if not profile:
        return {'span': span, 'text': text[slice(*span)], 'sentence_ids': []}
    hit = [i for i, s in enumerate(profile.sentences) if _touches(span, s.start, s.end)]
    if not hit:
        return {'span': span, 'text': text[slice(*span)], 'sentence_ids': []}
    sentences = profile.sentences[max(0, hit[0]-1):hit[-1]+2]
    a, b = sentences[0].start, sentences[-1].end
    return {'span': [a, b], 'text': text[a:b], 'sentence_ids': [s.id for s in sentences]}


def linguistic_transitions(before, after, before_profile, after_profile, edit):
    """Aligned changes only: never a flat, discontinuous pseudo-sentence."""
    groups = []
    for side, text, profile in (('before', before, before_profile), ('after', after, after_profile)):
        span = edit[side + '_span']
        groups.append([t for s in (profile.sentences if profile else []) for t in s.tokens
                       if span[0] < span[1] and _touches(span, t.start, t.end)])
    left, right = groups
    result = []
    for op, i, j, k, l in SequenceMatcher(None, [(t.form, t.tag) for t in left],
                                         [(t.form, t.tag) for t in right], autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        old, new = left[i:j], right[k:l]
        kinds = {'final_ending': lambda t: t.tag == 'EF',
                 'connective_ending': lambda t: t.tag == 'EC',
                 'particle': lambda t: t.tag.startswith('J')}
        if any(t.form in NEGATION and t.tag in {'MAG', 'VX', 'VA', 'VCN', 'VV'} for t in old + new):
            kinds['negation'] = lambda t: True
        for kind, include in kinds.items():
            a, b = [t for t in old if include(t)], [t for t in new if include(t)]
            if not a and not b:
                continue
            def source(tokens, text):
                return [{'form': t.form, 'tag': t.tag, 'span': [t.start, t.end],
                         'surface': text[t.start:t.end]} for t in tokens]
            result.append({'kind': kind, 'operation': op,
                           'before': source(a, before), 'after': source(b, after)})
    return result


def plan_edit_budget(plan):
    """Read explicit count phrases without changing the Planner contract.

    Evidence/preserve quotations are excluded: numbers in the essay are not a
    requested budget. Unspecified edit counts remain unspecified.
    """
    text = plan.goal + '\n' + plan.minimal_scope_reason
    values = []
    for match in re.finditer(r'(\d+)\s*(?:[~～–-]|에서)\s*(\d+)\s*(?:개(?:의)?\s*)?(?:문장|sentences?)', text, re.I):
        values.append(int(match[2]))
    for match in re.finditer(r'(\d+)\s*(?:개(?:의)?\s*)?(?:문장|sentences?)', text, re.I):
        values.append(int(match[1]))
    if re.search(r'한\s*두\s*(?:개의\s*)?문장', text):
        values.append(2)
    for word, number in (('한', 1), ('하나의', 1), ('두', 2), ('세', 3), ('네', 4)):
        if re.search(rf'(?<![가-힣]){word}\s*(?:개의\s*)?문장', text):
            values.append(number)
    for word, number in (('하나', 1), ('둘', 2), ('둘째', 2)):
        if re.search(rf'문장\s*{word}(?:만|를|을|로|가|이|\s|$)', text):
            values.append(number)
    explicit = max(values) if values else None
    maximum = explicit if explicit is not None else 1 if plan.scope == 'sentence' else None
    edit_limits = [int(m[1]) for m in re.finditer(r'(?:변경|수정)\s*(?:개수\s*[:=]?\s*)?(\d+)\s*(?:개|곳|회)', text)]
    return {'max_sentences': maximum, 'sentence_budget_source': 'explicit_plan_text' if explicit is not None
            else 'sentence_scope_default' if maximum else 'unspecified',
            'max_edits': min(edit_limits) if edit_limits else None}


def edit_verification_info(before, after, plan, before_profile, after_profile):
    edits = readable_edits(before, after, plan, after_profile)
    for edit in edits:
        edit['context'] = {'before': _context(before_profile, edit['before_span'], before),
                           'after': _context(after_profile, edit['after_span'], after)}
        edit['transitions'] = linguistic_transitions(before, after, before_profile, after_profile, edit)
    a, b = plan.span_before
    replacement_end = len(after) - (len(before) - b)
    old_sentences = [s for s in (before_profile.sentences if before_profile else [])
                     if a < b and a < s.end and s.start < b]
    new_sentences = [s for s in (after_profile.sentences if after_profile else [])
                     if a < replacement_end and a < s.end and s.start < replacement_end]
    if plan.action == 'ADD':
        sentence_count = sum(a <= s.start and s.end <= replacement_end for s in new_sentences)
    else:
        sentence_count = len(new_sentences)
    budget = plan_edit_budget(plan)
    reasons, style_checks = [], []
    if budget['max_sentences'] is not None and sentence_count > budget['max_sentences']:
        reasons.append('sentence_budget_exceeded')
    if budget['max_edits'] is not None and len(edits) > budget['max_edits']:
        reasons.append('edit_budget_exceeded')
    # A known register mismatch is a code gate. Ambiguous/unmapped endings are
    # evidence for the verifier, never guessed into a deterministic violation.
    if before_profile and after_profile:
        aligned = align(before_profile.sentences, after_profile.sentences)
        for i, j, k, l, _, ambiguous in aligned:
            old = before_profile.sentences[i:j]
            new = after_profile.sentences[k:l]
            if not any(any(_touches(e['after_span'], s.start, s.end) for e in edits) for s in new):
                continue
            if plan.action == 'ADD':
                old = [s for s in before_profile.sentences
                       if s.start < plan.target.end and plan.target.start < s.end]
            labels = {v for s in old for v in s.style_candidates}
            for sentence in new:
                new_labels = set(sentence.style_candidates)
                definite = len(labels) == len(new_labels) == 1 and not ambiguous
                mismatch = definite and labels != new_labels
                style_checks.append({'before_sentence_ids': [s.id for s in old],
                    'after_sentence_id': sentence.id, 'before': sorted(labels), 'after': sorted(new_labels),
                    'status': 'fail' if mismatch else 'pass' if definite else 'unknown'})
                if mismatch:
                    reasons.append('speech_level_changed')
    return {'edits': edits, 'edit_count': len(edits), 'raw_diff_count': len(surface_diff(before, after)),
            'counts': {'target_sentences_before': len(old_sentences), 'target_sentences_after': len(new_sentences),
                       'budgeted_sentence_count': sentence_count}, 'budget': budget,
            'style_checks': style_checks, 'hard_reasons': sorted(set(reasons))}
