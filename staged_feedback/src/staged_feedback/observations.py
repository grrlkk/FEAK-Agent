"""Passive surface observations. This module never accepts/rejects or edits text."""
from collections import Counter
from difflib import SequenceMatcher
import re


def changed_char_ratio(before, after):
    matched = sum(b.size for b in SequenceMatcher(None, before, after, autojunk=False).get_matching_blocks())
    return 1 - (2 * matched / (len(before) + len(after))) if before or after else 0.0


def overlaps(a, b, c, d):
    return a < d and c < b


def touches(a, b, c, d, *, include_start=False):
    return overlaps(a, b, c, d) if a != b else (c <= a < d if include_start else c < a < d)


class Observer:
    def __init__(self, *, sentence_spans, marker_pattern, placeholder_pattern):
        self.sentence_spans = sentence_spans
        self.marker_pattern, self.placeholder_pattern = marker_pattern, placeholder_pattern

    def targets(self, text, feedback):
        sentences, located, unlocated = self.sentence_spans(text), {}, []
        # Whitespace-tolerant lookup is for annotation only; ambiguity is retained.
        compact, offsets = [], []
        for index, char in enumerate(text):
            if not char.isspace():
                compact.append(char)
                offsets.append(index)
        compact = "".join(compact)
        for item in feedback:
            key, quote = item["issue_id"], item["target_quote"]
            needle = "".join(quote.split())
            ranges = []
            if needle:
                for match in re.finditer(f"(?={re.escape(needle)})", compact):
                    ranges.append((offsets[match.start()], offsets[match.start() + len(needle) - 1] + 1))
            matched = [s for s in sentences if any(overlaps(s["start"], s["end"], a, b) for a, b in ranges)]
            located[key] = list({s["id"]: s for s in [*located.get(key, []), *matched]}.values())
            if not ranges:
                unlocated.append(key)
        return located, unlocated

    def compare(self, before, after, *, feedback=(), previous_changed_ranges=(), target_ranges=()):
        old, new = self.sentence_spans(before), self.sentence_spans(after)
        char_ops = SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
        edits = [op for op in char_ops if op[0] != "equal"]
        changes, exact_old_indices, exact_new_indices = [], set(), set()
        for tag, a, b, c, d in SequenceMatcher(None, [s["text"] for s in old],
                                               [s["text"] for s in new], autojunk=False).get_opcodes():
            if tag == "equal":
                exact_old_indices.update(range(a, b))
                exact_new_indices.update(range(c, d))
                continue
            changes.append({"kind": tag, "before": before[old[a]["start"]:old[b-1]["end"]] if a < b else "",
                "after": after[new[c]["start"]:new[d-1]["end"]] if c < d else "",
                "before_sentences": old[a:b], "after_sentences": new[c:d]})
        # Sentence splitting ignores gaps. Preserve whitespace/paragraph-only edits too.
        gaps = [{"before": before[a:b], "after": after[c:d], "before_span": [a, b], "after_span": [c, d]}
                for _, a, b, c, d in edits if not any(touches(a, b, s["start"], s["end"]) for s in old)
                and not any(touches(c, d, s["start"], s["end"]) for s in new)]
        targets, unlocated = self.targets(before, feedback)
        for scope in target_ranges:
            for key in scope["issue_ids"]:
                matched = [s for s in old if overlaps(s["start"], s["end"], scope["start"], scope["end"])]
                targets[key] = list({s["id"]: s for s in [*targets.get(key, []), *matched]}.values())
                if key in unlocated:
                    unlocated.remove(key)
        target_ids = {s["id"] for items in targets.values() for s in items}
        touched = [s for s in old if any(touches(a, b, s["start"], s["end"], include_start=True) for _, a, b, _, _ in edits)]
        flags = []
        # A deletion observation describes surface alignment, never lost meaning.
        # Word alignment avoids treating a split/merge as deletion. Exact moved
        # sentences also survive even if the monotonic diff calls them deletions.
        tokens_old = list(re.finditer(r"\w+", before))
        tokens_new = list(re.finditer(r"\w+", after))
        surviving = set()
        for block in SequenceMatcher(None, [t.group() for t in tokens_old],
                                     [t.group() for t in tokens_new], autojunk=False).get_matching_blocks():
            surviving.update(range(block.a, block.a + block.size))
        exact_survivors = Counter(s["text"] for index, s in enumerate(new) if index not in exact_new_indices)
        for index, sentence in enumerate(old):
            if index in exact_old_indices:
                continue
            if exact_survivors[sentence["text"]]:
                exact_survivors[sentence["text"]] -= 1
                continue
            indices = [i for i, t in enumerate(tokens_old) if overlaps(t.start(), t.end(), sentence["start"], sentence["end"])]
            if indices and not surviving.intersection(indices):
                flags.append({"kind": "student_sentence_deleted", "sentence": sentence["text"],
                    "before_span": [sentence["start"], sentence["end"]],
                    "reason": "이전 문장의 단어가 전후 정렬에서 남지 않았습니다. 삭제인지 전면 재서술인지는 총괄이 판단합니다."})
        for sentence in touched:
            if any(overlaps(sentence["start"], sentence["end"], a, b) for a, b in previous_changed_ranges):
                flags.append({"kind": "earlier_revision_changed", "sentence": sentence["text"],
                    "reason": "앞 턴에서 바뀐 문장이 다시 변경됐습니다."})
            if sentence["id"] not in target_ids:
                flags.append({"kind": "outside_feedback_target", "sentence": sentence["text"],
                    "reason": "이번 턴 피드백의 인용으로 찾은 대상 문장 밖의 변경입니다."})
        for change in changes:
            if change["kind"] == "insert":
                flags.append({"kind": "outside_feedback_target", "sentence": change["after"],
                    "reason": "이전 글에 대응 문장이 없는 삽입입니다. 피드백과의 관련성은 총괄이 판단합니다."})
        before_markers, after_markers = self.marker_pattern.findall(before), self.marker_pattern.findall(after)
        if before_markers != after_markers or "__ANON_" in after:
            flags.append({"kind": "anonymization_changed", "before": before_markers, "after": after_markers,
                "unrestored_placeholders": self.placeholder_pattern.findall(after),
                "malformed_placeholder_present": "__ANON_" in self.placeholder_pattern.sub("", after),
                "reason": "비식별 표시의 값·개수·순서 또는 복원되지 않은 치환 표시가 달라졌습니다."})
        next_ranges = []
        for sentence in new:
            changed = any(touches(c, d, sentence["start"], sentence["end"]) for _, _, _, c, d in edits)
            inherited = any(overlaps(a, b, p, q) and overlaps(
                c + max(a, p) - a, c + min(b, q) - a, sentence["start"], sentence["end"])
                for tag, a, b, c, d in char_ops if tag == "equal" for p, q in previous_changed_ranges)
            if changed or inherited:
                next_ranges.append([sentence["start"], sentence["end"]])
        return {"changed_sentences": changes, "layout_changes": gaps, "flags": flags,
            "feedback_targets": targets, "unlocated_feedback_issue_ids": unlocated,
            "next_changed_ranges": next_ranges}
