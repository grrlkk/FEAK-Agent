"""Place scoped outputs and coordinator-requested restorations; no quality verdicts."""
import re


def document_units(text, sentence_spans):
    paragraphs = []
    for match in re.finditer(r"[^\r\n]+", text):
        a, b = match.start(), match.end()
        while a < b and text[a].isspace():
            a += 1
        while a < b and text[b - 1].isspace():
            b -= 1
        if a < b:
            paragraphs.append(dict(id=f"p{len(paragraphs)+1}", text=text[a:b], start=a, end=b))
    return {"sentences": sentence_spans(text), "paragraphs": paragraphs}


def public_units(units):
    # Character offsets belong to the local unmasked text; models select stable IDs.
    return {kind: [{"id": item["id"], "text": item["text"]} for item in items]
            for kind, items in units.items()}


def revision_scopes(text, feedback, issues, sentence_spans):
    units = document_units(text, sentence_spans)
    scopes, unavailable = [], []
    for item in feedback:
        if item.needs_student_input:
            continue
        issue = issues.get(item.issue_id)
        if issue is None:
            unavailable.append(dict(issue_id=item.issue_id, reason="대상 이슈의 범위를 확인할 수 없습니다."))
            continue
        kind = "paragraphs" if issue.level == "macro" else "sentences"
        indexed = {unit["id"]: unit for unit in units[kind]}
        selected = [indexed[key] for key in dict.fromkeys(item.target_ids) if key in indexed]
        if item.target_ids and len(selected) != len(set(item.target_ids)):
            selected = []
        if not selected:
            # A paragraph ID on a micro item is not grounds to skip an otherwise
            # located target. Use the unique quote, then expand only to the proper
            # sentence/paragraph units. Never guess between repeated quotes.
            quote = item.target_quote
            start = text.find(quote) if quote else -1
            if start >= 0 and text.find(quote, start + 1) < 0:
                selected = [unit for unit in units[kind] if unit["start"] < start + len(quote) and start < unit["end"]]
        if not selected:
            unavailable.append(dict(issue_id=item.issue_id, reason="피드백의 대상 범위를 유일하게 찾지 못했습니다."))
            continue
        for unit in selected:
            scopes.append(dict(start=unit["start"], end=unit["end"], target_ids=[unit["id"]], feedback=[item]))
    merged = []
    for scope in sorted(scopes, key=lambda s: (s["start"], s["end"])):
        # The same feedback can reorder adjacent target paragraphs. Never pull an
        # intervening untargeted sentence into a scope to bridge a gap.
        adjacent_same_feedback = (merged and scope["feedback"] == merged[-1]["feedback"]
            and not text[merged[-1]["end"]:scope["start"]].strip())
        if merged and (scope["start"] < merged[-1]["end"] or adjacent_same_feedback):
            kept = merged[-1]
            kept["end"] = max(kept["end"], scope["end"])
            kept["target_ids"] = list(dict.fromkeys(kept["target_ids"] + scope["target_ids"]))
            for item in scope["feedback"]:
                if item not in kept["feedback"]:
                    kept["feedback"].append(item)
        else:
            merged.append(scope)
    for index, scope in enumerate(merged, 1):
        before = [s for s in units["sentences"] if s["end"] <= scope["start"]]
        after = [s for s in units["sentences"] if s["start"] >= scope["end"]]
        scope.update(scope_id=f"scope{index}", source_text=text[scope["start"]:scope["end"]],
            context_before=before[-1]["text"] if before else "", context_after=after[0]["text"] if after else "")
    return merged, unavailable


def assemble(text, replacements):
    for item in sorted(replacements, key=lambda item: item["start"], reverse=True):
        text = text[:item["start"]] + item["replacement"] + text[item["end"]:]
    return text


def restore_sentences(previous, current, requests, sentence_spans):
    old, new = sentence_spans(previous), sentence_spans(current)
    old_by_id, new_by_id = {s["id"]: s for s in old}, {s["id"]: s for s in new}
    records, replacements = [], []

    def span(keys, units, lookup):
        if not keys or any(key not in lookup for key in keys):
            raise ValueError("지정한 문장 ID를 찾을 수 없습니다.")
        indices = sorted({units.index(lookup[key]) for key in keys})
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("떨어진 문장은 복원 항목을 나눠 지정해야 합니다.")
        return units[indices[0]]["start"], units[indices[-1]]["end"]

    for request in requests:
        record = request.model_dump()
        try:
            a, b = span(request.previous_sentence_ids, old, old_by_id)
            replacement = previous[a:b]
            if request.current_sentence_ids:
                c, d = span(request.current_sentence_ids, new, new_by_id)
            else:
                anchor = request.insert_before_sentence_id
                if anchor is not None and anchor not in new_by_id:
                    raise ValueError("삭제 문장을 되살릴 삽입 위치를 찾을 수 없습니다.")
                c = d = new_by_id[anchor]["start"] if anchor else len(current)
                if c and not current[c-1].isspace():
                    replacement = " " + replacement
                if c < len(current) and not current[c].isspace():
                    replacement += " "
            if any((c < p["end"] and p["start"] < d) or
                   (c == d and p["start"] < c < p["end"]) or
                   (p["start"] == p["end"] and c < p["start"] < d) for p in replacements):
                raise ValueError("다른 복원 지시와 대상 범위가 겹칩니다.")
            if any(p["start"] == c and p["end"] == d and p["replacement"] == replacement for p in replacements):
                record.update(status="duplicate", detail="같은 복원 지시는 한 번 실행합니다.")
            else:
                replacements.append(dict(start=c, end=d, replacement=replacement, original_start=a))
                record.update(status="applied", before=current[c:d], after=replacement,
                              previous_span=[a, b], current_span=[c, d])
        except ValueError as exc:
            # Mapping failures are visible, with the candidate preserved. No text
            # is rejected and no automatic quality judgment triggers restoration.
            record.update(status="unavailable", detail=str(exc))
        records.append(record)
    # Several deleted sentences can share one anchor; retain their original order.
    for item in sorted(replacements, key=lambda p: (p["start"], p["original_start"]), reverse=True):
        current = current[:item["start"]] + item["replacement"] + current[item["end"]:]
    return current, records
