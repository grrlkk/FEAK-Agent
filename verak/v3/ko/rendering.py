"""Full/compact views; display shortening never mutates the stored document."""

UNCERTAIN = {"antecedent": "선행사", "subject": "주어", "topic": "화제",
             "connectives": "연결", "style": "문체", "style_shift": "문체변화",
             "polarity": "극성", "modality": "양태", "modality_ill_formed": "양태구성",
             "relation_target": "접속대상"}
REL_SHORT = {"CAUSE": "원인", "SEQUENCE": "순서", "ADDITION": "나열", "CONTRAST": "대조",
             "CONCESSION": "양보", "CONDITION": "조건", "DISCOVERY": "발견", "BACKGROUND": "배경",
             "PURPOSE": "목적", "SIMULTANEOUS": "동시", "RESULT": "결과", "EXAMPLE": "예시",
             "RESTATEMENT": "환언", "REASON": "이유"}


def middle_ellipsis(text, limit):
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…" if limit else ""
    left = (limit - 1) // 2
    return text[:left] + "…" + text[-(limit - left - 1):]


def _flags(ann, structure, compact):
    flags = []
    if ann.subject.realized:
        marker = "(화제)" if ann.subject.marker == "TOPIC" else ""
        flags.append("주어:" + middle_ellipsis(ann.subject.surface, 24) + marker)
    elif ann.antecedent.status == "resolved":
        flags.append("주어 생략→" + ",".join(ann.antecedent.targets))
    elif ann.antecedent.status == "ambiguous":
        flags.append("주어?→" + ",".join(ann.antecedent.targets))
    else:
        flags.append("주어?")
    if ann.topic and ann.subject.marker != "TOPIC":
        flags.append("화제:" + middle_ellipsis(ann.topic, 16))
    if ann.initial_conj:
        relation = REL_SHORT[ann.initial_conj.relation]
        dst = next((edge.dst for edge in structure.edges if edge.kind == "REL" and edge.src == ann.sid), "?")
        flags.append(f"접속:{ann.initial_conj.form}({relation})←{dst}")
    if ann.connectives and not compact:
        values = list(dict.fromkeys(f"{conn.form}({ '/'.join(REL_SHORT[c] for c in conn.candidates) or '?' })"
                                   for conn in ann.connectives))
        flags.append("EC:" + ",".join(values))
    if ann.polarity != "POS":
        flags.append("극성:" + ann.polarity)
    if ann.modality is not None:
        flags.append("양태:" + ann.modality)
    flags.append(ann.style + ("체" if ann.style not in {"mixed", "unknown"} else ""))
    if ann.uncertain:
        flags.append("?" + "/".join(UNCERTAIN.get(value, value) for value in ann.uncertain))
    return " | ".join(flags)


def compact_flags(ann, structure):
    """Keep every decision-bearing flag and edge, without shortening any content."""
    flags = []
    if not ann.subject.realized:
        targets = ",".join(ann.antecedent.targets)
        flags.append("주어∅:" + ann.antecedent.status + ("→" + targets if targets else ""))
    topics = list(dict.fromkeys(candidate.surface + (
                               "@" + candidate.entity_id.rsplit("@", 1)[1] if candidate.entity_id else "")
                               for candidate in ann.subject_candidates
                               if candidate.marker == "TOPIC"))
    if topics:
        flags.append("화제:" + ",".join(topics))
    if ann.style != structure.dominant_style:
        flags.append("문체:" + ann.style)
    if ann.multi_unit:
        flags.append("복수종결?")
    relations = [f"{conn.token_id}:{conn.form}({','.join(conn.candidates)})" +
                 ("?" if conn.classification == "AMBIGUOUS" else "")
                 for conn in ann.connectives if conn.candidates]
    if relations:
        flags.append("EC:" + ";".join(relations))
    if ann.initial_conj:
        flags.append(f"접속:{ann.initial_conj.form}({ann.initial_conj.relation})")
    edges = [f"{edge.kind}→{edge.dst}({edge.label})" + ("?" if edge.confidence == "LOW" else "")
             for edge in structure.edges if edge.src == ann.sid]
    flags.extend(edges)
    focus = [f"{p['token_id']}:{p['form']}({p['function']})" for p in ann.focus_particles
             if p["function"] != "TOPIC_CONTRAST"]
    if focus:
        flags.append("초점:" + ";".join(focus))
    if ann.polarity != "POS":
        flags.append("극성:" + ann.polarity)
    if ann.modality is not None:
        flags.append("양태:" + ann.modality)
    if ann.uncertain:
        flags.append("?" + "/".join(UNCERTAIN.get(value, value) for value in ann.uncertain))
    return " | ".join(flags)


def render(structure, *, compact=True, max_line_chars=200):
    lines = [f"[주문체:{structure.dominant_style}]"] if compact else []
    paragraph = None
    for ann in structure.annotations:
        if ann.paragraph != paragraph:
            paragraph = ann.paragraph
            lines.append(f"[{paragraph}]")
        if compact:
            flags = compact_flags(ann, structure)
            lines.append(ann.sid + (" | " + flags if flags else ""))
        else:
            flags = _flags(ann, structure, False)
            allowance = max(1, max_line_chars - len(ann.sid) - len(flags) - 6)
            display = middle_ellipsis(ann.text.replace("\t", " "), allowance)
            lines.append(f"{ann.sid} | {display} | {flags}")
    return "\n".join(lines)


def render_with_budget(structure, count_tokens, token_budget=3000):
    if token_budget < 1:
        raise ValueError("Token budget must be positive")
    full = render(structure, compact=False)
    full_tokens = count_tokens(full)
    result = render(structure, compact=True)
    tokens = count_tokens(result)
    return {"text": result, "mode": "compact",
            "full_tokens": full_tokens, "tokens": count_tokens(result),
            "over_budget": tokens > token_budget, "eligible": tokens <= token_budget}
