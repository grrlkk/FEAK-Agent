"""Compact episode view with no inferred antecedent identity."""


def render_structural(structure, *, compact=True, max_line_chars=200):
    lines = [f"[주문체:{structure.dominant_style}]"]
    paragraph = None
    for ann in structure.annotations:
        if ann.paragraph != paragraph:
            paragraph = ann.paragraph
            lines.append(f"[{paragraph}]")
        flags = []
        if ann.subject_omitted:
            flags.append("주어∅")
        topics = [v["surface"] + ("@" + v["entity_id"].rsplit("@", 1)[1] if v.get("entity_id") else "")
                  for v in ann.subject_evidence if v["marker"] in {"은", "는"}]
        if topics:
            flags.append("화제:" + ",".join(dict.fromkeys(topics)))
        if ann.style != structure.dominant_style or ann.style == "unknown":
            flags.append("문체:" + ("?" if ann.style == "unknown" else ann.style))
        if ann.multi_unit:
            flags.append("복수종결?")
        relations = [f"{c['token_id']}:{c['form']}({c['coarse_class']})" for c in ann.connectives if c["eligible"]]
        if relations:
            flags.append("관계:" + ";".join(relations))
        if ann.initial_conj:
            c = ann.initial_conj
            flags.append(f"접속:{c['form']}({c['visible_relation']})" + ("?관찰용" if not c["eligible"] else ""))
        for edge in structure.edges:
            if edge["src"] == ann.sid:
                flags.append(f"{edge['kind']}→{edge['dst'] or 'START'}" +
                             (f"({edge['coarse_class']})" if edge["kind"] == "REL" else "") +
                             ("[문단간]" if edge["cross_paragraph"] else ""))
        focus = [f"{p['token_id']}:{p['form']}({p['function']})" for p in ann.focus_particles
                 if p["function"] != "TOPIC_CONTRAST"]
        if focus:
            flags.append("초점:" + ";".join(focus))
        if ann.polarity != "POS":
            flags.append("극성:" + ann.polarity)
        if ann.modality is not None:
            flags.append("양태:" + ann.modality)
        if ann.uncertain:
            flags.append("?" + "/".join(ann.uncertain))
        if not compact:
            # Full text is debugging only; compact flags are never truncated.
            from .rendering import middle_ellipsis
            flags.insert(0, middle_ellipsis(ann.text.replace("\n", " "), max_line_chars))
        lines.append(ann.sid + (" | " + " | ".join(flags) if flags else ""))
    return "\n".join(lines)
