"""Conservative, Bareun-tag-aware main-predicate polarity and modality patterns."""

PREDICATE_TAGS = {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}


def after_focus(tokens, index):
    while index < len(tokens) and tokens[index].tag in {"JX", "JKS"} and tokens[index].form in {"은", "는", "도", "만", "이", "가"}:
        index += 1
    return index


def predicate_features(tokens):
    uncertain = []
    endings = [i for i, token in enumerate(tokens) if token.tag == "EF"]
    if not endings:
        return "unknown", None, ["polarity", "modality"]
    main_end = endings[-1]
    # The predicate must terminate in this final EF (EP may intervene), not an
    # embedded predicate somewhere earlier in a nominal or incomplete sentence.
    last = main_end - 1
    while last >= 0 and tokens[last].tag == "EP":
        last -= 1
    if last < 0 or tokens[last].tag not in PREDICATE_TAGS:
        return "unknown", None, ["polarity", "modality"]
    core = [t for t in tokens[:last + 1] if t.tag != "EP"]
    pairs = [(t.form, t.tag) for t in core]

    # Patterns end at the main predicate. They cannot match an embedded example.
    modality = None
    if len(core) >= 3 and core[-1].form in {"있", "없"} and core[-1].tag in {"VA", "VV", "VX"}:
        pos = len(core) - 2
        while pos >= 0 and core[pos].tag in {"JX", "JKS"} and core[pos].form in {"은", "는", "도", "만", "이", "가"}:
            pos -= 1
        if pos < 0:
            return "NEG" if core[-1].form == "없" else "POS", None, uncertain
        exclusive = core[pos].form == "밖에" and core[pos].tag == "JX"
        if exclusive:
            pos -= 1
        if pos >= 1 and pairs[pos] == ("수", "NNB") and core[pos - 1].tag == "ETM" and core[pos - 1].form in {"ㄹ", "을"}:
            if exclusive:
                if core[-1].form == "없":
                    modality = "NECESSITY"
                else:
                    uncertain.append("modality_ill_formed")
            else:
                double_neg = pos >= 3 and pairs[pos - 2] == ("않", "VX") and pairs[pos - 3] == ("지", "EC")
                modality = ("NECESSITY" if double_neg and core[-1].form == "없" else
                            "IMPOSSIBLE" if core[-1].form == "없" else "POSSIBLE")
    if len(core) >= 2 and core[-1].form in {"하", "되"} and core[-1].tag in {"VV", "VX"} and pairs[-2] in {
            ("아야", "EC"), ("어야", "EC"), ("여야", "EC")}:
        modality = "OBLIGATION"

    # Bound the main predicate chain at clause/nominal endings; EC + VX is an
    # auxiliary chain (e.g. -지 않다), not an independent preceding clause.
    start = 0
    for i, token in enumerate(core[:-1]):
        if token.tag in {"ETM", "ETN", "EF"}:
            start = i + 1
        elif token.tag == "EC":
            following_index = after_focus(core, i + 1)
            following = core[min(following_index, len(core) - 1)]
            auxiliary = following.tag == "VX" or (
                token.form in {"아야", "어야", "여야"} and following.form in {"하", "되"})
            if not auxiliary:
                start = i + 1
    main = core[start:]
    negative = (pairs[-1] in {("없", "VA"), ("아니", "VCN")} or
                any(t.tag == "MAG" and t.form in {"안", "못"} for t in main) or
                any(a.form == "지" and a.tag == "EC" and after_focus(main, i + 1) < len(main) and
                    main[after_focus(main, i + 1)].form in {"않", "못하"} and
                    main[after_focus(main, i + 1)].tag == "VX" for i, a in enumerate(main)))
    # Necessity from 수밖에 없다 / 지 않을 수 없다 is not simple negative polarity.
    if modality == "NECESSITY":
        negative = False
    return "NEG" if negative else "POS", modality, uncertain
