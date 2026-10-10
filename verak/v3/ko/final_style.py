"""Last outside-quotation clause-final EF, independent of multi-unit flags."""

from .word_text import ending_styles

QUOTATIVES = {"고", "라고", "냐고", "자고"}
OPEN_CLOSE = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "『": "』"}


def quote_spans(text, offset=0):
    stack, spans = [], []
    for i, char in enumerate(text):
        if char == "'" and i and i + 1 < len(text) and text[i - 1].isascii() and text[i - 1].isalnum() and text[i + 1].isascii() and text[i + 1].isalnum():
            continue
        if stack and char == stack[-1][1]:
            start, _ = stack.pop()
            spans.append((offset + start, offset + i + 1))
        elif char in OPEN_CLOSE:
            stack.append((i, OPEN_CLOSE[char]))
    spans.extend((offset + start, offset + len(text)) for start, _ in stack)
    return spans


def final_style(sentence, lexicon, previous_style=None, quotation_spans=None):
    tokens = sentence.tokens
    spans = quotation_spans if quotation_spans is not None else quote_spans(sentence.text, sentence.start)
    _, endings = ending_styles(tokens, lexicon)
    eligible = []
    for value in endings:
        i = int(value["token_id"][1:]) - 1
        quoted = any(a <= tokens[i].start < b for a, b in spans)
        following = i + 1
        if following < len(tokens) and tokens[following].tag == "JX" and tokens[following].form == "요":
            following += 1
        quotative = following < len(tokens) and tokens[following].form in QUOTATIVES
        value.update(inside_quotation=quoted, followed_by_quotative=quotative,
                     selected=False, embedded=False)
        if not quoted and not quotative:
            eligible.append((i, value))
    selected = None
    if eligible:
        i, candidate = eligible[-1]
        # A nominalized/quoted EF followed by lexical material is not a final EF.
        tail = [t for t in tokens[i + 1:] if t.tag not in {"SF", "SP", "SS", "SE", "SO", "SW"}
                and not (t.tag == "JX" and t.form == "요")]
        if not tail:
            selected = candidate
    for value in endings:
        value["selected"] = value is selected
        value["embedded"] = value is not selected
    if selected is None:
        return "unknown", None, endings
    if selected["form"] in {"ㄹ까", "을까", "니"} and not selected["polite_particle"]:
        selected["style"] = previous_style if previous_style in {"해", "한다"} else "unknown"
        selected["context_style"] = previous_style
    style = selected["style"]
    return style if style != "mixed" else "unknown", selected, endings
