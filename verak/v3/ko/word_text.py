"""WORD observations and TEXT register from Bareun endings, without generation."""

from .patterns import after_focus
from .types import ConnInfo

STYLES = {"해라체": "한다", "하십시오체": "합니다", "해요체": "해요", "해체": "해"}
RELATIONS = {"원인": "CAUSE", "순서": "SEQUENCE", "나열": "ADDITION", "대조": "CONTRAST",
             "양보": "CONCESSION", "조건": "CONDITION", "발견": "DISCOVERY",
             "배경": "BACKGROUND", "목적": "PURPOSE", "동시": "SIMULTANEOUS"}
FOCUS = {"은": "TOPIC_CONTRAST", "는": "TOPIC_CONTRAST", "도": "ADDITIVE", "만": "ONLY",
         "조차": "SCALAR", "마저": "SCALAR", "까지": "SCALAR", "밖에": "EXCLUSIVE",
         "라도": "ALTERNATIVE", "이나": "ALTERNATIVE", "나": "ALTERNATIVE", "뿐": "ONLY"}
# Sampling flags only: no extra style classification from the essay's majority.
QUESTION_COLLOQUIAL = set("ㄴ가 는가 은가 ㄹ까 을까 니 나 지 거든 잖아 습니까 ㅂ니까 나요 까요 ㄹ까요 을까요 아 어 야 죠 지요 네 구나 느냐 냐 ㄹ게 을게 는지 ㄴ지 은지 ㄹ지 을지 더라 더라고 래 대 ㄴ대".split())


def connectives(tokens, lexicon):
    values = []
    for i, token in enumerate(tokens):
        if token.tag != "EC":
            continue
        following = after_focus(tokens, i + 1)
        auxiliary = following < len(tokens) and tokens[following].tag == "VX"
        relations = [] if auxiliary else [RELATIONS[label] for label in lexicon.get(token.form, [])]
        classification = "AUX" if auxiliary else "UNAMBIGUOUS" if len(relations) == 1 else "AMBIGUOUS"
        values.append(ConnInfo(f"M{i + 1}", token.form, [token.start, token.end], relations, classification))
    return values


def ending_styles(tokens, lexicon):
    values = []
    for i, token in enumerate(tokens):
        if token.tag != "EF":
            continue
        candidates = [STYLES[label] for label in lexicon.get(token.form, [])]
        # Bareun commonly separates '요/JX' from an interrogative/colloquial EF.
        polite = i + 1 < len(tokens) and tokens[i + 1].form == "요" and tokens[i + 1].tag == "JX"
        if polite:
            candidates = ["해요"]
        style = candidates[0] if len(set(candidates)) == 1 else "mixed" if candidates else "unknown"
        values.append({"token_id": f"M{i + 1}", "form": token.form, "span": [token.start, token.end],
                       "style": style, "polite_particle": polite,
                       "interrogative_or_colloquial": token.form in QUESTION_COLLOQUIAL,
                       "level": "TEXT"})
    styles = {v["style"] for v in values}
    style = next(iter(styles)) if len(styles) == 1 else "mixed" if styles else "unknown"
    return style, values


def focus_particles(tokens):
    return [{"token_id": f"M{i + 1}", "form": token.form, "span": [token.start, token.end],
             "function": FOCUS[token.form], "level": "WORD"}
            for i, token in enumerate(tokens) if token.tag == "JX" and token.form in FOCUS]
