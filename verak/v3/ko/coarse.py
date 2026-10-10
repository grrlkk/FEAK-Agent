"""Phase 2c relation observations. Eligibility is distinct from visibility."""

import re

from .annotation import initial_conjunction
from .patterns import after_focus
from .word_text import connectives

COARSE_EC = {
    "면": "CONDITION", "으면": "CONDITION", "거든": "CONDITION",
    "니까": "CAUSE", "으니까": "CAUSE", "므로": "CAUSE", "으므로": "CAUSE",
    "지만": "ADVERSATIVE", "더라도": "ADVERSATIVE",
    "어도": "ADVERSATIVE", "아도": "ADVERSATIVE", "여도": "ADVERSATIVE",
    "려고": "PURPOSE", "으려고": "PURPOSE", "고자": "PURPOSE",
}
CLOSED_CONJUNCTIONS = {
    **dict.fromkeys(("그러나", "하지만", "반면", "그런데"), "ADVERSATIVE"),
    **dict.fromkeys(("따라서", "그러므로", "그래서", "이 때문에", "이로 인해"), "RESULT"),
    **dict.fromkeys(("예를 들어", "예컨대"), "EXAMPLE"),
    **dict.fromkeys(("또한", "게다가", "더구나"), "ADDITION"),
    **dict.fromkeys(("즉", "다시 말해", "이처럼"), "RESTATEMENT"),
}


def coarse_connectives(sentence, lexicon):
    tokens = sentence.tokens
    observations = connectives(tokens, lexicon)
    result = []
    for conn in observations:
        i = int(conn.token_id[1:]) - 1
        relation = COARSE_EC.get(conn.form)
        reason = None
        if conn.classification == "AUX":
            reason = "auxiliary"
        # Fixed discourse expressions are not conditional clauses.
        prefix = "".join(t.form for t in tokens[max(0, i - 5):i + 1])
        if conn.form in {"면", "으면"} and re.search(
                r"(?:예를들|다시말하|바꾸어말하|바꿔말하|바꾸아말하)(?:으)?면$", prefix):
            reason = "fixed_expression"
        following = after_focus(tokens, i + 1)
        tail = tokens[following:]
        if conn.form == "거든" and not any(t.tag in {"VV", "VA", "VCP", "VCN", "XSV", "XSA"} for t in tail):
            reason = "conditional_use_unconfirmed"
        if conn.form in {"어도", "아도", "여도"} and tail and tail[0].form in {"되", "괜찮", "좋", "무방하"}:
            reason = "permission_not_concession"
        if reason:
            relation = None
        classification = "AUX" if conn.classification == "AUX" else "UNAMBIGUOUS" if relation else "AMBIGUOUS"
        result.append({"token_id": conn.token_id, "form": conn.form, "span": conn.span,
                       "kind": "EC", "coarse_class": relation,
                       "candidates": [relation] if relation else [], "classification": classification,
                       "eligible": relation is not None, "exclusion_reason": reason,
                       "level": "WORD"})
    # -기 때문에 is an ETN + bound noun + particle construction, never an EC tag.
    for i in range(len(tokens) - 2):
        a, b, c = tokens[i:i + 3]
        if (a.form, a.tag, b.form, b.tag, c.form, c.tag) == ("기", "ETN", "때문", "NNB", "에", "JKB"):
            result.append({"token_id": f"M{i + 1}:M{i + 3}", "form": "기 때문에", "span": [a.start, c.end],
                           "kind": "construction", "coarse_class": "CAUSE", "candidates": ["CAUSE"],
                           "classification": "UNAMBIGUOUS", "eligible": True,
                           "exclusion_reason": None, "level": "WORD"})
    return result


def coarse_conjunction(text, lexicon):
    value = initial_conjunction(text, {**lexicon, **CLOSED_CONJUNCTIONS})
    if value is None:
        return None
    coarse = CLOSED_CONJUNCTIONS.get(value.form)
    return {"form": value.form, "coarse_class": coarse, "eligible": coarse is not None,
            "visible_relation": coarse or value.relation, "level": "SENTENCE"}


def relation_swap_allowed(before, after):
    """Phase 3 contract only: never swap within a coarse relation class."""
    return bool(before.get("eligible") and after.get("eligible") and
                before.get("coarse_class") in set(COARSE_EC.values()) and
                after.get("coarse_class") in set(COARSE_EC.values()) and
                before["coarse_class"] != after["coarse_class"])
