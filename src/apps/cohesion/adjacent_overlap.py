import collections

pronounList = [
    "이",
    "그",
    "저",
    "이것",
    "그것",
    "저것",
    "무엇",
    "여기",
    "저기",
    "거기",
    "어디",
    "저희",
    "본인",
    "그대",
    "귀하",
    "너희",
    "당신",
    "여러분",
    "임자",
    "자기",
    "자네",
    "이런",
    "그들",
    "그녀",
    "당신",
    "저희",
    "놈",
    "얘",
    "걔",
    "쟤",
    "누구",
]


# Helper function to check content lemma
def is_content_lemma(pos_tag):
    return "NN" in pos_tag or "V" in pos_tag or "MA" in pos_tag


# Helper function to check function lemma
def is_function_lemma(pos_tag):
    return ("J" in pos_tag or "E" in pos_tag) and pos_tag not in {"MAJ", "SE"}


# Helper function to check noun lemma
def is_noun_lemma(pos_tag):
    return "N" in pos_tag and pos_tag != "ON"


# Helper function to check verb lemma
def is_verb_lemma(pos_tag):
    return "V" in pos_tag and pos_tag not in {"XPV", "XSV"}


# Helper function to check adjective lemma
def is_adjective_lemma(pos_tag):
    return "VXA" in pos_tag or "VA" in pos_tag


# Helper function to check adverb lemma
def is_adverb_lemma(pos_tag):
    return "MAG" in pos_tag or "MAJ" in pos_tag


# All lemmas
def adjacent_sentence_overlap_all_lemmas(now, target):
    lemma = set(item[1] for item in now)
    return sum(1 for item in target if item[1] in lemma)


def adjacent_sentence_overlap_all_lemmas_normed(now, target):
    lemma = set(item[1] for item in now)
    return 1 if any(item[1] in lemma for item in target) else 0


def binary_adjacent_sentence_overlap_all_lemmas(now, target):
    lemma = set(item[0] for item in now)
    return 1 if any(item[0] in lemma for item in target) else 0


def adjacent_two_sentence_overlap_all_lemmas(now, target1, target2):
    lemma = set(item[1] for item in now)
    overlap = {item[1] for item in target1 if item[1] in lemma}
    return sum(1 for item in target2 if item[1] in overlap)


def adjacent_two_sentence_overlap_all_lemmas_normed(now, target1, target2):
    lemma = set(item[1] for item in now)
    return (
        1
        if any(item[1] in lemma for item in target1)
        and any(item[1] in lemma for item in target2)
        else 0
    )


def binary_adjacent_two_sentence_overlap_all_lemmas(now, target1, target2):
    lemma = set(item[0] for item in now)
    return (
        1
        if any(item[0] in lemma for item in target1)
        and any(item[0] in lemma for item in target2)
        else 0
    )


# content lemmas
def adjacent_sentence_overlap_content_lemmas(now, target):
    lemma = set(item[1] for item in now if is_content_lemma(item[1]))
    return sum(1 for item in target if item[1] in lemma)


def adjacent_sentence_overlap_content_lemmas_normed(now, target):
    lemma = set(item[1] for item in now if is_content_lemma(item[1]))
    return 1 if any(item[1] in lemma for item in target) else 0


def binary_adjacent_sentence_overlap_content_lemmas(now, target):
    lemma = set(item[0] for item in now if is_content_lemma(item[1]))
    return 1 if any(item[0] in lemma for item in target) else 0


def adjacent_two_sentence_overlap_content_lemmas(now, target1, target2):
    lemma = set(item[1] for item in now if is_content_lemma(item[1]))
    overlap = {item[1] for item in target1 if item[1] in lemma}
    return sum(1 for item in target2 if item[1] in overlap)


def adjacent_two_sentence_overlap_content_lemmas_normed(now, target1, target2):
    lemma = set(item[1] for item in now if is_content_lemma(item[1]))
    return (
        1
        if any(item[1] in lemma for item in target1)
        and any(item[1] in lemma for item in target2)
        else 0
    )


def binary_adjacent_two_sentence_overlap_content_lemmas(now, target1, target2):
    lemma = set(item[0] for item in now if is_content_lemma(item[1]))
    return (
        1
        if any(item[0] in lemma for item in target1)
        and any(item[0] in lemma for item in target2)
        else 0
    )


# function lemmas
def adjacent_sentence_overlap_function_lemmas(now, target):
    lemma = set(item[1] for item in now if is_function_lemma(item[1]))
    return sum(1 for item in target if item[1] in lemma)


def adjacent_sentence_overlap_function_lemmas_normed(now, target):
    lemma = set(item[1] for item in now if is_function_lemma(item[1]))
    return 1 if any(item[1] in lemma for item in target) else 0


def binary_adjacent_sentence_overlap_function_lemmas(now, target):
    lemma = set(item[0] for item in now if is_function_lemma(item[1]))
    return 1 if any(item[0] in lemma for item in target) else 0


def adjacent_two_sentence_overlap_function_lemmas(now, target1, target2):
    lemma = set(item[1] for item in now if is_function_lemma(item[1]))
    overlap = {item[1] for item in target1 if item[1] in lemma}
    return sum(1 for item in target2 if item[1] in overlap)


def adjacent_two_sentence_overlap_function_lemmas_normed(now, target1, target2):
    lemma = set(item[1] for item in now if is_function_lemma(item[1]))
    return (
        1
        if any(item[1] in lemma for item in target1)
        and any(item[1] in lemma for item in target2)
        else 0
    )


def binary_adjacent_two_sentence_overlap_function_lemmas(now, target1, target2):
    lemma = set(item[0] for item in now if is_function_lemma(item[1]))
    return (
        1
        if any(item[0] in lemma for item in target1)
        and any(item[0] in lemma for item in target2)
        else 0
    )


# noun lemmas
def adjacent_sentence_overlap_noun_lemmas(now, target):
    lemma = set(item[1] for item in now if is_noun_lemma(item[1]))
    return sum(1 for item in target if item[1] in lemma)


def adjacent_sentence_overlap_noun_lemmas_normed(now, target):
    lemma = set(item[1] for item in now if is_noun_lemma(item[1]))
    return 1 if any(item[1] in lemma for item in target) else 0


def binary_adjacent_sentence_overlap_noun_lemmas(now, target):
    lemma = set(item[0] for item in now if is_noun_lemma(item[1]))
    return 1 if any(item[0] in lemma for item in target) else 0


def adjacent_two_sentence_overlap_noun_lemmas(now, target1, target2):
    lemma = set(item[1] for item in now if is_noun_lemma(item[1]))
    overlap = {item[1] for item in target1 if item[1] in lemma}
    return sum(1 for item in target2 if item[1] in overlap)


def adjacent_two_sentence_overlap_noun_lemmas_normed(now, target1, target2):
    lemma = set(item[1] for item in now if is_noun_lemma(item[1]))
    return (
        1
        if any(item[1] in lemma for item in target1)
        and any(item[1] in lemma for item in target2)
        else 0
    )


def binary_adjacent_two_sentence_overlap_noun_lemmas(now, target1, target2):
    lemma = set(item[0] for item in now if is_noun_lemma(item[1]))
    return (
        1
        if any(item[0] in lemma for item in target1)
        and any(item[0] in lemma for item in target2)
        else 0
    )


# verb lemmas
def adjacent_sentence_overlap_verb_lemmas(now, target):
    lemma = set(item[1] for item in now if is_verb_lemma(item[1]))
    return sum(1 for item in target if item[1] in lemma)


def adjacent_sentence_overlap_verb_lemmas_normed(now, target):
    lemma = set(item[1] for item in now if is_verb_lemma(item[1]))
    return 1 if any(item[1] in lemma for item in target) else 0


def binary_adjacent_sentence_overlap_verb_lemmas(now, target):
    lemma = set(item[0] for item in now if is_verb_lemma(item[1]))
    return 1 if any(item[0] in lemma for item in target) else 0


def adjacent_two_sentence_overlap_verb_lemmas(now, target1, target2):
    lemma = set(item[1] for item in now if is_verb_lemma(item[1]))
    overlap = {item[1] for item in target1 if item[1] in lemma}
    return sum(1 for item in target2 if item[1] in overlap)


def adjacent_two_sentence_overlap_verb_lemmas_normed(now, target1, target2):
    lemma = set(item[1] for item in now if is_verb_lemma(item[1]))
    return (
        1
        if any(item[1] in lemma for item in target1)
        and any(item[1] in lemma for item in target2)
        else 0
    )


def binary_adjacent_two_sentence_overlap_verb_lemmas(now, target1, target2):
    lemma = set(item[0] for item in now if is_verb_lemma(item[1]))
    return (
        1
        if any(item[0] in lemma for item in target1)
        and any(item[0] in lemma for item in target2)
        else 0
    )


# adjective lemmas
def adjacent_sentence_overlap_adjective_lemmas(now, target):
    lemma = set(item[1] for item in now if is_adjective_lemma(item[1]))
    return sum(1 for item in target if item[1] in lemma)


def adjacent_sentence_overlap_adjective_lemmas_normed(now, target):
    lemma = set(item[1] for item in now if is_adjective_lemma(item[1]))
    return 1 if any(item[1] in lemma for item in target) else 0


def binary_adjacent_sentence_overlap_adjective_lemmas(now, target):
    lemma = set(item[0] for item in now if is_adjective_lemma(item[1]))
    return 1 if any(item[0] in lemma for item in target) else 0


def adjacent_two_sentence_overlap_adjective_lemmas(now, target1, target2):
    lemma = set(item[1] for item in now if is_adjective_lemma(item[1]))
    overlap = {item[1] for item in target1 if item[1] in lemma}
    return sum(1 for item in target2 if item[1] in overlap)


def adjacent_two_sentence_overlap_adjective_lemmas_normed(now, target1, target2):
    lemma = set(item[1] for item in now if is_adjective_lemma(item[1]))
    return (
        1
        if any(item[1] in lemma for item in target1)
        and any(item[1] in lemma for item in target2)
        else 0
    )


def binary_adjacent_two_sentence_overlap_adjective_lemmas(now, target1, target2):
    lemma = set(item[0] for item in now if is_adjective_lemma(item[1]))
    return (
        1
        if any(item[0] in lemma for item in target1)
        and any(item[0] in lemma for item in target2)
        else 0
    )


# adverb lemmas
def adjacent_sentence_overlap_adverb_lemmas(now, target):
    lemma = set(item[1] for item in now if is_adverb_lemma(item[1]))
    return sum(1 for item in target if item[1] in lemma)


def adjacent_sentence_overlap_adverb_lemmas_normed(now, target):
    lemma = set(item[1] for item in now if is_adverb_lemma(item[1]))
    return 1 if any(item[1] in lemma for item in target) else 0


def binary_adjacent_sentence_overlap_adverb_lemmas(now, target):
    lemma = set(item[0] for item in now if is_adverb_lemma(item[1]))
    return 1 if any(item[0] in lemma for item in target) else 0


def adjacent_two_sentence_overlap_adverb_lemmas(now, target1, target2):
    lemma = set(item[1] for item in now if is_adverb_lemma(item[1]))
    overlap = {item[1] for item in target1 if item[1] in lemma}
    return sum(1 for item in target2 if item[1] in overlap)


def adjacent_two_sentence_overlap_adverb_lemmas_normed(now, target1, target2):
    lemma = set(item[1] for item in now if is_adverb_lemma(item[1]))
    return (
        1
        if any(item[1] in lemma for item in target1)
        and any(item[1] in lemma for item in target2)
        else 0
    )


def binary_adjacent_two_sentence_overlap_adverb_lemmas(now, target1, target2):
    lemma = set(item[0] for item in now if is_adverb_lemma(item[1]))
    return (
        1
        if any(item[0] in lemma for item in target1)
        and any(item[0] in lemma for item in target2)
        else 0
    )
