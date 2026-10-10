"""Conservative, cached-Bareun dependency screen for the G_DEL_LINK retry.

These are candidate hints, not semantic verdicts. Sol still checks real damage,
repair and recoverability on the exact deletion, with the unchanged QC prompt.
"""
import re

from .operators import label_targets

NOUN_TAGS = {'NNG', 'NNP', 'NNB', 'NP', 'NR'}
PUNCTUATION = {'SF', 'SP', 'SS', 'SSO', 'SSC', 'SE', 'SO', 'SW'}
ANAPHOR = re.compile(r'^(이것|그것|이는|이러한|이런|이처럼|그러한)')
ORDINAL = re.compile(r'첫째|첫\s*번째|다음으로|마지막으로')
LIST_NOUNS = {'이유', '방법', '문제', '장점', '단점', '특징', '종류', '나라',
              '국가', '사례', '단계', '측면', '방안', '효과', '변화', '활동',
              '역할', '차이', '원인', '결과', '내용', '요소', '가지'}
LIST_MARKERS = {'여러', '다양', '다음', '몇', '첫째', '둘째', '셋째', '두', '세',
                '네', '다섯', '둘', '셋', '넷', '하나', '1', '2', '3', '4', '5'}
QUESTION_ENDINGS = {'까', '니', '냐', '는가', '은가', '을까', 'ㄹ까', '습니까',
                    '는지', '은지', '을지', 'ㄹ지'}


def lexical(unit):
    return [t for t in unit.tokens if t.tag not in PUNCTUATION]


def question(unit):
    final = [t.form for t in unit.tokens if t.tag == 'EF']
    return '?' in unit.text[-4:] and bool(final) or bool(final and final[-1] in QUESTION_ENDINGS)


def evidence(source, sid):
    """Return auditable lexical evidence for the immediately following sentence."""
    pi, si, deleted = source.locate(sid)
    position = next(i for i, u in enumerate(source.units) if u.sid == sid)
    if position + 1 >= len(source.units):
        return None
    following = source.units[position + 1]
    npi, _, _ = source.locate(following.sid)
    tokens = lexical(following)
    if not tokens:
        return None
    # Surface anchoring prevents mid-sentence occurrences from passing. Bareun
    # offsets/tags distinguish the determiner 이 from unrelated syllable prefixes.
    text = following.text.lstrip(' \t\r\n\"\'“‘「『(')
    first = tokens[0]
    hints = []
    match = ANAPHOR.match(text)
    if match and first.tag in {'NP', 'MM', 'MMD', 'MMA', 'VA', 'VV', 'MAG'}:
        # The lexical form must actually overlap the anchored expression.
        if first.start < following.text.find(text) + match.end():
            hints.append({'kind': 'anaphoric_start', 'surface': match.group()})
    if len(tokens) >= 2 and first.form in {'그', '이'} and first.tag in {'MM', 'MMD'} and tokens[1].tag in NOUN_TAGS:
        prefix = following.text[first.start:tokens[1].end]
        if re.match(r'^(그|이)\s+\S', prefix) and following.text[:first.start].strip(' \t\"\'“‘「『(') == '':
            hints.append({'kind': 'demonstrative_noun_start', 'surface': prefix})
    conditional = next((s for s in ('그렇다면', '그러면') if text.startswith(s)), None)
    if conditional and first.form in {'그렇', '그러', conditional} and first.tag in {'VA', 'VV', 'MAJ', 'MAG'}:
        hints.append({'kind': 'conditional_start', 'surface': conditional})
    ordinal = ORDINAL.search(following.text)
    if ordinal:
        forms = {t.form for t in deleted.tokens}
        nouns = {t.form for t in deleted.tokens if t.tag in NOUN_TAGS}
        # A list marker alone does not establish that the deletion introduced it.
        # Require an explicit enumeration/variety cue and a listable noun, or a
        # question explicitly introducing why/how followed by a listed answer.
        list_intro = bool(nouns & LIST_NOUNS and forms & LIST_MARKERS)
        question_intro = question(deleted) and bool(forms & {'왜', '어떻게', '무엇', '어떤', '이유', '방법'})
        ordinal_tokens = [t for t in following.tokens if t.start < ordinal.end() and t.end > ordinal.start()]
        if (list_intro or question_intro) and ordinal_tokens:
            hints.append({'kind': 'introduced_ordinal', 'surface': ordinal.group(),
                          'introduction': 'explicit_list' if list_intro else 'list_question'})
    if npi == pi + 1 and si == len(source.paragraphs[pi].units) - 1 and question(deleted) and not question(following):
        # Bareun cannot certify answer semantics. Demand a shared substantive noun
        # between the question and the directly following paragraph, and retain
        # this only as a candidate for independent Sol damage/recovery QC.
        nouns = {t.form for t in deleted.tokens if t.tag in {'NNG', 'NNP'} and len(t.form) > 1}
        next_nouns = {t.form for u in source.paragraphs[npi].units for t in u.tokens
                      if t.tag in {'NNG', 'NNP'} and len(t.form) > 1}
        shared = sorted(nouns & next_nouns)
        if shared:
            hints.append({'kind': 'question_next_paragraph', 'shared_content_nouns': shared})
    if not hints:
        return None
    return {'sid': sid, 'next_sid': following.sid, 'next_paragraph': source.paragraphs[npi].pid,
            'next_sentence': following.text, 'hints': hints,
            'bareun_tokens': [{'form': t.form, 'tag': t.tag, 'start': t.start, 'end': t.end} for t in tokens],
            'semantic_status': 'candidate_hint; unchanged independent Sol QC is required'}


def candidates(source):
    return [item for sid in label_targets(source) if (item := evidence(source, sid)) is not None]
