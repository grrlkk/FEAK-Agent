import pytest

from verak.v4.feedback import validate
from verak.v4.data import feedback_parts, normalized_source


def row():
    return {'source_id':'train:1','paragraphs':[{'id':'P1','sentences':[{'id':'S1','text':'예문.'}]}]}


def test_feedback_rejects_invented_location_and_duplicate_items():
    item={'rubric':'grammar','problem':'종결체 혼용','location':['S99'],'owner':'korean','needs_search':'no'}
    with pytest.raises(ValueError,match='Ungrounded'):
        validate({'items':[item]},row())
    item['location']=['S1']
    with pytest.raises(ValueError,match='Duplicate'):
        validate({'items':[dict(item),dict(item)]},row())


def test_feedback_keeps_original_rubric_text_and_excludes_score_line():
    text='2 3 4 5 6 7 8 9\n\n### Feedback:\n- 어법의 적절성:\n 문체를 통일하세요.\n- 글의 통일성:\n 일관됩니다.'
    result=feedback_parts(text)
    assert result==[{'rubric_native':'어법의 적절성','text':'문체를 통일하세요.'},
                    {'rubric_native':'글의 통일성','text':'일관됩니다.'}]
    assert normalized_source('같은  글.\n다음 문장.')==normalized_source('같은 글. 다음 문장.')


def test_feedback_ids_are_source_specific():
    item={'rubric':'task','problem':'설명이 부족함','location':['essay'],'owner':'revision','needs_search':'no'}
    assert validate({'items':[item]},row())['items'][0]['item_id']=='train:1:I1'
