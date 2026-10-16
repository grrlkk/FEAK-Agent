import pytest

from staged_feedback.edits import Rejected, changed_char_ratio, quote_spans
from staged_feedback.models import Operation


def operation(kind, **kwargs):
    return Operation(id="op1", issue_ids=["i1"], op=kind, **kwargs)


def test_replace_normalizes_quotes_and_preserves_space(executor):
    text, record = executor.apply("나는 책을 읽을수 있다.", operation("replace", quote="읽을 수", replacement="읽을 수"))
    assert text == "나는 책을 읽을 수 있다."
    assert "바꿈" in record["description"]


@pytest.mark.parametrize("position,expected", [
    (dict(kind="before", anchor_quote="하나."), "셋. 하나. 둘."),
    (dict(kind="after", anchor_quote="하나."), "하나. 셋. 둘."),
    (dict(kind="paragraph_start", paragraph=1), "셋. 하나. 둘."),
    (dict(kind="paragraph_end", paragraph=1), "하나. 둘. 셋."),
])
def test_move(executor, position, expected):
    # Put the source in another paragraph for paragraph_end to be a real move.
    actual, record = executor.apply("하나. 둘.\n\n셋.", operation("move", quote="셋.", position=position))
    assert " ".join(actual.split()) == expected
    assert "3번 문장" in record["description"]


def test_move_forward_and_keep_anchor(executor):
    text, _ = executor.apply("하나. 둘. 셋.", operation("move", quote="하나.",
        position=dict(kind="after", anchor_quote="셋.")), keep=["셋."])
    assert " ".join(text.split()) == "둘. 셋. 하나."


def test_merge_retains_content_words(executor):
    result, _ = executor.apply("사과. 배.", operation("merge", sentence_a="사과.", sentence_b="배.", replacement="사과 배."))
    assert result.strip() == "사과 배."


@pytest.mark.parametrize("replacement,code", [("사과.", "merge_content_loss"), ("", "merge_shape")])
def test_merge_loss_rejected(executor, replacement, code):
    with pytest.raises(Rejected) as error:
        executor.apply("사과. 배.", operation("merge", sentence_a="사과.", sentence_b="배.", replacement=replacement))
    assert error.value.code == code


def test_merge_requires_bareun_analysis(executor):
    executor.pos = lambda _: []
    with pytest.raises(Rejected) as error:
        executor.apply("사과. 배.", operation("merge", sentence_a="사과.", sentence_b="배.", replacement="사과 배."))
    assert error.value.code == "merge_analysis_unavailable"


def test_split(executor):
    result, _ = executor.apply("사과 배.", operation("split", quote="사과 배.", sentences=["사과.", "배."]))
    assert result == "사과. 배."


def test_connect(executor):
    result, _ = executor.apply("나는 걷는다. 기분이 좋다.", operation("connect", sentence_a="나는 걷는다.",
        sentence_b="기분이 좋다.", connector="그래서"))
    assert result == "나는 걷는다. 그래서 기분이 좋다."


@pytest.mark.parametrize("op", [
    operation("replace", quote="사과", replacement="배"),
    operation("move", quote="사과.", position=dict(kind="after", anchor_quote="배.")),
    operation("merge", sentence_a="사과.", sentence_b="배.", replacement="사과 배."),
    operation("split", quote="사과.", sentences=["사과.", "사과."]),
    operation("connect", sentence_a="배.", sentence_b="사과.", connector="또한"),
])
def test_keep_list_rejects_all_operations(executor, op):
    with pytest.raises(Rejected) as error:
        executor.apply("배. 사과.", op, keep=["사과."])
    assert error.value.code == "preserve_violation"


@pytest.mark.parametrize("text,quote,count", [("사과. 사과.", "사과", 2), ("사과.", "배", 0)])
def test_nonunique_quotes(executor, text, quote, count):
    assert len(quote_spans(text, quote)) == count
    with pytest.raises(Rejected) as error:
        executor.apply(text, operation("replace", quote=quote, replacement="포도"))
    assert error.value.code == "quote_not_unique"


@pytest.mark.parametrize("replacement", ["", " ", "."])
def test_cannot_delete_student_sentence(executor, replacement):
    with pytest.raises(Rejected):
        executor.apply("사과.", operation("replace", quote="사과.", replacement=replacement))


def test_cannot_replace_multiple_sentences(executor):
    with pytest.raises(Rejected) as error:
        executor.apply("사과. 배.", operation("replace", quote="사과. 배.", replacement="포도."))
    assert error.value.code == "replace_scope"


@pytest.mark.parametrize("replacement", ["이름", "#@지역#", "#@이름# #@이름#", "__ANON_x_1__"])
def test_markers_cannot_change(executor, replacement):
    with pytest.raises(Rejected) as error:
        executor.apply("#@이름# 학생이다.", operation("replace", quote="#@이름#", replacement=replacement))
    assert error.value.code == "anonymization_violation"


def test_markers_preserved_on_move(executor):
    result, _ = executor.apply("#@이름# 학생이다. #@지역# 마을이다.", operation("move", quote="#@지역# 마을이다.",
        position=dict(kind="before", anchor_quote="#@이름# 학생이다.")))
    assert result.count("#@이름#") == result.count("#@지역#") == 1


def test_wrong_issue_rejected(executor):
    with pytest.raises(Rejected) as error:
        executor.apply("사과.", operation("replace", quote="사과", replacement="배"), allowed_issue_ids={"i2"})
    assert error.value.code == "outside_feedback"


def test_ratio_is_descriptive():
    assert changed_char_ratio("가나다", "가나다") == 0
    assert 0 < changed_char_ratio("가나다", "가 나다") <= 1
