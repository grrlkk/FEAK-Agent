"""Retry-specific dependency, source-denominator, cache and isolation contracts."""
from copy import deepcopy

import pytest

from verak.src.schemas import Token
from verak.v3.common import write_json
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v3.v2_ops.config import config_for, operators
from verak.v3.v2_ops.dependency import candidates, evidence
from verak.v3.v2_ops.qc import summary
from verak.v3.v2_ops.retry import config_for_retry, qc_identity


def unit(sid, text, tokens):
    return Unit(sid, text, [Token(form, tag, start, end) for form, tag, start, end in tokens])


def document(first, second, *, separate=False):
    if separate:
        return Document([Paragraph('P1', [first]), Paragraph('P2', [second])], ['', '\n\n'])
    return Document([Paragraph('P1', [first, second])], [''])


@pytest.mark.parametrize('text,tokens,kind', [
    ('이것은 중요하다.', [('이것', 'NP', 0, 2), ('은', 'JX', 2, 3)], 'anaphoric_start'),
    ('이러한 결과가 있다.', [('이러한', 'MMD', 0, 3), ('결과', 'NNG', 4, 6)], 'anaphoric_start'),
    ('그 이유는 분명하다.', [('그', 'MMD', 0, 1), ('이유', 'NNG', 2, 4)], 'demonstrative_noun_start'),
    ('그러면 어떻게 될까?', [('그러', 'VV', 0, 2), ('면', 'EC', 2, 3)], 'conditional_start'),
])
def test_immediate_successor_bareun_anaphors(text, tokens, kind):
    first = unit('S1', '앞 문장이다.', [('문장', 'NNG', 2, 4)])
    second = unit('S2', text, tokens)
    doc = document(first, second)
    assert kind in {h['kind'] for h in evidence(doc, 'S1')['hints']}
    assert [p['sid'] for p in candidates(doc)] == ['S1']
    assert evidence(doc, 'S2') is None


def test_mid_sentence_or_unrelated_prefix_never_passes():
    first = unit('S1', '앞 문장이다.', [('문장', 'NNG', 2, 4)])
    for second in [unit('S2', '오늘 이것을 했다.', [('오늘', 'NNG', 0, 2), ('이것', 'NP', 3, 5)]),
                   unit('S2', '이 그림이 있다.', [('이', 'NNG', 0, 1), ('그림', 'NNG', 2, 4)])]:
        assert evidence(document(first, second), 'S1') is None


def test_ordinal_requires_deleted_sentence_to_introduce_list():
    second = unit('S2', '첫째, 시간을 절약한다.', [('첫째', 'NR', 0, 2), ('시간', 'NNG', 4, 6)])
    generic = unit('S1', '우리는 독서를 한다.', [('우리', 'NP', 0, 2), ('독서', 'NNG', 4, 6)])
    assert evidence(document(generic, second), 'S1') is None
    intro = unit('S1', '독서에는 여러 장점이 있다.', [('독서', 'NNG', 0, 2), ('여러', 'MMA', 5, 7), ('장점', 'NNG', 8, 10)])
    assert evidence(document(intro, second), 'S1')['hints'][0]['kind'] == 'introduced_ordinal'


def test_question_requires_immediate_next_paragraph_with_shared_content():
    first = unit('S1', '독서는 왜 중요할까?', [('독서', 'NNG', 0, 2), ('왜', 'MAG', 4, 5), ('까', 'EF', 10, 11)])
    second = unit('S2', '독서는 사고력을 높인다.', [('독서', 'NNG', 0, 2), ('사고력', 'NNG', 4, 7), ('다', 'EF', 11, 12)])
    assert evidence(document(first, second), 'S1') is None
    assert evidence(document(first, second, separate=True), 'S1')['hints'][0]['kind'] == 'question_next_paragraph'
    unrelated = unit('S2', '운동은 건강에 좋다.', [('운동', 'NNG', 0, 2), ('건강', 'NNG', 4, 6), ('다', 'EF', 9, 10)])
    assert evidence(document(first, unrelated, separate=True), 'S1') is None


def test_retry_config_isolated_and_shortfall_stays_in_requested_denominator(tmp_path):
    original = config_for(version='v2')
    retry = config_for_retry()
    assert operators(original) == ('G_DEL_LINK', 'L_FUSE')
    assert operators(retry) == ('G_DEL_LINK',)
    assert retry['paths']['v2_ops_output'] != original['paths']['v2_ops_output']
    assert retry['v2_retry']['gpu_access'] is False
    config = {'paths': {'v2_ops_output': tmp_path}, 'v2_ops': {'operators': ['G_DEL_LINK'], 'qc_min_yield': .3}}
    write_json(tmp_path / 'source_plan.json', {'plans': {'G_DEL_LINK': {'agent_train': [], 'agent_dev': []}},
                                             'requested_source_counts': {'agent_train': 300, 'agent_dev': 80}})
    result = summary(config)['operators']['G_DEL_LINK']
    assert result['counts']['planned'] == 380 and result['counts']['source_shortfall'] == 380
    assert result['decision'] == 'drop' and result['yield_upper_bound'] == 0


def test_qc_reuse_identity_ignores_only_identifiers_and_private_dependency():
    row = {'operator': 'G_DEL_LINK', 'question': '문항', 'source_text': '원문', 'corrupted_text': '손상',
           'episode_id': 'v2:old', 'records': [{'original_text': {'S1': '삭제'}, 'params': {'deleted_label': 'topic'}}]}
    revised = deepcopy(row)
    revised.update(episode_id='v2retry:new', dependency_screen={'sid': 'S1'})
    assert qc_identity(row) == qc_identity(revised)
    revised['corrupted_text'] = '다른 손상'
    assert qc_identity(row) != qc_identity(revised)


def test_retry_analysis_yields_only_to_a_inference(tmp_path):
    from verak.v3.v2_ops.retry_resources import priority_active
    config = {'paths': {'repo': tmp_path}}
    path = tmp_path / 'verak/v3/outputs/phase8_rft1/status.json'
    for stage in ('rollouts', 'evaluating_rft1', 'evaluating_oneshot'):
        write_json(path, {'stage': stage})
        assert priority_active(config)
    for stage in ('training_global', 'training_korean', 'training_oneshot', 'selecting', 'a_complete', 'complete'):
        write_json(path, {'stage': stage})
        assert not priority_active(config)
