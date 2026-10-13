import json

import pytest

from verak.v5.fast1_common import (make_pair_messages, normalize_whitespace,
                                   validate_fast1_judgment)


def test_whitespace_normalization_changes_whitespace_only():
    assert normalize_whitespace('  가\n\t나  다 ') == '가 나 다'
    assert normalize_whitespace('가, 나!') == '가, 나!'


def test_pair_orders_are_separate_and_mirror_left_right_without_condition_names():
    case = {'case_id': 'essay:1', 'question': '질문', 'original': '원문.',
            'variants': {'new_condition': '수정.', 'saved_control': '대조.'}}
    primary = make_pair_messages(case, anchor='new_condition', baselines=('saved_control',), order='primary')
    reverse = make_pair_messages(case, anchor='new_condition', baselines=('saved_control',), order='reverse')
    p = json.loads(primary['messages'][1]['content'])
    r = json.loads(reverse['messages'][1]['content'])
    assert p['pairwise_comparisons'][0]['left'] == r['pairwise_comparisons'][0]['right']
    assert p['pairwise_comparisons'][0]['right'] == r['pairwise_comparisons'][0]['left']
    content = json.dumps(primary['messages'], ensure_ascii=False)
    assert 'new_condition' not in content
    assert 'saved_control' not in content
    assert primary['id'] != reverse['id']


def test_identical_original_gets_no_absolute_judgment_and_identical_pair_is_omitted():
    case = {'case_id': 'essay:2', 'question': '질문', 'original': '원문  내용',
            'variants': {'anchor': '원문 내용', 'same': '원문\n내용'}}
    request = make_pair_messages(case, anchor='anchor', baselines=('same',), order='primary')
    payload = json.loads(request['messages'][1]['content'])
    assert request['identical_conditions'] == ['anchor', 'same']
    assert request['expected_absolute_ids'] == []
    assert request['expected_pair_ids'] == []
    assert payload['pairwise_comparisons'] == []


def test_judgment_schema_requires_exact_ids_and_labels():
    value = {'absolute': [{'id': 'A', 'improved': 'yes', 'meaning_preserved': 'yes',
                           'fabrication': 'absent', 'reason': '근거'}],
             'pairs': [{'id': 'P1', 'preference': 'left', 'reason': '근거'}]}
    assert validate_fast1_judgment(json.dumps(value), {'A'}, {'P1'}) == value
    value['pairs'][0]['preference'] = 'system_name'
    with pytest.raises(ValueError):
        validate_fast1_judgment(json.dumps(value), {'A'}, {'P1'})


def test_frozen_judge_requests_reject_changed_text_under_same_case_id(tmp_path):
    from verak.v5.fast1_common import prepare_judge_manifest
    case={'case_id':'fixed','question':'질문','original':'처음 글',
          'variants':{'anchor':'고친 글','baseline':'비교 글'}}
    prepare_judge_manifest(tmp_path,[case],anchor='anchor',baselines=('baseline',))
    case['variants']['anchor']='나중에 바뀐 글'
    with pytest.raises(ValueError,match='essay text'):
        prepare_judge_manifest(tmp_path,[case],anchor='anchor',baselines=('baseline',))
