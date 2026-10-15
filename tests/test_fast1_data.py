from collections import Counter

from verak.v5 import fast1_data as fast1


def sample_row():
    first = '첫 문장입니다. 둘째 문장입니다.'
    second = '다른 문단입니다.'
    text = first + '\n\n' + second
    return {
        'source_id': 'train:1', 'text': text,
        'paragraphs': [
            {'id': 'P1', 'sentences': [{'id': 'S1', 'text': '첫 문장입니다.'},
                                      {'id': 'S2', 'text': '둘째 문장입니다.'}]},
            {'id': 'P2', 'sentences': [{'id': 'S3', 'text': second}]},
        ],
        'profile': {'sentences': [
            {'id': '1.1', 'paragraph': 1, 'start': 0, 'end': 8, 'tokens': []},
            {'id': '1.2', 'paragraph': 1, 'start': 9, 'end': 18, 'tokens': []},
            {'id': '2.1', 'paragraph': 2, 'start': 20, 'end': len(text), 'tokens': []},
        ]},
    }


def test_paragraph_replacement_uses_exact_source_spans():
    row = sample_row()
    assert fast1._paragraph_source_text(row, 'P1') == '첫 문장입니다. 둘째 문장입니다.'
    changed = fast1._replace_paragraph_span(row, 'P1', '첫 문단 수정입니다.')
    assert changed == '첫 문단 수정입니다.\n\n다른 문단입니다.'


def test_paragraph_assembly_changes_only_planned_paragraphs():
    row = sample_row()
    changed = fast1._assemble_changed_paragraphs(row, {'P2': '둘째 문단 수정입니다.'})
    assert changed == '첫 문장입니다. 둘째 문장입니다.\n\n둘째 문단 수정입니다.'


def test_noncontiguous_paragraph_ids_map_to_profile_offsets():
    row=sample_row()
    row['paragraphs'][1]['id']='P3'
    row['profile']['sentences'][2]['paragraph']=3
    assert fast1._paragraph_source_text(row,'P3')=='다른 문단입니다.'


def test_adjacent_word_filter_detects_only_new_repeats(monkeypatch):
    row = sample_row()
    monkeypatch.setattr(fast1, '_new_nnp', lambda original, candidate, row: (Counter(), Counter(), None))
    row['text'] = '이 글은 매우 매우 중요합니다.'
    row['profile']['sentences'] = []
    existing = fast1.machine_filter(row, row['text'])
    assert existing['pass']
    new_repeat = fast1.machine_filter(row, '이 글은 정말 정말 중요합니다.')
    assert not new_repeat['pass']
    assert new_repeat['immediate_repeats'] == ['정말 정말']


def test_number_filter_detects_added_number_only(monkeypatch):
    row = sample_row()
    monkeypatch.setattr(fast1, '_new_nnp', lambda original, candidate, row: (Counter(), Counter(), None))
    assert fast1.machine_filter(row, row['text'])['pass']
    result = fast1.machine_filter(row, row['text'] + ' 12개가 늘었습니다.')
    assert not result['pass']
    assert result['new_numbers']['12'] == 1


def test_b1_samples_are_eight_and_balanced_with_failed_case_per_group():
    rows = []
    for weakness in fast1.WEAKNESSES:
        for index in range(4):
            rows.append({'source_id': f'{weakness}:{index}', 'weakness': weakness,
                'machine_filter': {'pass': index != 0}, 'pass': index != 0,
                'question': '문항', 'paragraph_index': 1, 'paragraph_count': 2})
    sample = fast1._b1_samples(rows)
    assert len(sample) == 8
    assert all(sum(x['weakness'] == weakness for x in sample) == 2 for weakness in fast1.WEAKNESSES)
    assert all(any(x['weakness'] == weakness and not x['pass'] for x in sample)
               for weakness in fast1.WEAKNESSES)


def test_q_is_eight_rubric_expected_mean():
    assert fast1._q({'mean': 4.25, 'expected': [4.25]*8}) == 4.25


def test_retry_policy_only_known_terminal_failures():
    assert fast1._retry_is_auditable({'error_type':'ConnectError'})
    assert fast1._retry_is_auditable({'error_type':'HTTPStatusError','http_status':400})
    assert fast1._retry_is_auditable({'status':'incomplete','http_status':200})
    assert not fast1._retry_is_auditable({'error_type':'HTTPStatusError','http_status':408})
    assert not fast1._retry_is_auditable({'status':'pending'})


def test_duplicate_planner_targets_coalesce_to_one_rewrite_item():
    plan = fast1._merge_plan_items([
        {'paragraph_id': 'P1', 'problem': '주장 모호함', 'goal': '주장을 분명히'},
        {'paragraph_id': 'P1', 'problem': '설명 부족', 'goal': '근거 설명을 보완'},
        {'paragraph_id': 'P2', 'problem': '연결 약함', 'goal': '앞뒤 연결을 명확히'},
    ])
    assert [item['paragraph_id'] for item in plan['items']] == ['P1', 'P2']
    assert '설명 부족' in plan['items'][0]['problem']


def test_b2_paragraph_sample_uses_full_same_target_text_for_both_efforts():
    row=sample_row()
    row.update({'original':row['text'],'plan':{'items':[
        {'paragraph_id':'P1','problem':'연결 약함','goal':'연결을 분명히'}]}})
    none={'paragraph_replacements':{'P1':'none 수정 문단 전체입니다.'}}
    low={'paragraph_replacements':{'P1':'low 수정 문단 전체입니다.'}}
    pair=fast1._b2_paragraph_sample(row,[none,low],'fixed-pair')
    assert pair['plan']['paragraph_id']=='P1'
    assert pair['original_paragraph']=='첫 문장입니다. 둘째 문장입니다.'
    assert pair['rewrites']==['none 수정 문단 전체입니다.','low 수정 문단 전체입니다.']


def test_notice_formatter_removes_sids_and_python_enum_repr():
    notice={'type':'ec_relation_change','before':[],'after':[['지만','ADVERSATIVE','S7']],
            'message':'S8 연결어미 관계 후보: [] → [(\'지만\', \'ADVERSATIVE\')]'}
    rendered=fast1._notice_ko(notice)
    assert rendered=='연결어미 없음 → 지만(대조)'
    assert 'S7' not in rendered and 'ADVERSATIVE' not in rendered
