from verak.v5.check_outputs import affected, structural_kind, prepare_case, summarize


def paras(ids):
    return [{'id': 'P1', 'sentences': [{'id': s, 'text': s+'.'} for s in ids]}]


def test_move_and_delete_cover_both_changed_boundaries():
    before = paras(['S1', 'S2', 'S3', 'S4'])
    after = paras(['S1', 'S3', 'S2', 'S4'])
    action = {'valid': True, 'action': 'MOVE'}
    assert affected(before, after, action) == ['S2', 'S3', 'S4']
    assert affected(before, paras(['S1', 'S3', 'S4']), action) == ['S3']
    assert structural_kind({'valid': False, 'action': 'MOVE'}, before, after) is None
    assert structural_kind({'valid': True, 'action': 'EDIT'}, before, before) is None
    assert structural_kind({'valid': True, 'action': 'EDIT'}, before, paras(['S1','S2','S3','S4','N1'])) == 'SPLIT'


def test_one_link_shared_by_two_actions_is_not_two_broken_links():
    case = prepare_case({'source': 'unit', 'case_id': 'one', 'original': paras(['S1','S2']),
        'final': paras(['S2','S1']), 'protocol': False, 'lineage': {},
        'actions': [{'id': 'E1', 'kind': 'MOVE', 'sites': ['S1']},
                    {'id': 'E2', 'kind': 'INSERT', 'sites': ['S1','deleted']} ]})
    assert len(case['notices']) == 1
    result = {'source': 'unit', 'case_id': 'one', 'verdicts': [
        {'id': 'N1', 'broken_after_revision': 'yes'}]}
    stats = summarize([case], [result])['unit']
    assert stats['broken_links'] == 1 and stats['structural_actions_with_broken_link'] == 2
    assert stats['essays_with_broken_link'] == 1 and stats['protocol_complete'] == 0
    missing = summarize([case], [])['unit']
    assert missing['unjudged_links'] == 1 and missing['judged_essays'] == 0
    assert missing['by_edit_type']['MOVE']['unjudged_actions'] == 1
