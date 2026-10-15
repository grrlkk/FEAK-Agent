from verak.v5.check_local import retain_control_notices, NoticeEnvironment
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v4.render_v43 import assert_rendered


def test_notice_ablation_removes_all_cohesion_copies_but_retains_id_notices():
    split = {'kind': 'sentence_split', 'source_sid': 'S1', 'result_sids': ['S1','S1a']}
    change = {'type': 'conjunction_change', 'sid': 'S2'}
    payload = {'notices': [change, split], 'work_journal': [{'marker_change_notices': [change, split]}],
               'last_result': {'marker_change_notices': [change]}, 'paragraphs': [{'text': '원문'}]}
    result = retain_control_notices(payload)
    assert result['notices'] == [split]
    assert result['work_journal'][0]['marker_change_notices'] == [split]
    assert result['last_result']['marker_change_notices'] == []
    assert result['paragraphs'] == payload['paragraphs']
    assert payload['notices'] == [change, split]


def test_paired_observations_differ_only_in_new_notice_field(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **kw: {})
    messages = []
    for show in (False, True):
        env = NoticeEnvironment(row('첫 문장이다. 둘째 문장이다.'), FakeBoundaryAnalysis(), show_notices=show)
        env.delegate([{'location': ['S1'], 'action': 'EDIT', 'instruction': '검토', 'needs_search': 'no'}], 1)
        env.last_change_notices = [{'id': 'T1_N1', 'type': 'style_vs_majority', 'sid': 'S1'}]
        messages.append(env.public_messages('revision'))
    assert messages[0][0] == messages[1][0]
    a,b = [assert_rendered(m) for m in messages]
    assert not a['notices'] and len(b['notices']) == 1
    b['notices'] = []
    assert a == b
