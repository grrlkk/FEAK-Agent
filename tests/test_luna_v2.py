import json
from types import SimpleNamespace

from verak.v5.luna_v2_env import alignment, Environment, PHI
from verak.v5.luna_v2_run import korean_prompt, p1, render
from verak.v3.corrupt.document import Document, Paragraph, Unit


def doc(texts):
    return Document([Paragraph('P1', [Unit(f'S{i+1}', x, [], '' if i == 0 else ' ') for i,x in enumerate(texts)])], [''])


def test_moves_not_mislabeled_as_rewrites():
    a = alignment(['가나다.', '라마바.', '사아자.'], ['사아자.', '가나다.', '라마바.'])
    assert a['changed_sentence_ratio'] == 0
    assert sorted(b['old'][0] for b in a['blocks']) == [0,1,2]


def test_split_and_merge_are_explicit():
    a = alignment(['비가 왔지만 나는 밖에 나가서 산책을 했다.'], ['비가 왔다.', '하지만 나는 밖에 나가서 산책을 했다.'])
    assert any(len(b['old']) == 1 and len(b['new']) == 2 for b in a['blocks'])
    b = alignment(['비가 왔다.', '나는 밖에 나가서 산책을 했다.'], ['비가 왔지만 나는 밖에 나가서 산책을 했다.'])
    assert any(len(x['old']) == 2 and len(x['new']) == 1 for x in b['blocks'])


def test_c_removes_only_notice_information():
    d = doc(['철수는 집에 갔다.', '그래서 쉬었다.'])
    plan = {'style': '평서체', 'referents': ['철수']}
    notices = [{'message': '이전 문장: 영희는 학교에 갔다.'}]
    b, c = [korean_prompt(d, 'S2', plan, notices, arm) for arm in ['B','C']]
    assert '영희' not in c and '영희' in b
    assert '[현재 문단]' in c and '[바뀐 점]' not in c
    assert b.replace(' 글의 다른 부분이 고쳐지면서 [문장]과 앞뒤 문장의 연결이 바뀌었을 수 있습니다.', '').replace('[바뀐 점]\n- '+notices[0]['message']+'\n\n', '') == c


def test_policy_never_receives_feedback():
    row = {'question': '문항', 'text': '글', 'feedback': 'SECRET_FEEDBACK'}
    scores = {'genre': '설명', 'expected': [5.]*8}
    for arm in ['A', 'A+']:
        prompt = p1(row, scores, arm)
        assert 'SECRET_FEEDBACK' not in prompt and '{글.' not in prompt
    assert '연결어미와 접속어' in p1(row, scores, 'A+')


def test_render_does_not_interpolate_student_placeholders():
    value = render('P1', {'{문항}': '{만점}', '{만점}': '9'})
    assert '"{만점}"' in value


def test_new_numbers_and_masks_checked_without_mutating_source():
    original = doc(['#@이름#은 친구다.'])
    candidate = doc(['철수는 30% 성공했다.'])
    reasons, _ = Environment.safety(original, candidate, 1.)
    assert {r[0] for r in reasons} == {'anonymization', 'new_number'}
    assert original.text == '#@이름#은 친구다.'


def test_quoted_politeness_not_inferred_from_surface_regex():
    # Native final_style is responsible for quotation handling; the environment
    # must use its unknown classification rather than scan the raw ending.
    a = SimpleNamespace(sid='S1', predecessor_id=None, text='“안녕하세요.”', connectives=[],
                        initial_conj=None, subject_omitted=False, style='unknown', final_ending=None)
    fake = SimpleNamespace(structure=lambda: SimpleNamespace(annotations=[a]))
    assert Environment.notices(fake, fake, {'S1':['S1']}, {'S1'}, '평서체') == []


def test_edit_denominator_does_not_count_shifted_sentences_as_moves():
    from verak.v5.luna_v2_eval import edit_cases
    cases, _ = edit_cases('철수는 집에 갔다. 영희는 학교에 갔다.', '날이 밝았다. 철수는 집에 갔다. 영희는 학교에 갔다.')
    assert [x['kind'] for x in cases] == ['INSERT']


def test_paired_zero_and_effect_direction():
    from verak.v5.luna_v2_eval import paired
    assert paired({'x':1},{'x':1})['rank_biserial'] == 0
    assert paired({'x':1,'y':2},{'x':0,'y':0})['rank_biserial'] == 1


def test_batch_reserves_before_dispatch_and_shares_total_cap(tmp_path):
    from verak.v5.luna_v2_batch import Batch
    from verak.v5.api import BudgetStop
    import pytest
    contract = {'user_approved':True,'total_cap_usd':.00001,
                'luna':{'model':'gpt-6-luna','cap_usd':1.,'input_per_million':.1,'output_per_million':.5,'reasoning_effort':'none','temperature':0}}
    (tmp_path/'api_contract.json').write_text(json.dumps(contract))
    client = Batch('luna',root=tmp_path)
    with pytest.raises(BudgetStop): client.send('test',[{'custom_id':'a','prompt':'한글','max_output':100}])
    with client.db() as db: assert db.execute('select count(*) from calls').fetchone()[0] == 0
    client.close()
