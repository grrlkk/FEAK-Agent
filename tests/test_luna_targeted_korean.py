import json

import pytest

from verak.v5 import luna_targeted_korean as k


def test_moved_nearly_unchanged_sentence_uses_its_original_predecessor():
    old = '비가 내렸다. 그래서 나는 집에 있었다.\n\n친구가 도착했다. 우리는 책을 읽었다.'
    new = '친구가 도착했다. 그래서 나는 집에 있었다.\n\n비가 내렸다. 우리는 책을 읽었다.'
    items = k.select('train:fake',old,new)['selected']
    target = next(x for x in items if x['current_sentence'].startswith('그래서'))
    assert target['old_previous'] == '비가 내렸다.'
    assert target['current_previous'] == '친구가 도착했다.'
    assert target['similarity'] == 1


def test_spelling_only_predecessor_changes_are_included():
    old = '밥을 먹 었다. 나는 집에 갔다.'
    new = '밥을 먹었다. 나는 집에 갔다.'
    target = k.select('train:fake',old,new)['selected']
    assert len(target)==1
    assert target[0]['current_sentence']=='나는 집에 갔다.'


def test_threshold_and_duplicate_matching_never_reuse_a_source_sentence():
    old = k.sentences('비가 왔다. 비가 왔다. 친구가 왔다.')
    new = k.sentences('비가 왔다. 비가 왔다. 비가 왔다. 완전히 다른 이야기다.')
    matches = k.match_sentences(old,new)
    assert len(matches)==2
    assert len({m['old_index'] for m in matches})==2
    assert all(m['similarity']>=.9 for m in matches)


def test_no_change_selects_nothing_and_start_boundary_is_explicit():
    text = '친구가 왔다. 그래서 우리는 놀았다.'
    assert not k.select('train:fake',text,text)['selected']
    target = k.select('train:fake',text,'그래서 우리는 놀았다.')['selected'][0]
    assert target['current_previous']==''


def test_prompt_has_only_four_fields_and_fixed_question():
    case = {'old_previous':'이전 앞.', 'old_sentence':'이전 문장.',
            'current_previous':'현재 앞.', 'current_sentence':'현재 문장.',
            'feedback':'PRIVATE', 'source_id':'HIDDEN'}
    observation = k.messages(case)
    assert 'PRIVATE' not in json.dumps(observation)
    assert 'HIDDEN' not in json.dumps(observation)
    assert observation[1]['content'].endswith(k.QUESTION)
    assert k.parse('유지','문장.')['sentence']=='문장.'
    assert k.parse('고침: 문장.','문장.')['action']=='유지'
    assert not k.parse('다시 쓰겠습니다.','문장.')['format_valid']


def test_preparation_uses_saved_intermediate_and_checks_source_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(k,'SIMPLE',tmp_path/'simple')
    monkeypatch.setattr(k,'ROOT',tmp_path/'root')
    monkeypatch.setattr(k,'OUT',tmp_path/'probe')
    monkeypatch.setattr(k,'SOURCE_IDS',['train:fake'])
    path=k.SIMPLE/'episodes/three_agent/train_fake/result.json'
    row={'source_id':'train:fake', 'original':'비가 내렸다. 그래서 나는 집에 있었다.',
         'revision':'친구가 도착했다. 그래서 나는 집에 있었다.',
         'final':'사용해서는 안 되는 이전 한국어 결과.'}
    k.write_json(path,row)
    k.write_json(k.ROOT/'api_contract.json',{'luna':{'model':'gpt-6-luna','reasoning_effort':'none','temperature':0},'total_cap_usd':5})
    selected=k.prepare()
    assert len(selected)==1
    assert selected[0]['current_previous']=='친구가 도착했다.'
    assert k.prepare()==selected
    row['revision']='변경된 원본.'
    k.write_json(path,row)
    with pytest.raises(ValueError,match='Frozen artifact differs'): k.prepare()
