import json

from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v4.scale2_projection import save_document
from verak.v5 import cooperate_eval as module
from verak.v5.common import write_json, ref


def document(text='둘이다.'):
    return save_document(Document([Paragraph('P1', [Unit('S1','하나다.',[]),
        Unit('S2',text,[],' ')])],['']))


def test_judge_is_full_context_blinded_and_identical_repeats_are_deduplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path/'experiment')
    monkeypatch.setattr(module,'REPO',tmp_path/'repo')
    write_json(module.ROOT/'inference_complete.json',{})
    write_json(module.ROOT/'api_contract.json',{'user_approved':True,'total_cap_usd':5})
    write_json(module.REPO/'verak/v5/outputs/check/validation/final.json',
               {'continue':True,'validated_setting':'full'})
    row=module.ROOT/'row.json';write_json(row,{'question':'설명하시오.'})
    case={'case_id':'private_case','cohort':'smoke20','row':ref(row)}
    a,b=document(),document('둘이었다.')
    values={'onepass':{'initial':a,'final':a,'alignment':{'failed':False}},
            'luna':{'initial':a,'final':b}}
    values.update({c:{'final_token_document':a if c.startswith('old') else b} for c in module.CONDITIONS[:6]})
    monkeypatch.setattr(module,'verify_design',lambda:{'cases':[case]})
    monkeypatch.setattr(module,'results',lambda c:(values,{}))
    manifest=module.judging_manifest()
    assert len(manifest['requests']) == 2
    assert len(manifest['cases'][0]['aliases']) == 9
    for req in manifest['requests']:
        body=json.loads(req['messages'][1]['content'])
        assert len(body['original'][0]['sentences']) == len(body['revised'][0]['sentences']) == 2
        assert body['variants'] == [{'id':'V1','paragraphs':body['revised']}]
        assert req['messages'][0]['content'] == module.JUDGE_PROMPT
        assert not any(label in req['messages'][1]['content'] for label in module.CONDITIONS)
        assert 'private_case' not in req['messages'][1]['content']
    assert module.judging_manifest() == manifest


def test_missing_judgment_and_missing_site_are_not_treated_as_fine(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    jc={'aliases':{'x':'missing'},'sid_maps':{'x':{'N1':'S1'}},'selected_original_sites':['S1','S2']}
    quality,labels=module.verdict(jc,'x')
    assert quality == {}
    assert labels == {'S1':'unjudged','S2':'unavailable'}
    m=module.transitions({'S1':'yes','S2':'no'},labels,set())
    assert m['uncertain_or_unjudged_pairs'] == 2
    assert m.get('fixed',0) == m.get('new_broken',0) == 0


def test_alignment_failure_excludes_onepass_from_link_site_sampling_only():
    a=document();b=document('다른 문장이다.')
    assert module.selected_sites('x',{'original':a,'onepass':b},True) == []
    assert module.selected_sites('x',{'original':a,'onepass':b},False) == ['S2']
    assert module.text_changes(a,b)['changed_source_sentences'] == 1
