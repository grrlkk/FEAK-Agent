from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from verak.src.schemas import Profile, Sentence, Token
from verak.v3.common import file_sha, read_json, sha_text, write_json
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v3.env.analysis import ParagraphAnalyzer
from verak.v3.global_boost.config import config_for
from verak.v3.global_boost.gpu_recovery import SavedParagraphs, differences, rebuild


def saved_profile(directory, text, neighbors, *, form='내용'):
    profile=Profile([Sentence('S1',1,0,len(text),text,[Token(form,'NNG',0,len(text))])],[], 'bareun','saved-test')
    key=sha_text(json.dumps([text,sha_text(json.dumps(neighbors,ensure_ascii=False))],ensure_ascii=False))
    path=directory/(key+'.json')
    write_json(path,{'text':text,'neighbors':neighbors,'profile':profile.to_dict()})
    return path,profile


def test_cache_only_restore_reuses_identical_analyzer_text_with_new_neighbors(tmp_path,monkeypatch):
    from verak.v3.ko.annotation import KoreanStructure
    monkeypatch.setattr(KoreanStructure,'from_config',lambda *_:pytest.fail('No new Bareun analyzer'))
    path,profile=saved_profile(tmp_path,'내용', ['old neighboring sentence'])
    before=file_sha(path)
    reader=SavedParagraphs(config_for(),[tmp_path])
    actual=reader.profile('내용',['new neighboring sentence'])
    assert actual.to_dict()==profile.to_dict()
    assert reader.uses['identical_analyzer_text']==1 and not reader.calls
    assert reader.proofs[str(path)]['sha256']==before and file_sha(path)==before
    with pytest.raises(FileNotFoundError,match='No saved Bareun'):
        reader.profile('new uncached text',[])


def test_conflicting_identical_text_cache_profiles_are_a_blocker(tmp_path):
    saved_profile(tmp_path,'내용',['first'],form='one')
    saved_profile(tmp_path,'내용',['second'],form='two')
    reader=SavedParagraphs(config_for(),[tmp_path])
    with pytest.raises(ValueError,match='Ambiguous'):
        reader.profile('내용',['third'])


def test_saved_gpu_reward_rebuild_matches_frozen_formula_for_sol_attempt_without_scoring(monkeypatch):
    from verak.v3.corrupt.operators import apply, Proposal
    from verak.v3.global_boost import gpu_recovery as module
    from verak.v3.reward.total import rewards
    source=Document([Paragraph('P1',[Unit('S1','가.',[Token('가','NNG',0,1)])]),
                     Paragraph('P2',[Unit('S2','나.',[Token('나','NNG',0,1)])])],['','\n\n'])
    class Bank:
        def tokens(self,text):return [Token(text[0],'NNG',0,1)]
    corrupted,record=apply(source,Proposal('G_PARA_SWAP',['S1','S2'],{'paragraph_indices':[0,1]}),Bank())
    raw={'completed':True,'generation_completed':True,'reward':None,'data_boost':{'attempt':3},
        'stage1_layout':source.snapshot(),'final_layout':{'must_not_restore':'KOREAN'},
        'actions_by_role':{'global':[{'action':'MOVE','valid':True,'args':{'target':'P1'}},
                                   {'action':'STOP','valid':True}],'korean':[]},
        'calls':[{'raw':'saved Sol output'}]}
    candidate={'genre':'논증','question':'Q','records':[record],
        'corrupted_layout':corrupted.snapshot(),'preexisting_spell_spans':[]}
    resources=SimpleNamespace(source=lambda _:source,restore=lambda x:Document.restore(x,Bank()))
    observed=[]
    def saved_score(_,question,text):
        observed.append(text)
        return {'mean':5. if text==corrupted.text else 6.,'execution_device':'gpu_reference','scorer_fingerprint':'reference'}
    monkeypatch.setattr(module,'score_gpu_reference',saved_score)
    config=config_for();before=deepcopy(raw)
    value=rebuild(config,candidate,raw,resources,'reference','completed_gpu_hash')
    expected=rewards(source,corrupted,source,[record],genre='논증',q_corrupted=5.,q_stage1=6.,q_final=6.,
        config=config['reward'],mode='two_stage',stage1=source,stage1_actions=raw['actions_by_role']['global'],stage2_actions=[])['global']
    assert value['global_only_reward']==expected and raw==before
    assert observed==[corrupted.text,source.text]
    assert value['calls']==raw['calls'] and value['reward'] is None
    assert value['canonical_for_selection'] and value['score_source']=='gpu_reference'
    assert not differences(expected,value['global_only_reward'])
    wrong=deepcopy(expected);wrong['R_q']+=.001
    assert differences(expected,wrong)[0]['path']=='reward.R_q'


def test_difference_tolerance_never_masks_record_or_shape_changes():
    assert not differences({'R':.8},{'R':.8+1e-13})
    assert differences({'R':.8},{'R':.8+1e-6})
    assert differences({'success':True},{'success':1})
    assert differences({'records':[1]},{'records':[1,0]})
