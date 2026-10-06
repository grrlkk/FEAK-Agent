"""Answer recovery and role attribution, without an episode runner or paid calls."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verak.src.schemas import Token
from verak.v3.common import load_config, read_json
from verak.v3.corrupt.document import BareunBank, Document
from verak.v3.corrupt.operators import Proposal, apply, restore_record
from verak.v3.phase2 import restore_profile
from verak.v3.reward.overedit import edit_distance, mapped_span, overedit
from verak.v3.reward.recovery import (ACTIVE_OPERATORS, coupled_recovery, deletion_recovery,
                                     graded_similarity, main_recovery, recover_records)
from verak.v3.reward.total import quality_reward, rewards
from verak.v3.tests.test_instance_pilot import sample as instance_example


@pytest.fixture
def example(tmp_path):
    fixture = read_json(Path(__file__).with_name('fixtures')/'corrupt_bareun.json')
    config = load_config()
    bank = BareunBank(config, cache_dir=tmp_path)
    bank.memory = {s: [Token(**t) for t in tokens] for s,tokens in fixture['tokens'].items()}
    bank.tokens = lambda text: deepcopy(bank.memory[text])
    source = Document.from_profile(fixture['text'], restore_profile(fixture['profile']))
    return source,bank,fixture,config['reward']


@pytest.mark.parametrize('op', sorted(ACTIVE_OPERATORS))
def test_each_active_operator_source_and_unchanged(example,op):
    source,bank,fixture,_ = example
    changed,record = apply(source,Proposal(**fixture['proposals'][op]),bank)
    restored = restore_record(changed,record,bank)
    assert main_recovery(source,restored,record) == 1
    value = main_recovery(source,changed,record)
    if op=='G_PARA_SWAP':
        assert value==pytest.approx(2/3)  # Unchanged Kendall partial credit, explicitly reported.
    else:
        assert value==0
    assert overedit(source,source,restored,[record])['value']==0


def test_unrelated_rewrite_penalized_without_losing_recovery(example):
    source,bank,fixture,_ = example
    _,record = apply(source,Proposal(**fixture['proposals']['L_CONN']),bank)
    final=source.clone()
    unit=final.locate('S8')[2]
    unit.text='항상 실천해야 한다.'
    unit.tokens=bank.tokens(unit.text)
    assert main_recovery(source,final,record)==1
    assert overedit(source,source,final,[record])['value']>0


@pytest.mark.parametrize('sim,expected',[(.69,0),(.70,0),(.75,.5),(.80,1),(.9,1)])
def test_graded_support_recovery(sim,expected):
    assert graded_similarity(sim,.75)==pytest.approx(expected)
    assert deletion_recovery(sim,.75,at_deletion_site=False)==0
    assert deletion_recovery(None,.75,at_deletion_site=True)==0


def test_deleted_site_accepts_new_id_but_not_an_existing_neighbor(example):
    source,bank,fixture,_=example
    corrupted,record=apply(source,Proposal(**fixture['proposals']['G_DELETE_SUPPORT']),bank)
    assert main_recovery(source,corrupted,record,similarity=lambda a,b:1,tau=.7)==0
    reconstructed=source.clone()
    unit=reconstructed.locate('S2')[2]
    unit.sid='N1'
    unit.text='항상 실천해야 한다.'
    unit.tokens=bank.tokens(unit.text)
    assert main_recovery(source,reconstructed,record,similarity=lambda a,b:.75,tau=.75)==pytest.approx(.5)
    reconstructed.paragraphs[0].units.remove(unit)
    reconstructed.paragraphs[2].units.append(unit)
    assert main_recovery(source,reconstructed,record,similarity=lambda a,b:1,tau=.75)==0


def test_known_offtopic_insertion_never_counts_as_missing_support(example):
    source,bank,fixture,_=example
    deleted,deletion=apply(source,Proposal(**fixture['proposals']['G_DELETE_SUPPORT']),bank)
    proposal=Proposal(**deepcopy(fixture['proposals']['G_OFFTOPIC']))
    proposal.params['position']=1
    corrupted,offtopic=apply(deleted,proposal,bank)
    rows=recover_records(source,corrupted,[deletion,offtopic])
    assert [r['main'] for r in rows]==[0,0]


def test_same_relation_different_ec_and_shifted_site(example):
    source,bank,fixture,_=example
    _,record=apply(source,Proposal(**fixture['proposals']['L_CONN']),bank)
    ann=source.structure().annotations[2]
    different=replace(ann,text=ann.text.replace('하면','하거든'))
    different.connectives=[{'eligible': True,'kind':'EC','coarse_class':'CONDITION',
                            'span':[ann.start+6,ann.start+8]}]
    assert main_recovery(source,source,record,annotations={'S3':different})==1
    different.connectives[0]['span'][1]+=5
    assert main_recovery(source,source,record,annotations={'S3':different})==0


def test_coupled_dep_hint_lower_weight_and_uncertain_omission_not_explicit():
    record={'coupled_changes':[
        {'kind':'CONJ','sid':'S2','previous_before':'S1','coarse_class_before':'RESULT'},
        {'kind':'DEP','sid':'S3','previous_before':'S2'}]}
    ann={'S2':SimpleNamespace(predecessor_id='S1',initial_conj=None),
         'S3':SimpleNamespace(predecessor_id='S9',subject_evidence=[],subject_omitted=False)}
    value,detail=coupled_recovery(record,ann)
    assert value==pytest.approx(2/3)
    assert detail[1]['recovered']==0
    ann['S3'].subject_evidence=[{'marker':'은'}]
    assert coupled_recovery(record,ann)[0]==1
    assert coupled_recovery(record,ann,blocked_subject_ids={'S3'})[0]==pytest.approx(2/3)


def test_subject_corruption_cannot_earn_dependency_repair_credit(example):
    source,bank,fixture,_=example
    changed,subject=apply(source,Proposal(**fixture['proposals']['L_SUBJ_INSERT']),bank)
    p=changed.paragraphs[1]
    p.units[0],p.units[2]=p.units[2],p.units[0]
    global_record={'op':'G_PARA_SWAP','level':'GLOBAL','sids':['S4'],
        'recovery_target':{'paragraph_ids':['P1','P2','P3']},
        'coupled_changes':[{'kind':'DEP','sid':'S5','previous_before':'S4'}]}
    results=recover_records(source,changed,[global_record,subject])
    assert results[0]['coupled']==0
    assert results[1]['main']==0


def test_coupled_same_class_new_conjunction_and_wrong_class():
    record={'coupled_changes':[{'kind':'CONJ','sid':'S2','previous_before':'S1','coarse_class_before':'RESULT'}]}
    src={'S2':SimpleNamespace(initial_conj={'form':'그래서','coarse_class':'RESULT'})}
    current={'S2':SimpleNamespace(predecessor_id='S9',initial_conj={'form':'따라서','eligible':True,'coarse_class':'RESULT'})}
    assert coupled_recovery(record,current,src)[0]==1
    current['S2'].initial_conj['coarse_class']='ADDITION'
    assert coupled_recovery(record,current,src)[0]==0


def test_coupled_target_is_source_predecessor_not_prior_corruption():
    record={'coupled_changes':[{'kind':'DEP','sid':'S3','previous_before':'S9',
                                'recovery_predecessor_id':'S2'}]}
    restored={'S3':SimpleNamespace(predecessor_id='S2',subject_evidence=[])}
    assert coupled_recovery(record,restored)[0]==1
    restored['S3'].predecessor_id='S9'
    assert coupled_recovery(record,restored)[0]==0


def test_main_recovered_but_coupled_position_unrecovered_gives_point_seven(example):
    source,_,_,_=example
    record={'op':'G_PARA_SWAP','level':'GLOBAL','sids':['S4'],
            'recovery_target':{'paragraph_ids':['P1','P2','P3']},
            'coupled_changes':[{'kind':'DEP','sid':'S5','previous_before':'S4'}]}
    final=source.clone()
    p=final.paragraphs[1]
    p.units[0],p.units[2]=p.units[2],p.units[0]
    result=recover_records(source,final,[record])[0]
    assert result['main']==1 and result['coupled']==0
    assert result['recovery']==pytest.approx(.7)


def test_role_split_uses_middle_global_final_local_and_separate_quality(example):
    source,bank,fixture,config=example
    # Stage 1 restores order, while S3 and S2 remain locally corrupt.
    local,word=apply(source,Proposal(**fixture['proposals']['L_CONN']),bank)
    local,text=apply(local,Proposal(**fixture['proposals']['L_REGISTER']),bank)
    local,sentence=apply(local,Proposal(**fixture['proposals']['L_CONJ']),bank)
    corrupted,global_record=apply(local,Proposal(**fixture['proposals']['G_PARA_SWAP']),bank)
    global_record['coupled_changes']=[{'kind':'DEP','sid':'S5','previous_before':'S4'}]
    logs1=[{'action':'MOVE','args':{'target':'P1'}},{'action':'STOP'}]
    logs2=[{'action':'EDIT','changed_sids':['S2','S3','S5']},{'action':'CHECK'},{'action':'STOP'}]
    result=rewards(source,corrupted,source,[global_record,word,sentence,text],genre='논증',
        q_corrupted=5,q_stage1=6,q_final=5.5,config=config,mode='two_stage',
        stage1=local,stage1_actions=logs1,stage2_actions=logs2)
    assert result['global']['recovery_terms']==[1]
    assert result['korean']['recovery_terms']==[1,1,1,1]
    assert {r['level'] for r in result['korean']['per_record']}=={'GLOBAL','WORD','SENTENCE','TEXT'}
    assert result['global']['R_step']==2 and result['korean']['R_step']==3
    assert result['combined']['R_step']==5
    assert result['global']['quality']['delta']==1
    assert result['korean']['quality']['delta']==-.5
    assert result['combined']['quality']['delta']==.5
    assert result['combined']['R_rec']==1
    # A stage-2 restoration must never retroactively give stage-1 order credit.
    result2=rewards(source,corrupted,source,[global_record,word,sentence,text],genre='논증',
        q_corrupted=5,q_stage1=5,q_final=6,config=config,mode='two_stage',stage1=corrupted,
        stage2_actions=logs2)
    assert result2['global']['R_rec']==pytest.approx(2/3)


def test_role_overedit_attribution_undo_and_excluded_sites(example):
    source,bank,fixture,config=example
    corrupted,record=apply(source,Proposal(**fixture['proposals']['L_CONN']),bank)
    middle=corrupted.clone()
    unit=middle.locate('S8')[2]
    unit.text='항상 실천해야 한다.'
    unit.tokens=bank.tokens(unit.text)
    result=rewards(source,corrupted,middle,[record],genre='설명',q_corrupted=5,q_stage1=5,
        q_final=5,config=config,mode='two_stage',stage1=middle,
        stage1_actions=[{'action':'EDIT','changed_sids':['S8']}])
    assert result['global']['R_over']>0 and result['korean']['R_over']==0
    with pytest.raises(ValueError,match='Unattributed'):
        overedit(source,corrupted,middle,[record],actions=[])
    assert overedit(source,corrupted,corrupted,[record],actions=[
        {'action':'EDIT','changed_sids':['S8']},{'action':'UNDO'}])['value']==0
    assert overedit(source,source,middle,[record,{'sids':['S8']}])['value']==0


def test_no_record_roles_get_zero_not_free_recovery(example):
    source,_,_,config=example
    result=rewards(source,source,source,[],genre='정서',q_corrupted=5,q_stage1=5,q_final=5,
                   config=config,mode='two_stage',stage1=source)
    assert all(result[k]['R_rec']==0 for k in ('global','korean','combined'))


@pytest.mark.parametrize('genre,floor',[('설명',.71466138),('논증',.21266690),('정서',.52354468),('?',.5294307612337095)])
def test_genre_dead_zone_symmetric_without_clipping(example,genre,floor):
    config=example[3]
    assert quality_reward(5,5+floor/2,genre,config)['value']==0
    assert quality_reward(4,6,genre,config)['value']==pytest.approx((2-floor)/2)
    assert quality_reward(6,4,genre,config)['value']==pytest.approx(-(2-floor)/2)
    assert quality_reward(1,9,genre,config)['value']>1


def test_surface_projection_and_morpheme_edit_distance():
    assert mapped_span('물을 낭비하면 자원이 부족하다.','늘 물을 낭비하거든 자원이 부족하다.',6,7)==(8,10)
    assert edit_distance([('가','VV')],[('가','JKS')])==1
    assert edit_distance([],['a','b'])==2


def test_preexisting_spelling_spans_excluded(example):
    source,bank,fixture,_=example
    final,_=apply(source,Proposal(**fixture['proposals']['L_SPACING']),bank)
    # Full-sentence pre-existing span also verifies unit-local masking of tokens.
    assert overedit(source,source,final,[],preexisting_spell_spans=[
        {'sid':'S5','start':0,'end':len(source.locate('S5')[2].text)}])['value']==0


@pytest.mark.parametrize('op',['L_POLARITY','G_DELETE_SUPPORT'])
def test_updated_phase3b_operator_targets(instance_example,op):
    from verak.v3.corrupt.instance_policy import candidates
    source,bank=instance_example
    # The immutable Bareun fixture stores the first application of each operator.
    for proposal in candidates(source,op)[:1]:
        final,record=apply(source,proposal,bank)
        assert main_recovery(source,final,record)==0
        assert main_recovery(source,source,record)==1


def test_real_reward_uses_structural_changes_not_antecedent_identities(example):
    from verak.v3.reward.total import real_reward
    source,bank,fixture,config=example
    result=real_reward(source,source,q_before=5,q_after=5,genre='설명',actions=[],config=config)
    assert result['R']==0
    changed,_=apply(source,Proposal(**fixture['proposals']['L_CONN']),bank)
    result=real_reward(source,changed,q_before=5,q_after=5,genre='설명',actions=[{'action':'EDIT'}],config=config)
    assert result['R']<0
    assert result['cohesion']['counts_by_level']['WORD']==1
    assert not any('antecedent' in c for c in result['cohesion']['changes'])
