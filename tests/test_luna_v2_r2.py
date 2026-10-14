import os
import pytest
from types import SimpleNamespace

from verak.v5.luna_v2_r2_env import BoundaryEnvironment,character_edits,josa,main_topic,references,conjunction
from verak.v3.corrupt.document import Document,Paragraph,Unit


def test_character_edit_ratio_and_particles():
    assert character_edits('가나다','가라다')['ratio']==pytest.approx(1/3)
    assert josa('철수')=='가' and josa('학생')=='이'
    assert josa('글',('으로','로'))=='로'


def test_character_distance_matches_reference_dp():
    import random
    rng=random.Random(17)
    for _ in range(100):
        a=''.join(rng.choices('가나다 .',k=rng.randrange(25)))
        b=''.join(rng.choices('가나다 .',k=rng.randrange(25)))
        row=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            nxt=[i]
            for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
            row=nxt
        assert character_edits(a,b)['distance']==row[-1]


def test_affected_scope_never_includes_direct_rewrite():
    d=Document([Paragraph('P1',[Unit('S1','앞.',[])]),Paragraph('P2',[Unit('S2','수정.',[]),Unit('S3','수정.',[])]),Paragraph('P3',[Unit('S4','뒤.',[])])],['','\n\n','\n\n'])
    assert BoundaryEnvironment.affected(d,{'P2'})=={'S1','S4'}


def test_deletion_limit_before_analyzer_calls():
    env=object.__new__(BoundaryEnvironment)
    d=Document([Paragraph('P1',[Unit(f'S{i}','문장.',[]) for i in range(10)])],[''])
    with pytest.raises(ValueError,match='30%'):env.structure(d,{'deleted':['S1','S2','S3','S4'],'paragraphs':[]})


@pytest.mark.skipif(os.environ.get('LUNA_R2_NATIVE_TEST')!='1',reason='requires existing local Bareun service')
def test_native_boundaries_references_topics_and_repairs():
    env=BoundaryEnvironment()
    old=env.document('철수는 비가 와서 집에 갔다.\n\n그래서 쉬었다. 이것은 좋은 선택이다.')
    candidate,lineage,affected,_=env.rewrite(old,'P1','영희는 맑은 날 학교에 갔다.',1)
    notices=env.notices(old,candidate,lineage,affected,'평서체')
    assert all(n['sid']=='S2' for n in notices)
    assert any(n['phi']=='phi_j' for n in notices)
    assert all(n['source_sids']==['S1'] for n in notices)
    assert main_topic(old.structure().annotations[0])['surface']=='철수'
    changed=env.repair(candidate,'S2','그래서 즐겁게 공부했다.')
    assert env.repair_scope(candidate,changed,'S2',old)
    same=env.repair(candidate,'S2','그래서 쉬었다.')
    assert env.repair_scope(candidate,same,'S2',old) is None
    refs=env.document('이 학생은 잘한다. 그 사람도 좋다. 저 친구는 온다. 이렇게 생각한다. 그렇게 한다. 이에 동의한다. 이로 인해 늦었다. 이처럼 한다.')
    annotations=refs.structure().annotations
    assert all(references(a) for a in annotations),[(a.text,[(t.form,t.tag) for t in a.tokens]) for a in annotations if not references(a)]
    assert conjunction(candidate.structure().annotations[1])['form']=='그래서'


@pytest.mark.skipif(os.environ.get('LUNA_R2_NATIVE_TEST')!='1',reason='requires existing local Bareun service')
def test_trivial_reference_and_obligation_suppression():
    env=BoundaryEnvironment()
    old=env.document('철수는 오늘 집에서 열심히 공부를 했다.\n\n이것은 좋은 일이다. 열심히 공부해야 한다.')
    changed,lineage,affected,_=env.rewrite(old,'P1','철수는 오늘 집에서 열심히 공부를 했다!',1)
    assert not any(n['phi']=='phi_z_reference' for n in env.notices(old,changed,lineage,affected,'평서체'))
    old=env.document('철수는 집에 갔다.\n\n열심히 공부해야 한다.')
    changed,lineage,affected,_=env.rewrite(old,'P1','영희는 학교에 갔다.',1)
    assert not any(n['phi']=='phi_z_subject' for n in env.notices(old,changed,lineage,affected,'평서체'))


@pytest.mark.skipif(os.environ.get('LUNA_R2_NATIVE_TEST')!='1',reason='requires existing local Bareun service')
@pytest.mark.parametrize('response,expected',[('고침: 그래서 쉬었다.','유지'),('고침: 그래서 즐겁게 공부했다.','막힘')])
def test_effective_korean_action_and_transaction_log(monkeypatch,response,expected):
    from verak.v5.luna_v2_r2 import RevisedEpisode
    from verak.v5 import luna_v2_run as base
    env=BoundaryEnvironment();original=env.document('철수는 집에 갔다.\n\n그래서 쉬었다.')
    candidate,lineage,affected,_=env.rewrite(original,'P1','영희는 학교에 갔다.',1)
    ep=object.__new__(RevisedEpisode);ep.env=env;ep.condition='B';ep.row={'question':'문항'}
    ep.config={'modification_limit':1.,'delta':0.,'enforce_repair_scope':True}
    events=[];ep.log=lambda kind,**kw:events.append({'kind':kind,**kw})
    ep.call=lambda *a,**k:response
    ep.meaning=lambda *a,**k:{'thesis_kept':True,'role_kept':True,'new_content':[],'reason':''}
    monkeypatch.setattr(base,'score',lambda *a,**k:{'q':4.})
    accepted,_,_=ep.inspect(original,candidate,lineage,affected,{'style':'평서체','referents':['철수','영희'],'thesis':'원래 주장','roles':{'P1':'근거','P2':'결론'}},'P1',discovery=True)
    decisions=[e for e in events if e['kind']=='korean_decision']
    assert decisions and all(e['action']==expected for e in decisions)
    assert all(e['before']==e['after']=='그래서 쉬었다.' for e in decisions)
    assert original.text=='철수는 집에 갔다.\n\n그래서 쉬었다.'
    assert accepted.locate('S2')[2].text=='그래서 쉬었다.'
