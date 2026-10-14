"""Rerun the same train10 after the user's 14-point human review. No eval50."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time

import numpy as np

from verak.v4.common import read_json,write_json,file_sha
from .luna_v2_data import ROOT
from .luna_v2_env import paragraph_text
from . import luna_v2_run as base
from .luna_v2_r2_env import BoundaryEnvironment,character_edits


def activate_prompts():
    base.TEMPLATES=read_json(Path(__file__).with_name('luna_v2_r2_prompts.json'))


class RevisedEpisode(base.Episode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.env=BoundaryEnvironment()

    def plan(self,doc,scores):
        value=super().plan(doc,scores)
        if value['thesis']!='명시된 주장 없음' and value['thesis'] not in doc.text:
            self.log('thesis_extraction',passed=False,proposed=value['thesis'])
            prompt='아래 계획에서 thesis만 고치세요. 학생이 원문에서 실제로 한 주장 문장을 원문 그대로 인용해야 합니다. 명시된 주장이 없으면 "명시된 주장 없음"이라고 쓰세요. 새 주장이나 해결책을 만들지 마세요. 나머지 필드는 그대로 두고 전체 JSON을 반환하세요.\n\n[원문]\n'+doc.text+'\n\n[계획]\n'+json.dumps(value,ensure_ascii=False)
            repaired=base.parse_json(self.call('orchestrator_thesis_repair',prompt,2048))
            thesis=repaired.get('thesis','')
            if not thesis or (thesis!='명시된 주장 없음' and thesis not in doc.text): raise ValueError('Thesis is not an original student assertion')
            value={**value,'thesis':thesis}
        self.log('thesis_extraction',passed=True,thesis=value['thesis']);return value

    def meaning(self,old_text,new_text,plan,role):
        if role!='글 전체 구성': return super().meaning(old_text,new_text,plan,role)
        prompt=base.render('P_structure_check',{'{thesis}':plan['thesis'],'{original}':old_text,'{candidate}':new_text})
        value=base.parse_json(self.call('orchestrator_structure_check',prompt,768))
        if type(value.get('claim_lost')) is not bool: raise ValueError('Invalid structure claim-loss check')
        self.log('structure_meaning',**value)
        return {'thesis_kept':not value['claim_lost'],'role_kept':True,'new_content':[],
                'reason':value.get('reason',''),'criterion':'only original assertion lost through deletion'}

    def modification(self,old,candidate,pid):
        result=super().modification(old,candidate,pid)
        a=old.text if pid=='structure' else paragraph_text(next(p for p in old.paragraphs if p.pid==pid))
        b=candidate.text if pid=='structure' else paragraph_text(next(p for p in candidate.paragraphs if p.pid==pid))
        chars=character_edits(a,b)
        return {**result,'character_edits':chars,'modification_ratio':chars['ratio'],
                'message':f"글자 단위 편집 비율이 {chars['ratio']:.1%}입니다. 고칠 점과 관계없는 부분은 그대로 두세요."}

    def run(self,**kwargs):
        result=super().run(**kwargs)
        result['modification']=character_edits(result['original'],result['final'])
        result['policy_revision']='human_review_r2'
        write_json(self.path/'result.json',result)
        return result


def source(meta):
    return read_json(ROOT/'sources'/(meta['source_id'].replace(':','_')+'.json'))


def discover(meta):
    row=source(meta); source_id=row['source_id'].replace(':','_')
    path=ROOT/'prep_r2_discovery/B'/source_id/'result.json'
    cfg={'delta':0.,'modification_limit':1.,'k':0,'N':1,'enforce_repair_scope':True}
    if path.exists(): result=read_json(path)
    else: result=RevisedEpisode(row,'B',cfg,'prep_r2_discovery').run(discovery=True)
    # Revalidate the same surface-only paraphrase under the revised P6 policy.
    old=read_json(ROOT/'calibration'/(source_id+'.json'))
    cpath=ROOT/'calibration_r2'/(source_id+'.json')
    if not cpath.exists():
        ep=RevisedEpisode(row,'B',cfg,'calibration_r2')
        meaning=ep.meaning(old['original_paragraph'],old['candidate'],result['plan'],'원래 문단의 역할')
        good=old['same_sentence_count'] and not old['safety'] and meaning['thesis_kept'] and meaning['role_kept'] and not meaning['new_content']
        delta=old['abs_delta']
        if good and delta is None:
            doc=ep.env.document(row['text']);p=doc.paragraphs[len(doc.paragraphs)//2]
            candidate,_,_,_=ep.env.rewrite(doc,p.pid,old['candidate'],0)
            delta=abs(base.score(row['question'],doc.text)['q']-base.score(row['question'],candidate.text)['q'])
        write_json(cpath,{**old,'meaning':meaning,'accepted':bool(good),'abs_delta':delta,
                         'paraphrase_reused':True,'meaning_rechecked_with_revised_policy':True})
        ep.api.close()
    print(json.dumps({'part':'checks_off','essay_id':row['source_id'],'notices':result['notices_by_type'],'calls':len(result['calls'])}),flush=True)
    return result


def checked(meta,cfg):
    row=source(meta);path=ROOT/'prep_r2_checked/B'/row['source_id'].replace(':','_')/'result.json'
    if path.exists(): result=read_json(path)
    else: result=RevisedEpisode(row,'B',cfg,'prep_r2_checked').run()
    print(json.dumps({'part':'checks_on','essay_id':row['source_id'],'notices':result['notices_by_type'],
                      'kept_original':sum(i['kept_original'] for i in result['items']),'paragraphs':len(result['items'])}),flush=True)
    return result


def parallel(function,rows):
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(function,row):row for row in rows}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as exc:
                write_json(ROOT/'prep_r2_errors'/(fs[f]['source_id'].replace(':','_')+'.json'),{'source_id':fs[f]['source_id'],'error_type':type(exc).__name__,'message':str(exc)})
                raise
    return results


def run():
    started=time.monotonic();activate_prompts()
    rows=read_json(ROOT/'selection.json')['prep10']
    start_off=time.monotonic();off=parallel(discover,rows);off_seconds=time.monotonic()-start_off
    ratios=[]
    for meta in rows:
        events=ROOT/'prep_r2_discovery/B'/meta['source_id'].replace(':','_')/'events.jsonl'
        current_safety=None
        for line in events.read_text().splitlines():
            e=json.loads(line)
            if e['kind']=='safety':current_safety=e['alignment']
            if e['kind']=='attempt' and e.get('meaning') and e.get('alignment') and all(m['thesis_kept'] and m['role_kept'] and not m['new_content'] for m in e['meaning']):
                measured=e['alignment'] if 'character_edits' in e['alignment'] else current_safety
                ratios.append(measured['character_edits']['ratio'])
            if e['kind']=='attempt':current_safety=None
    noise=[read_json(ROOT/'calibration_r2'/(r['source_id'].replace(':','_')+'.json')) for r in rows]
    accepted=[n['abs_delta'] for n in noise if n['accepted'] and n['abs_delta'] is not None]
    if not accepted or not ratios:raise ValueError('No valid training-only calibration values')
    cfg={'version':'human_review_r2','N':1,'k':1,'max_targets':3,'enforce_repair_scope':True,
         'delta':float(np.quantile(accepted,.9)),'delta_sample_count':len(accepted),
         'modification_limit':float(np.quantile(ratios,.9)),'modification_unit':'Levenshtein/max(original,candidate) Unicode characters including whitespace',
         'modification_calibration_samples':len(ratios),'structure_delete_limit':.30,
         'evaluation_50_authorized':False,'training':False,'same_prep10':True,
         'prompts_sha256':file_sha(Path(__file__).with_name('luna_v2_r2_prompts.json')),
         'zero_phi_policy':'Report zeros; never invent notices on directly rewritten sentences to satisfy coverage.'}
    cpath=ROOT/'prep_r2_config.json'
    if cpath.exists() and read_json(cpath)!=cfg:raise ValueError('R2 fixed thresholds changed')
    write_json(cpath,cfg)
    start_on=time.monotonic();on=parallel(lambda m:checked(m,cfg),rows);on_seconds=time.monotonic()-start_on
    write_json(ROOT/'prep_r2_complete.json',{'checks_off':len(off),'checks_on':len(on),'wall_seconds':time.monotonic()-started,
               'parts_seconds':{'checks_off_and_noise_calibration':off_seconds,'checks_on':on_seconds},'config':cfg})
    print(json.dumps({'complete':True,'config':cfg,'wall_seconds':time.monotonic()-started},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
