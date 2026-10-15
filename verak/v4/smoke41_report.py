"""Saved-evidence B1 diagnostics; never calls a model, analyzer or environment."""
from collections import Counter
import json
import re

from .common import atomic_new,collection_lock,constrain_cpu,file_sha,read_json,safe_id
from .smoke41 import ROOT


def classify_edit(call):
    action=call['action'];value=action.get('value',{})
    if action['action']!='EDIT' or action['valid']:
        raise ValueError('Expected a rejected EDIT')
    old,new=value.get('old'),value.get('new')
    payload=json.loads(call['public_messages'][1]['content'])
    before=next((s['text'] for p in payload['paragraphs'] for s in p['sentences']
                 if s['id']==value.get('sentence')),None)
    after=before.replace(old,new,1) if isinstance(old,str) and isinstance(new,str) and before is not None and before.count(old)==1 else None
    result={'sentence':value.get('sentence'),'old':old,'new':new,'before_sentence':before,
            'proposed_sentence':after,'error':action.get('error'),'category':'other_rejected_EDIT'}
    if not (isinstance(old,str) and isinstance(new,str)): return result
    boundary='문장 수' in action.get('error','')
    if old!=new and re.sub(r'\s+','',old)==re.sub(r'\s+','',new):
        result['category']='pure_spacing_boundary_rejection' if boundary else 'pure_spacing_other_rejection'
    elif boundary and after is not None:
        old_ends=len(re.findall(r'[.!?。！？]',before));new_ends=len(re.findall(r'[.!?。！？]',after))
        result.update(before_terminal_punctuation=old_ends,after_terminal_punctuation=new_ends)
        result['category']='added_terminal_punctuation_split' if new_ends>old_ends else 'boundary_rejection_without_new_terminal_punctuation'
    return result


def diagnostics(attempts):
    grouped={};other=[]
    for attempt in attempts:
        for call in attempt['calls']:
            if call['action']['valid']: continue
            provenance={'source_id':attempt['source_id'],'role':call['role'],
                'delegation':call['delegation'],'turn':call['turn'],'phase_call':call['phase_call']}
            if call['action']['action']!='EDIT':
                other.append({**provenance,'raw':call['raw'],'action':call['action']});continue
            value=classify_edit(call)
            key=json.dumps([attempt['source_id'],call['role'],value],ensure_ascii=False,sort_keys=True)
            if key not in grouped:
                grouped[key]={'source_id':attempt['source_id'],'role':call['role'],**value,'occurrences':[]}
            grouped[key]['occurrences'].append(provenance)
    groups=sorted(grouped.values(),key=lambda r:(r['category'],r['source_id'],r['role'],str(r['sentence']),str(r['old'])))
    for group in groups: group['count']=len(group['occurrences'])
    counts={role:dict(Counter({category:sum(g['count'] for g in groups if g['role']==role and g['category']==category)
            for category in sorted({g['category'] for g in groups if g['role']==role})})) for role in ('revision','korean')}
    return {'basis':'Saved public observations, rejected action values and exact errors only; no re-analysis or model call',
        'gate_adjusted':False,'rejected_EDIT_calls':sum(g['count'] for g in groups),
        'unique_rejected_EDIT_variants':len(groups),'rejected_EDIT_by_role_category':counts,
        'rejected_EDIT_groups':groups,'other_invalid_calls':other,
        'classification_limits':'Pure spacing means the old/new strings are identical after whitespace removal. New terminal punctuation is a textual split indicator, not a semantic judgment. Other segmentation rejections remain explicitly separate.'}


def render(value):
    labels={'pure_spacing_boundary_rejection':'Pure spacing; boundary validator rejected',
        'pure_spacing_other_rejection':'Pure spacing; another constraint rejected',
        'added_terminal_punctuation_split':'EDIT added sentence-terminal punctuation',
        'boundary_rejection_without_new_terminal_punctuation':'Boundary rejection without added terminal punctuation',
        'other_rejected_EDIT':'Other rejected EDIT'}
    lines=['# Saved-action diagnostic','',
        'All original invalid actions stay in the gate denominator. This analysis changes no gate, prompt, validator or outcome. '
        'Counts are action-weighted: repeated rejected proposals naturally contribute repeated invalid calls, even when they target the same source span.','',
        '|Role|Observed rejected EDIT category|Calls|','|---|---|---:|']
    for role,counts in value['rejected_EDIT_by_role_category'].items():
        for category,count in counts.items(): lines.append(f'|{role}|{labels[category]}|{count}|')
    lines += ['',value['classification_limits'],'',
        'The inherited EDIT validator reports “KOREAN은 문장 수를 늘릴 수 없습니다.” for both roles. '
        'The role column below comes from the actual calling editor. A pure spacing correction can change Bareun sentence segmentation; '
        'that remains an invalid action under the frozen contract, not evidence that the model invented another sentence.','']
    for index,group in enumerate(value['rejected_EDIT_groups'],1):
        lines += [f'## {index}. {group["source_id"]} / {group["role"]} / {group["sentence"]}','',
            f'{labels[group["category"]]}; repeated count **{group["count"]}**.','',
            '**Proposed substring change**','','```text',f'old: {group["old"]}',f'new: {group["new"]}','```','',
            '**Observed sentence before / proposed sentence**','','```text',f'before: {group["before_sentence"]}',
            f'proposed: {group["proposed_sentence"]}','```','',f'Frozen rejection: `{group["error"]}`.','',
            'Saved calls: '+', '.join(f'd{x["delegation"]}/t{x["turn"]}/API{x["phase_call"]}' for x in group['occurrences'])+'.','']
    return '\n'.join(lines)


def run():
    constrain_cpu()
    with collection_lock(ROOT):
        final=ROOT/'final_complete.json'
        if final.exists(): return read_json(final)
        complete=read_json(ROOT/'complete.json')
        if not complete['stopped'] or not complete['no_live_paid_calls'] or complete['api']['pending']:
            raise ValueError('B1 collection is not settled')
        for kind in ('metrics','report'):
            if file_sha(complete[kind+'_path'])!=complete[kind+'_sha256']:
                raise ValueError('Original B1 artifact identity changed')
        sample=read_json(ROOT/'sample.json')
        paths=[ROOT/'content/attempts'/(safe_id(s)+'_a1.json') for s in sample['source_ids']]
        attempts=[read_json(p) for p in paths]
        if len(attempts)!=20 or {a['source_id'] for a in attempts}!=set(sample['source_ids']):
            raise ValueError('Expected all 20 fixed outcomes')
        result=diagnostics(attempts)
        result.update(attempt_files_sha256={str(p):file_sha(p) for p in paths},
            original_complete_sha256=file_sha(ROOT/'complete.json'),original_gate=complete['gate'])
        diagnostic_path=ROOT/'diagnostics.json'
        if diagnostic_path.exists():
            if read_json(diagnostic_path)!=result: raise ValueError('Saved diagnostic changed')
        else: atomic_new(diagnostic_path,result)
        text=render(result)
        (ROOT/'diagnostics.md').write_text(text,encoding='utf-8')
        report=ROOT/'component_report_final.md'
        report.write_text((ROOT/'component_report.md').read_text()+'\n\n'+text,encoding='utf-8')
        marker={**complete,'report_path':str(report),'report_sha256':file_sha(report),
            'original_complete_path':str(ROOT/'complete.json'),'original_complete_sha256':file_sha(ROOT/'complete.json'),
            'diagnostics_path':str(ROOT/'diagnostics.json'),'diagnostics_sha256':file_sha(ROOT/'diagnostics.json'),
            'diagnostics_report_path':str(ROOT/'diagnostics.md'),'diagnostics_report_sha256':file_sha(ROOT/'diagnostics.md'),
            'gate_unchanged':True,'new_paid_calls':0,'new_analyzer_calls':0,'new_GPU_calls':0}
        atomic_new(final,marker);return marker


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
