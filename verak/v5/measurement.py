"""Post-generation record audit. Gold never enters a deployed policy request."""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from difflib import SequenceMatcher
import re
import time

from .common import *
from verak.v4.environment_v43 import prepared_document
from verak.v4.environment_v45 import match_span
from verak.v4.render_v43 import assert_rendered
from verak.v4.scale2_projection import (restore_document, project_document, private_document, ProjectionUnknown)
from verak.v3.reward.recovery import recover_records
from verak.v3.reward.overedit import overedit, mapped_span

OPERATORS=('G_PARA_SWAP','G_SENT_MOVE','G_OFFTOPIC','L_CONN','L_CONJ','L_REGISTER','L_SUBJ_INSERT','L_SPACING','L_POLARITY')
# Initial lexical candidates, retained for audit/testing only. Reported a1
# recall uses the saved semantic task-text review, with ambiguity explicit.
PATTERNS={
    'G_PARA_SWAP':r'문단.*(순서|배치|이동|전개)|(?:순서|배치|이동).*문단',
    'G_SENT_MOVE':r'문장.*(위치|순서|배치|이동)|(?:위치|순서|배치|이동).*문장',
    'G_OFFTOPIC':r'주제|문항|관련.?없|무관|벗어나|딴.?이야기',
    'L_CONN':r'연결.?어미|접속.?어미|절.*관계|어미.*(?:인과|조건|양보)',
    'L_CONJ':r'접속[사어]|문두|접속 표현|연결어(?!미)|그래서|그러나|따라서|하지만|그런데|예를 들어',
    'L_REGISTER':r'문체|종결|높임|경어|존댓말|반말|[한해]다체|해요체|합니다체',
    'L_SUBJ_INSERT':r'주어|주체|주격|생략|그는|그녀는',
    'L_SPACING':r'띄어.?쓰|붙여.?쓰|띄워.?쓰',
    'L_POLARITY':r'부정|양태|긍정|수밖에|뿐만 아니라|지 않|지 못',
}


def problem_types(item):
    # Recall concerns finding the problem. Choosing the wrong action does not
    # erase a correctly identified problem; execution quality is measured later.
    text=item.get('instruction','')
    return {op for op,pattern in PATTERNS.items() if re.search(pattern,text)}


def task_key(item,owner):
    return sha_text(json.dumps([owner,item['action'],item['instruction']],ensure_ascii=False))


@lru_cache(maxsize=1)
def semantic_review():
    value=read_json(ROOT/'A/problem_type_review.json')
    if value['source_sha256']!=file_sha(ROOT/'A/problem_type_review_input.json'):
        raise ValueError('Semantic task review input changed')
    return value['items']


def record_public_ids(prepared,record):
    inverse={v:k for k,v in prepared['stable_id_aliases']['public_to_private_sentences'].items()}
    mapping=prepared['policy_row']['normalization']['sid_mapping']
    return {s for sid in record['sids'] for s in mapping.get(inverse.get(sid,sid),[])}


def expanded_scope(locations,paragraphs,lineage=None):
    sids={s['id'] for p in paragraphs if p['id'] in locations for s in p['sentences']}
    sids.update(s['id'] for p in paragraphs for s in p['sentences'] if s['id'] in locations)
    for sid in list(sids):
        seen=set()
        while sid in (lineage or {}) and sid not in seen:
            seen.add(sid);parent=lineage[sid]['parent_sid'];sids.add(parent)
            if parent==sid:break
            sid=parent
    return sids


def a1_planning(prepared,result,audit=None):
    records=prepared['corpus']['records'];ids={r['record_id']:record_public_ids(prepared,r) for r in records}
    delegations=[];outside=[]
    for call in result['calls']:
        action=call.get('action',{})
        if call.get('role')=='orchestrator' and action.get('valid') and action.get('action')=='DELEGATE':
            action=action['value'];payload=read_json(call['prefix']['path'])['messages']
            state=__import__('json').loads(payload[1]['content'])
            items=[]
            for item in action['items']:
                lineage=state.get('sentence_lineage',result.get('final',{}).get('lineage',{}))
                scope=expanded_scope(item['location'],state['paragraphs'],lineage)
                key=task_key(item,item.get('owner',action['to']))
                if audit is None:
                    types=problem_types(item);possible=set();review=None
                else:
                    if key not in audit:raise ValueError('Unreviewed Orchestrator task '+key)
                    review=audit[key];types=set(review['operator_types']);possible=set(review['possible_types'])
                items.append({'task':item,'expanded_scope':sorted(scope),'problem_types':sorted(types),
                    'possible_problem_types':sorted(possible),'semantic_review_key':key if review else None,
                    'touches_records':[rid for rid,sids in ids.items() if scope&sids],
                    'matched_records':[r['record_id'] for r in records if scope&ids[r['record_id']] and r['op'] in types],
                    'possibly_matched_records':[r['record_id'] for r in records if scope&ids[r['record_id']] and r['op'] in possible]})
            touches=sorted({rid for i in items for rid in i['touches_records']})
            delegations.append({'to':action['to'],'items':items,'touches_no_record':not touches,
                                'touches_records':touches})
        if call.get('role')=='revision' and action.get('action')=='MOVE' and not action.get('valid') and '이동 목적지도 배정된 위치' in action.get('error',''):
            state=assert_rendered(read_json(call['prefix']['path'])['messages'])
            scope=expanded_scope([action.get('value',{}).get('sentence')],state['paragraphs'],state.get('sentence_lineage'))
            ops=sorted({r['op'] for r in records if scope&ids[r['record_id']]})
            outside.append({'action':action,'operators':ops,'stage':call.get('stage')})
    recall=[]
    for r in records:
        rid=r['record_id']
        matching=[i for d in delegations for i in d['items'] if rid in i['matched_records']]
        possible=[i for d in delegations for i in d['items'] if rid in i['possibly_matched_records']]
        scope=any(rid in d['touches_records'] for d in delegations)
        recall.append({'record_id':rid,'operator':r['op'],'in_explicit_delegated_scope':scope,
                       'matching_problem_type':bool(matching),'matching_items':matching,
                       'ambiguous_problem_type_only':bool(possible) and not matching,'possibly_matching_items':possible})
    return {'records':recall,'delegations':delegations,'no_record_delegations':sum(d['touches_no_record'] for d in delegations),
            'destination_scope_rejections':outside,
            'type_assignment':'post-hoc Codex semantic task-text review; uncertain matches separate' if audit is not None else 'lexical candidate classifier only',
            'ambiguous_items':[i['task'] for d in delegations for i in d['items'] if i['possible_problem_types']],
            'unclassified_items':[i['task'] for d in delegations for i in d['items'] if not i['problem_types']]}


def final_korean_changes(prepared,result,condition):
    records=[r for r in prepared['corpus']['records'] if r['op'] in {'L_CONN','L_CONJ'}]
    if condition=='a1':
        indices=[i+1 for i,s in enumerate(result['stages']) if s['role']=='korean']
        last=indices[-1] if indices else None
        calls=[c for c in result['calls'] if c.get('stage')==last and c['role']=='korean']
    else:
        final=next((s for s in result['stages'] if s.get('final_whole_essay_pass')),None)
        calls=final['calls'] if final else []
    rows=[]
    for record in records:
        sids=record_public_ids(prepared,record);edits=[]
        for c in calls:
            a=c.get('action',{})
            if not a.get('valid') or a.get('action')!='EDIT':continue
            value=a['value'];sid=value['sentence']
            if sid not in sids:continue
            if 'before' in c:
                before={u.sid:u.text for u in restore_document(c['before']).units}.get(sid)
            else:
                payload=assert_rendered(read_json(c['prefix']['path'])['messages'])
                before=next((s['text'] for p in payload['paragraphs'] for s in p['sentences'] if s['id']==sid),None)
            if before is None:raise ValueError('Edited sentence absent from recorded prefix')
            span,_=match_span(before,value['old'])
            if span is None:raise ValueError('Accepted action cannot be mapped to its saved observation')
            after=before[:span[0]]+value['new']+before[span[1]:]
            private=record['sids'][0]
            site=mapped_span(record['original_text'][private],before,*record['recovery_target']['site'])
            intersects=(max(span[0],site[0])<min(span[1],site[1]) or
                        span[0]==span[1] and site[0]<=span[0]<=site[1]) if site else None
            changes=[]
            if site:
                for tag,i,j,k,l in SequenceMatcher(None,before,after,autojunk=False).get_opcodes():
                    if tag=='equal':continue
                    touches=(max(i,site[0])<min(j,site[1]) or i==j and site[0]<=i<site[1])
                    if touches:changes.append({'before_span':[i,j],'after_span':[k,l],
                        'old':before[i:j],'new':after[k:l]})
            edits.append({'sid':sid,'old':value['old'],'new':value['new'],'sentence_before':before,'sentence_after':after,
                          'edit_span':span,'mapped_record_site':site,'edits_record_position':intersects,
                          'record_position_text_changed':bool(changes) if site else None,'record_position_changes':changes})
        rows.append({'record_id':record['record_id'],'operator':record['op'],'sentence_edited':bool(edits),
            'position_edited':any(e['edits_record_position'] is True for e in edits),
            'position_text_changed':any(e['record_position_text_changed'] is True for e in edits),
            'position_unknown':any(e['edits_record_position'] is None for e in edits),'edits':edits})
    return rows


def record_changed(record,before,after):
    def positions(doc):
        return {u.sid:(p.pid,pi,si,u.text) for pi,p in enumerate(doc.paragraphs) for si,u in enumerate(p.units)}
    a,b=positions(before),positions(after)
    if record['op']=='G_PARA_SWAP':return [p.pid for p in before.paragraphs]!=[p.pid for p in after.paragraphs]
    if record['op']=='G_SENT_MOVE':return any(a.get(s)!=b.get(s) for s in record['sids'])
    return any((a.get(s) or (None,None,None,None))[3]!=(b.get(s) or (None,None,None,None))[3] for s in record['sids'])


def measure_case(case,condition):
    prepared=verified(case['prepared']);row=prepared['corpus'];eid=case['case_id']
    if condition=='a1':path=ROOT/'A/a1/attempts'/(safe_id(eid)+'.json')
    else:path=ROOT/'A'/condition/safe_id(eid)/('attempt.json' if condition=='a2' else 'onepass.json')
    result=read_json(path)
    out={'case_id':eid,'operators':sorted({r['op'] for r in row['records']}),'condition':condition,'artifact':ref(path)}
    if condition=='a1':
        doc=restore_document(result['final_token_document']);lineage=result['final']['lineage']
        calls=[c for c in result['calls'] if c['role'] in ('revision','korean')]
        out.update(protocol=result['completion']['no_protocol_error'],all_done=result['completion']['all_stages_STOP_done'],
                   planning=a1_planning(prepared,result,semantic_review()))
    else:
        doc=restore_document(result['final']);lineage=result.get('lineage',prepared['policy_row']['normalization']['lineage'])
        calls=[c for s in result.get('stages',[]) for c in s['calls']]
        out.update(protocol=result['no_protocol_error'],all_done=result.get('all_done'))
    out['actions']=dict(Counter(c.get('action',{}).get('action','NOT_EXECUTED') for c in calls)) if condition!='a3' else result['actions']
    out['invalid_actions']=sum(not c.get('action',{}).get('valid',False) for c in calls)
    if condition in ('a1','a2'):out['final_whole_korean']=final_korean_changes(prepared,result,condition)
    source=restore_document(prepared['reward_source']);initial=restore_document(prepared['reward_initial'])
    out['alignment']=result.get('alignment') if condition=='a3' else None
    if condition=='a3' and out['alignment']:
        before={u.sid:u.text for u in restore_document(result['initial']).units}
        after={u.sid:u.text for u in doc.units}
        out['inferred_changes']={'edited_aligned_sentences':sum(before[s]!=after[s] for s in before.keys() & after.keys()),
            'unmatched_originals':len(out['alignment']['deleted_or_unmatched_originals']),
            'unmatched_revisions':len(out['alignment']['inserted_or_unmatched_revisions']),
            'order_inversion_participants':sum(m['order_inversion'] for m in out['alignment']['matches']),
            'inferred_not_an_observed_tool_trace':True,'uncertain':out['alignment']['alignment_failed']}
    try:
        if condition=='a3' and (not result.get('alignment') or result['alignment']['alignment_failed']):
            raise ProjectionUnknown('onepass_alignment_failure_or_missing')
        final=private_document(project_document(doc,prepared['policy_row']['normalization'],lineage),prepared['stable_id_aliases'])
        rec=recover_records(source,final,row['records'],corrupted=initial)
        out['recovery']=rec;out['R_over']=overedit(source,initial,final,row['records'])
        if condition=='a1':
            found={r['record_id']:r for r in out['planning']['records']}
            out['unrestored']=[]
            for record,r in zip(row['records'],rec):
                if r['main']>=1:continue
                in_scope=found[r['record_id']]['in_explicit_delegated_scope']
                changed=record_changed(record,initial,final)
                out['unrestored'].append({'record_id':r['record_id'],'operator':r['op'],'main_recovery':r['main'],
                    'bin':'not_in_delegated_scope' if not in_scope else 'changed_not_restored' if changed else 'in_scope_unchanged',
                    'scope_is_explicit_orchestrator_delegation':True,
                    'in_any_scope_including_final_korean':bool(result['stages']) and any(s['role']=='korean' for s in result['stages']),
                    'record_changed':changed})
        out['projection_status']='exact' if condition!='a3' else 'public_alignment_inferred'
    except (ProjectionUnknown,KeyError) as exc:
        out.update(projection_status='unknown',projection_reason=str(exc),recovery=None,R_over=None)
    return out


def summarize(rows):
    result={}
    for condition in ('a1','a2','a3'):
        cases=[r for r in rows if r['condition']==condition];ops={}
        for op in OPERATORS:
            episodes=[r for r in cases if op in r['operators']]
            records=[x for r in episodes for x in (r['recovery'] or []) if x['op']==op]
            stats={'episodes':len(episodes),'measurable_records':len(records),'fully_restored':sum(x['main']==1 for x in records),
                'mean_main':sum(x['main'] for x in records)/len(records) if records else None,
                'mean_combined_recovery':sum(x['recovery'] for x in records)/len(records) if records else None,
                'protocol_complete':sum(r['protocol'] for r in episodes),
                'actions':dict(sum((Counter(r['actions']) for r in episodes),Counter())),
                'mean_R_over':sum(r['R_over']['value'] for r in episodes if r['R_over'] is not None)/sum(r['R_over'] is not None for r in episodes)
                    if any(r['R_over'] is not None for r in episodes) else None,
                'unknown_episodes':sum(r['recovery'] is None for r in episodes)}
            if condition=='a1':
                oracle=[x for r in episodes for x in r['planning']['records'] if x['operator']==op]
                stats.update(orchestrator_recall=sum(x['matching_problem_type'] for x in oracle),record_count=len(oracle),
                    orchestrator_recall_ambiguous=sum(x['ambiguous_problem_type_only'] for x in oracle),
                    scope_coverage=sum(x['in_explicit_delegated_scope'] for x in oracle),
                    unrestored_bins=dict(Counter(x['bin'] for r in episodes for x in r.get('unrestored',[]) if x['operator']==op)),
                    MOVE_destination_scope_rejections=sum(op in x['operators'] for r in episodes for x in r['planning']['destination_scope_rejections']))
            ops[op]=stats
        result[condition]={'episodes':len(cases),'operators':ops,'protocol_complete':sum(r['protocol'] for r in cases),
            'projection_unknown':sum(r['recovery'] is None for r in cases)}
        if condition in ('a1','a2'):
            result[condition]['final_whole_korean']={op:{
                'records':sum(x['operator']==op for r in cases for x in r['final_whole_korean']),
                'sentence_edited':sum(x['operator']==op and x['sentence_edited'] for r in cases for x in r['final_whole_korean']),
                'position_edited':sum(x['operator']==op and x['position_edited'] for r in cases for x in r['final_whole_korean']),
                'position_text_changed':sum(x['operator']==op and x['position_text_changed'] for r in cases for x in r['final_whole_korean']),
                'position_unknown':sum(x['operator']==op and x['position_unknown'] for r in cases for x in r['final_whole_korean'])}
                for op in ('L_CONN','L_CONJ')}
        if condition=='a1':
            ds=[d for r in cases for d in r['planning']['delegations']]
            result[condition]['delegations']={'total':len(ds),'touching_no_record':sum(d['touches_no_record'] for d in ds),
                'no_record_is_not_an_error':True,'unclassified_items':sum(len(r['planning']['unclassified_items']) for r in cases),
                'ambiguous_items':sum(len(r['planning']['ambiguous_items']) for r in cases),
                'MOVE_scope_rejections_unattributed':sum(not x['operators'] for r in cases for x in r['planning']['destination_scope_rejections'])}
    return result


def report_a(*,refresh=False):
    if (ROOT/'A/report_complete.json').exists() and not refresh:return read_json(ROOT/'A/metrics.json')
    started=time.monotonic();cases=read_json(ROOT/'A/manifest.json')['cases']
    rows=[measure_case(c,condition) for condition in ('a1','a2','a3') for c in cases]
    stats=summarize(rows);write_json(ROOT/'A/record_audit.json',rows);write_json(ROOT/'A/metrics.json',stats)
    lines=['dev100; each condition uses the same 100 corrupted inputs. Restoration is the frozen v1 record main term; '
        'coupled-weighted recovery is retained in the JSON. Unknown projections/alignment are excluded from the measurable denominator and disclosed. '
        'R_over is the unchanged morpheme/order cost on unreferenced sentences. Protocol completion counts reasoned model STOP(blocked) as valid, '
        'but not controller stops, step/context limits or invalid endings. One-pass completion means an untruncated text response; it has no STOP action.',
        '', 'a1 is unchanged full v4.6. a2 gives one record/site/template delegation, structural first, form second, then a final whole-essay Korean pass. '
        'Oracle MOVE targets are restricted to the record; destination anchors may be anywhere in current public text. No clean form or original destination is shown. '
        'a2 reuses a1’s original public MAP; it does not ask an Orchestrator to find damage.', '',
        '|Condition|Operator|Full restoration / measurable|Mean main|R_over|Actions per episode|Protocol / episodes|Unknown|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    f=lambda x:'n/a' if x is None else f'{x:.4f}'
    for condition,v in stats.items():
        for op,s in v['operators'].items():
            lines.append(f'|{condition}|{op}|{s["fully_restored"]}/{s["measurable_records"]}|{f(s["mean_main"])}|{f(s["mean_R_over"])}|'
                f'{f(sum(s["actions"].values())/s["episodes"] if s["episodes"] else None)}|{s["protocol_complete"]}/{s["episodes"]}|{s["unknown_episodes"]}|')
    lines+=['','Action and R_over rows are episode means among essays containing an operator; multi-record episodes can appear in several rows. '
        'a3 counts one text-generation call, not an editor action; its inferred changed/unmatched sentences and order inversions are saved separately.',
        '', '|a1 operator|Matching-type recall / records|Additional ambiguous type matches|Scope coverage|No scope|In scope unchanged|Changed not restored|MOVE destination-scope rejections|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for op,s in stats['a1']['operators'].items():
        b=s['unrestored_bins'];lines.append(f'|{op}|{s["orchestrator_recall"]}/{s["record_count"]}|{s["orchestrator_recall_ambiguous"]}|{s["scope_coverage"]}|'
            f'{b.get("not_in_delegated_scope",0)}|{b.get("in_scope_unchanged",0)}|{b.get("changed_not_restored",0)}|{s["MOVE_destination_scope_rejections"]}|')
    d=stats['a1']['delegations']
    lines+=['',f'Delegations touching no record: {d["touching_no_record"]}/{d["total"]}; these are not counted as errors. '
        'Problem type uses a post-hoc Codex semantic review of all 134 unique task texts (owner, action, instruction only; no record labels or repair outcomes). '
        'This is AI annotation, not a human standard. Initial lexical labels were retained but not used for reported recall: they confused '
        'prohibitions on inventing facts with polarity and missed spacing instructions written as “띄어 쓴다”. '
        f'{d["unclassified_items"]} item occurrences have no definite nine-operator type, and {d["ambiguous_items"]} have possible types. '
        'Definite and ambiguous record matches are separated; useful tasks outside these nine families are not errors. '
        'The complete review and rationales are in `A/problem_type_review.json`. '
        'Primary scope bins use explicit Orchestrator assignments, not the mandatory whole-essay Korean fallback. '
        'All-scope coverage including that pass is retained per record. MOVE rejection counts attribute target overlap, with co-occurring operators explicitly nonexclusive.',
        '', '|Final whole-essay Korean pass|Operator|Edit span covers record position / records|Text at record position changed|Any edit in record sentence|Position mapping unknown|',
        '|---|---|---:|---:|---:|---:|']
    for condition in ('a1','a2'):
        for op,s in stats[condition]['final_whole_korean'].items():
            lines.append(f'|{condition}|{op}|{s["position_edited"]}/{s["records"]}|{s["position_text_changed"]}|{s["sentence_edited"]}|{s["position_unknown"]}|')
    lines+=['','Final whole-essay Korean edits in L_CONN/L_CONJ record sentences (site overlap is reported separately):',
        '', '|Condition|Essay|Operator|Sentence|Old → new|Edit span covers site|Text at site changed|', '|---|---|---|---|---|---|---|']
    def cell(text):return str(text).replace('|','\\|').replace('\n',' ')
    for row in rows:
        for record in row.get('final_whole_korean',[]):
            for edit in record['edits']:
                lines.append('|'+ '|'.join(map(cell,[row['condition'],row['case_id'],record['operator'],edit['sid'],
                    edit['old']+' → '+edit['new'],edit['edits_record_position'],edit['record_position_text_changed']]))+'|')
    lines+=['','Actual position changes use character-diff overlap with the mapped record site; whole-span replacement alone does not imply the marker changed. '
        'This is a surface-change audit, not a claim that the relation category changed or was repaired. Every complete before/after sentence, '
        'site mapping and overlapping diff is preserved in `verak/v5/outputs/pipe1/A/record_audit.json`.']
    seconds=sum(read_json(ROOT/'A'/c/'complete.json')['wall_seconds'] for c in ('a1','a2','a3'))
    append_report('A — dev100 diagnosis','\n'.join(lines),seconds=seconds)
    write_json(ROOT/'A/report_complete.json',{'measurement_wall_seconds':time.monotonic()-started,'reported_inference_seconds':seconds})
    return stats


if __name__=='__main__':report_a()
