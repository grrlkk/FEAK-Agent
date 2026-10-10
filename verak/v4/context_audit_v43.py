"""File/tokenizer-only migration probes of archived context-overflow states."""
from copy import deepcopy
import json
from pathlib import Path
import re

from .common import REPO, file_sha, read_json, sha_text, write_json
from .render_v43 import render, assert_rendered

PROFILE='[Korean document profile: 한국어 문서 프로필; 생략·DEP는 위치 힌트]\n'
SID=re.compile(r'((?:S|N)\d+[a-z]*)\s*(?:\| (.*))?$')
PID=re.compile(r'\[(P\d+)\]$')


def materialize(messages,actions,layout):
    """Apply full/delta public paragraph views; verify exact final layout units.

    Raw morphological outputs are never synthesized. The current public profile
    preserves every last-observed sentence annotation verbatim, including empty
    flag sets. Removed IDs disappear only after a saved removal/update.
    """
    paragraphs={};facts={};order=[];dominant=None;metadata={};notices=[];handoff=None
    updates=0;latest_full=None
    for number,message in enumerate(messages):
        if message['role']!='user': continue
        content=message['content'];lines=content.splitlines()
        for line in lines:
            for prefix,key in (('[문항] ','question'),('[장르] ','genre'),('[루브릭] ','rubric'),
                               ('[남은 예산] ','budget')):
                if line.startswith(prefix): metadata[key]=line[len(prefix):]
            if line.startswith('[문단 순서] '): order=line[len('[문단 순서] '):].split(' → ')
        marker='[marker-change notices: 표지 변화 알림]\n'
        if marker in content:
            block=content.split(marker,1)[1].split('\n[',1)[0]
            if block!='없음': notices.extend(block.splitlines())
        for line in lines:
            if line.startswith(('[실행 오류] ','[단계 종료] ','[JSON 형식 오류] ')):
                notices.append(line)
        marker='[GLOBAL 인계: 행동 및 누적 marker-change notices]\n'
        if marker in content:
            saved=json.loads(content.split(marker,1)[1].split('\n[',1)[0])
            if handoff is not None and handoff!=saved: raise ValueError('Archived handoff changed')
            handoff=saved
        removed=[]
        for line in lines:
            if line.startswith('[제거된 문장] '): removed+=line[len('[제거된 문장] '):].split(', ')
        if '[글]\n' in content:
            full='[전체 갱신]' in content
            if full: paragraphs={};facts={};latest_full=number
            elif latest_full is None: raise ValueError('Partial state without a full snapshot')
            block=content.split('[글]\n',1)[1].split(PROFILE,1)[0]
            observed={};pid=None
            for line in block.rstrip('\n').splitlines():
                match=PID.fullmatch(line)
                if match: pid=match[1];observed[pid]=[];continue
                match=SID.fullmatch(line)
                if not match or match[2] is None or pid is None:
                    raise ValueError('Unrecognized archived text line: '+line)
                observed[pid].append({'id':match[1],'text':match[2]})
            affected=order if full else next((line[len('[갱신 문단] '):].split(', ')
                    for line in lines if line.startswith('[갱신 문단] ')),[])
            for pid in affected: paragraphs[pid]=observed.get(pid,[])
            # Neighbour context is an exact current sentence, but never an
            # instruction to replace that neighbour's whole paragraph.
            for pid,units in observed.items():
                if pid in affected: continue
                known={u['id']:u['text'] for u in paragraphs.get(pid,[])}
                if any(known.get(u['id'])!=u['text'] for u in units):
                    raise ValueError('Neighbour text changed without paragraph replacement')
            updates+=1
        if PROFILE in content:
            for line in content.split(PROFILE,1)[1].splitlines():
                if line.startswith('[주문체:'): dominant=line;continue
                if PID.fullmatch(line): continue
                match=SID.fullmatch(line)
                if not match: raise ValueError('Unrecognized archived profile line: '+line)
                facts[match[1]]=line
        for sid in removed:
            facts.pop(sid,None)
            for pid in paragraphs: paragraphs[pid]=[u for u in paragraphs[pid] if u['id']!=sid]
    if not order or dominant is None: raise ValueError('Incomplete archived state')
    observed=[{'id':pid,'sentences':paragraphs[pid]} for pid in order]
    expected=[{'id':p['pid'],'sentences':[{'id':u['sid'],'text':u['text']} for u in p['units']]}
              for p in layout['paragraphs']]
    if observed!=expected: raise ValueError('Materialized current IDs/text/order differ from saved layout')
    ids=[s['id'] for p in observed for s in p['sentences']]
    facts={sid:line for sid,line in facts.items() if sid in ids}
    if set(facts)!=set(ids): raise ValueError('Missing current public profile annotations')
    profile='\n'.join([dominant]+[line for p in observed
        for line in ['['+p['id']+']']+[facts[s['id']] for s in p['sentences']]])
    journal=[{'action':a['action'],'args':deepcopy(a['args']),'valid':a['valid']} for a in actions]
    payload={**metadata,'paragraphs':observed,'korean_markers':profile,
        'rendering':{'gaps':layout['gaps'],'tail':layout['tail'],
                     'leading':[[u['sid'],u['leading']] for p in layout['paragraphs'] for u in p['units']]},
        'work_journal':journal,'notices':notices}
    if handoff is not None: payload['archived_handoff']=handoff
    text=''.join(gap+''.join(u['leading']+u['text'] for u in p['units'])
                 for gap,p in zip(layout['gaps'],layout['paragraphs']))+layout['tail']
    return payload,{'latest_full_message':latest_full,'applied_full_or_partial_views':updates,
        'sentences':len(ids),'profile_annotations':len(facts),'current_text_sha256':sha_text(text),
        'current_profile_sha256':sha_text(profile),'IDs_text_order_equal_saved_layout':True,
        'journal_entries':len(journal),'handoff_present':handoff is not None,'notice_occurrences':len(notices)}


def audit(tokenizer,*,output_root,system_prompts):
    index=REPO/'verak/v3/outputs/phase8_rft1/context_audit.json'
    design=read_json(index);cases=[]
    for source,essay in design['essays'].items():
        for condition,condition_data in essay['conditions'].items():
            path=Path(condition_data['path'])
            if file_sha(path)!=condition_data['sha256']: raise ValueError('Archived v1 result changed')
            episode=read_json(path)
            for role,messages in episode['messages_by_role'].items():
                layout=episode['stage1_layout'] if role=='global' and episode['stage1_layout'] else episode['final_layout']
                payload,proof=materialize(messages,episode['actions_by_role'][role],layout)
                policy_role='revision' if role=='global' else role
                rendered=render(payload,system_prompts[policy_role])
                if assert_rendered(rendered)!=payload: raise ValueError('Archived renderer equivalence failed')
                length=len(tokenizer.apply_chat_template(rendered,tokenize=True,add_generation_prompt=True))
                if length>7168: raise ValueError(f'Untruncated current state still exceeds limit: {source}/{condition}/{role}: {length}')
                case={'source_id':source,'condition':condition,'role':role,'episode_path':str(path),
                    'episode_sha256':file_sha(path),'archived_episode_completed':episode['completed'],
                    'old_mandatory_plus_last_tokens':condition_data['histories'][role]['mandatory_plus_last_observation_tokens'],
                    'new_prefix_tokens':length,'with_1024_output_reserve':length+1024,
                    'within_7168_prefix_and_8192_context':True,'equivalence':proof,
                    'messages':rendered}
                cases.append(case)
    result={'status':'complete','unique_essays':len(design['essays']),'histories':len(cases),
        'archived_failing_episodes':sum(not c['archived_episode_completed'] for c in cases),
        'cases':cases,'maximum_prefix_tokens':max(c['new_prefix_tokens'] for c in cases),
        'no_essay_or_profile_truncation':True,'no_v1_mutation':True,'context_limit':8192,
        'output_reserve':1024,'scorer_calls':0,'api_calls':0,'GPU_used':False,
        'source_audit_sha256':file_sha(index)}
    write_json(Path(output_root)/'context_audit.json',result)
    return result
