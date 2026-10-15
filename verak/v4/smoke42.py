"""One paired v4.2 attempt on the frozen v4.1 B1 cohort, with no scale launch."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
import time

from .common import (REPO,atomic_new,collection_lock,constrain_cpu,file_sha,load_config,
                     read_json,safe_id,sha_text,write_json)
from . import policy_prompts_v42 as prompts
from .environment_v42 import (BoundaryParagraphs,ENVIRONMENT_VERSION,RULES,
                              normalize_row,sentence_spans)
from .runtime_v42 import VersionedAPI,teach,export
from .smoke41 import action_metrics

SCALE=REPO/'verak/v4/outputs/scale2'
ROOT=SCALE/'B1'
PRIOR=REPO/'verak/v4/outputs/scale/B1'
BASE_COMMIT='7ed07f67072aa410819815cc808a2c3295355126'
PROTECTED=('policy_prompts.py','prompts/v4_editors.json','policy_env.py','content_env.py',
    'prep2_content.py','prep3_content.py','prep3_content_report.py','prep3_common.py',
    'policy_prompts_v41.py','prompts/v41_editors.json','runtime_v41.py','smoke41.py','smoke41_report.py')


def immutable(path,value):
    if path.exists():
        if read_json(path)!=value: raise ValueError(f'Frozen v4.2 artifact changed: {path}')
    else: atomic_new(path,value)
    return value


def protected_hashes():
    base=Path(__file__).parent
    return {str(base/name):file_sha(base/name) for name in PROTECTED}|{
        str(REPO/'imple/FEAK_AGENT_METHOD.md'):file_sha(REPO/'imple/FEAK_AGENT_METHOD.md')}


def reference_files():
    files=[p for p in PRIOR.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.md'}]
    files.append(PRIOR/'api/ledger.sqlite')
    return {str(p):file_sha(p) for p in sorted(files)}


def freeze_contract():
    manifest=prompts.verify_frozen();prior=read_json(PRIOR/'metrics.json')
    if prior['api']['pending'] or prior['api']['reserved_usd']:
        raise ValueError('Prior smoke must remain fully settled')
    budget={'content_cumulative_cap_usd':50.,'prior_v41_confirmed_usd':prior['api']['confirmed_usd'],
        'prior_complete_sha256':file_sha(PRIOR/'complete.json'),'v42_cap_usd':2.,
        'v42_ledger':str(ROOT/'api/ledger.sqlite'),
        'B2_remaining_cap_rule':'50 - v4.1 B1 confirmed - v4.2 B1 confirmed - v4.2 B1 reserved',
        'downstream_requires_all20_v42_gate_pass':True}
    immutable(ROOT/'budget_contract.json',budget)
    value={'version':'v4.2_B1_paired_smoke','base_commit':BASE_COMMIT,'prompt_version':prompts.VERSION,
        'environment_version':ENVIRONMENT_VERSION,'prompt_manifest_sha256':file_sha(prompts.FROZEN),
        'prompts':manifest,'environment_rules':RULES,'protected_files':protected_hashes(),
        'authorization':'Latest user Run: isolated v4.2 environment/prompts and same20 paired smoke, then gate',
        'sources':20,'attempts_per_source':1,'sample_sha256':file_sha(PRIOR/'sample.json'),
        'reference_final_complete_sha256':file_sha(PRIOR/'final_complete.json'),
        'budget_contract_sha256':file_sha(ROOT/'budget_contract.json'),
        'primary_gate':'valid returned action records / all returned action records >=0.90',
        'all20_required':True,'malformed_JSON_and_environment_invalid_in_denominator':True,
        'API_incomplete_errors_separate_and_all_scheduled_denominator_reported':True,
        'planner':'reuse all20 saved Sol Dv3 plans; only ID expansion and correspondence note',
        'new_planner_calls':0,'teacher':'pinned Luna low; independent unseeded requests',
        'revision_steps_per_delegation':6,'korean_whole_essay_steps':14,
        'output_limit':1024,'context_limit':8192,'prefix_limit':7168,'workers':1,
        'quality_judging':False,'training':False,'GPU_used':False,'new_scale_or_Wiki_calls':False}
    immutable(ROOT/'contract.json',value)
    path=ROOT/'reference_files.json'
    if not path.exists(): atomic_new(path,reference_files())
    for name,digest in read_json(path).items():
        if file_sha(name)!=digest: raise ValueError(f'Prior B1 artifact changed: {name}')
    return value


def remap_plan(original,row,*,path=None):
    """Keep tasks/evidence exactly; expand IDs covering the same source characters."""
    if original['status']!='completed': raise ValueError('Paired smoke requires the original completed planner')
    value=deepcopy(original);mapping=row['normalization']['sid_mapping'];audits=[]
    locations={p['id']:' '.join(s['text'] for s in p['sentences']) for p in row['paragraphs']}
    locations.update({s['id']:s['text'] for p in row['paragraphs'] for s in p['sentences']})
    for old,item in zip(original['items'],value['items']):
        expanded=[]
        for loc in old['location']:
            expanded.extend(mapping.get(loc,[loc]))
        if len(expanded)!=len(set(expanded)) or any(x not in locations for x in expanded):
            raise ValueError('Planner location expansion is not exact and unique')
        item['location']=expanded
        named=set(old['location'])|set(re.findall(r'(?<![A-Za-z0-9])S\d+[a-z]*(?![A-Za-z0-9])',old['instruction']))
        changed={sid:mapping[sid] for sid in mapping if sid in named and mapping[sid]!=[sid]}
        if changed:
            note='; '.join(f'원래 {sid} 전체는 현재 '+', '.join(ids)+'이다' for sid,ids in changed.items())
            item['instruction']=old['instruction']+'\n문장 ID 대응: '+note+'. 원래 문장 앞/뒤는 해당 범위 앞/뒤를 뜻한다.'
        evidence=''.join(old['evidence'].split())
        exact=evidence in ''.join(''.join(locations[x].split()) for x in expanded)
        if not exact: raise ValueError('Original planner evidence is not preserved after normalization')
        if any(item[k]!=v for k,v in old.items() if k not in {'location','instruction'}):
            raise ValueError('Planner task semantics changed')
        if not item['instruction'].startswith(old['instruction']): raise ValueError('Original instruction changed')
        audits.append({'item_id':item['item_id'],'original_location':old['location'],
            'mapped_location':expanded,'correspondence':changed,'exact_evidence_preserved':exact,
            'original_instruction_preserved':True})
    value['v42_reuse']={'original_path':str(path) if path else None,
        'original_sha256':file_sha(path) if path else sha_text(json.dumps(original,ensure_ascii=False,sort_keys=True)),
        'original_phase_call':original['phase_call'],'new_planner_calls':0,
        'item_count_unchanged':len(value['items'])==len(original['items']),'audits':audits}
    return value


def spacing_regression(rows):
    by_source={r['source_id']:r for r in rows};old=read_json(PRIOR/'diagnostics.json');checks=[]
    for group in old['rejected_EDIT_groups']:
        if group['category']!='pure_spacing_boundary_rejection': continue
        row=by_source[group['source_id']];parent=group['sentence'];ids=row['normalization']['sid_mapping'][parent]
        lookup={s['id']:s['text'] for p in row['paragraphs'] for s in p['sentences']}
        normalized=' '.join(lookup[s] for s in ids)
        # The archived old/new pair differs only in whitespace. Its inserted
        # post-terminal space must already exist at the prepared boundary.
        fixed=group['new'] in normalized and group['old'] not in normalized
        checks.append({'source_id':row['source_id'],'original_sid':parent,'normalized_ids':ids,
            'old':group['old'],'new':group['new'],'occurrences':len(group['occurrences']),
            'fixed_in_initial_state':fixed,'counted_as_agent_action':False})
    if sum(c['occurrences'] for c in checks)!=13 or not all(c['fixed_in_initial_state'] for c in checks):
        raise ValueError('The13 archived whitespace-boundary regressions were not all normalized initially')
    return {'archived_rejections':13,'distinct_variants':len(checks),'all_fixed_before_actions':True,'checks':checks}


def prepare():
    constrain_cpu();freeze_contract()
    from verak.v3.v2_ops.local import load_environment
    config=load_config();load_environment(config)
    analyzer=BoundaryParagraphs(config,cache_dir=ROOT/'bareun_sources')
    sample=read_json(PRIOR/'sample.json')
    immutable(ROOT/'sample.json',{**sample,'paired_reference_path':str(PRIOR/'sample.json'),
        'paired_reference_sha256':file_sha(PRIOR/'sample.json'),'version':prompts.VERSION,'resampled':False})
    rows=[];files={};plans={}
    for n,source in enumerate(sample['source_ids'],1):
        source_path=PRIOR/'essays'/(safe_id(source)+'.json');path=ROOT/'essays'/(safe_id(source)+'.json')
        if path.exists(): row=read_json(path)
        else:
            row=normalize_row(read_json(source_path),analyzer)
            row.update(original_source_path=str(source_path),original_source_sha256=file_sha(source_path),
                normalized_essay_hash=sha_text(row['text']))
            atomic_new(path,row)
        rows.append(row);files[source]={'path':str(path),'sha256':file_sha(path)}
        original_plan=PRIOR/'content/items'/(safe_id(source)+'.json')
        plan=remap_plan(read_json(original_plan),row,path=original_plan)
        plan_path=ROOT/'content/items'/(safe_id(source)+'.json');immutable(plan_path,plan)
        plans[source]={'path':str(plan_path),'sha256':file_sha(plan_path),'reuse':plan['v42_reuse']}
        write_json(ROOT/'status.json',{'stage':'preparing','processed':n,'planned':20,'paid_calls':0,'at':time.time()})
    immutable(ROOT/'source_files.json',files);immutable(ROOT/'planner_reuse.json',plans)
    regression=spacing_regression(rows);immutable(ROOT/'spacing_regression.json',regression)
    summary={'sources':len(rows),'same_source_order':sample['source_ids']==[r['source_id'] for r in rows],
        'all_original_nonwhitespace_preserved':all(r['normalization']['source_characters_preserved_except_whitespace'] for r in rows),
        'rendered_text_changed':sum(r['text']!=r['original_text'] for r in rows),
        'original_units_split':sum(len(ids)>1 for r in rows for ids in r['normalization']['sid_mapping'].values()),
        'derived_initial_units':sum(len(ids)-1 for r in rows for ids in r['normalization']['sid_mapping'].values()),
        'plans_reused':len(plans),'tasks':sum(len(p['reuse']['audits']) for p in plans.values()),
        'mapped_tasks':sum(bool(a['correspondence']) for p in plans.values() for a in p['reuse']['audits']),
        'spacing_regression':regression,'prompt_manifest_sha256':file_sha(prompts.FROZEN),
        'environment_sha256':prompts.environment_hashes(),'new_planner_calls':0,'paid_calls':0,'gpu_used':False}
    immutable(ROOT/'prepared.json',summary);write_json(ROOT/'status.json',{'stage':'prepared',**summary})
    return rows


def accounting():
    path=ROOT/'api/ledger.sqlite'
    if not path.exists(): return {'calls':0,'confirmed_usd':0.,'reserved_usd':0.,'pending':0,'by_stage':{}}
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        rows=db.execute('SELECT stage,item_id,status,reserved,confirmed FROM calls').fetchall()
    return {'calls':sum(r[2]!='blocked_before_send' for r in rows),'confirmed_usd':sum(r[4] for r in rows),
        'reserved_usd':sum(r[3] for r in rows),'pending':sum(r[2]=='pending' for r in rows),
        'by_stage':{s:{'requests':sum(r[0]==s and r[2]!='blocked_before_send' for r in rows),
            'logical_requests':len({r[1] for r in rows if r[0]==s and r[2]!='blocked_before_send'}),
            'statuses':dict(Counter(r[2] for r in rows if r[0]==s)),
            'confirmed_usd':sum(r[4] for r in rows if r[0]==s)} for s in sorted({r[0] for r in rows})}}


def api_for():
    from verak.v3.v2_ops.local import load_environment
    freeze_contract();config=load_config();load_environment(config)
    luna=read_json(config['paths']['phase4_output']/'models.json')['luna_model'];phase='v42_B1_smoke'
    config[phase]={'model':luna,'max_cost_usd':2.,'max_concurrent_requests':1,'phase_api_ceiling':1200}
    config['paths'][phase+'_output']=ROOT
    api=VersionedAPI(config,1200,phase=phase);api.allowed_models={luna}
    return api


def run():
    constrain_cpu()
    with collection_lock(ROOT):
        if (ROOT/'complete.json').exists(): return read_json(ROOT/'complete.json')
        if not (ROOT/'prepared.json').exists(): raise ValueError('Finish and review prepare before any paid call')
        freeze_contract();sources=read_json(ROOT/'sample.json')['source_ids'];files=read_json(ROOT/'source_files.json')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True)
        luna=api_for()
        try:
            luna.settle_interrupted()
            for n,source in enumerate(sources,1):
                file=files[source]
                if file_sha(file['path'])!=file['sha256']: raise ValueError('Frozen prepared source changed')
                row=read_json(file['path']);plan_path=ROOT/'content/items'/(safe_id(source)+'.json')
                if file_sha(plan_path)!=read_json(ROOT/'planner_reuse.json')[source]['sha256']:
                    raise ValueError('Frozen mapped planner changed')
                attempt=teach(row,read_json(plan_path),1,luna,tokenizer,output_root=ROOT)
                write_json(ROOT/'status.json',{'stage':'smoke','processed':n,'planned':20,'last_source':source,
                    'last_status':attempt['status'],'api':accounting(),'at':time.time()})
            from .smoke42_report import report
            result=report(tokenizer);write_json(ROOT/'status.json',result);return result
        finally: luna.close()


def main():
    import argparse,os,subprocess,sys
    constrain_cpu();parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','prepare','run','launch','report'))
    args=parser.parse_args()
    if args.command=='freeze':
        from transformers import AutoTokenizer
        path=load_config()['paths']['policy_base']
        manifest=prompts.freeze(AutoTokenizer.from_pretrained(str(path),local_files_only=True),path)
        result={'prompt_version':manifest['version'],'roles':{k:{x:v[x] for x in ('sha256','policy_tokens','system_message_tokens')}
            for k,v in manifest['roles'].items()},'environment_sha256':manifest['environment_sha256']}
    elif args.command=='prepare': prepare();result=read_json(ROOT/'prepared.json')
    elif args.command=='run': result=run()
    elif args.command=='report':
        from .smoke42_report import report
        from transformers import AutoTokenizer
        result=report(AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True))
    else:
        if not (ROOT/'prepared.json').exists(): raise ValueError('Prepare must finish before launching paid smoke')
        if (ROOT/'complete.json').exists(): result=read_json(ROOT/'complete.json')
        else:
            marker=ROOT/'launch.json'
            if marker.exists():
                old=read_json(marker);p=Path('/proc')/str(old['pid'])/'cmdline'
                if p.exists() and b'verak.v4.smoke42' in p.read_bytes().split(b'\0'):
                    print(json.dumps(old));return
            with (ROOT/'worker.log').open('a') as log:
                process=subprocess.Popen([sys.executable,'-m','verak.v4.smoke42','run'],cwd=Path(__file__).resolve().parents[2],
                    env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result={'pid':process.pid,'worktree':str(Path(__file__).resolve().parents[2]),
                'command':'python -m verak.v4.smoke42 run','gpu_used':False}
            write_json(marker,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
