"""Isolated v4.3 same-20 smoke; root-reviewed preflight precedes paid calls."""
from copy import deepcopy
import json
from pathlib import Path
import time

from .common import (REPO,atomic_new,collection_lock,constrain_cpu,file_sha,load_config,
                     read_json,safe_id,write_json)
from .runtime_v41 import bind
from . import policy_prompts_v43 as prompts
from .environment_v43 import ENVIRONMENT_VERSION,prepared_document,check_mask_edit
from .runtime_v43 import VersionedAPI,teach,export

ROOT=REPO/'verak/v4/outputs/scale3/B1'
PRIOR41=REPO/'verak/v4/outputs/scale/B1'
PRIOR42=REPO/'verak/v4/outputs/scale2/B1'
BASE_COMMIT='510b2810b6c362272b7f6db5055563e57eff6db8'


def immutable(path,value):
    if path.exists():
        if read_json(path)!=value: raise ValueError('Frozen v4.3 artifact changed: '+str(path))
    else: atomic_new(path,value)
    return value


def prior_files():
    from .smoke42 import PROTECTED
    base=Path(__file__).parent
    code=(*PROTECTED,'environment_v42.py','policy_prompts_v42.py','runtime_v42.py',
          'prompts/v42_editors.json','smoke42.py','smoke42_report.py')
    paths=[base/name for name in code]+[REPO/'imple/FEAK_AGENT_METHOD.md']
    for root in (PRIOR41,PRIOR42):
        paths += [p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.md'}]
        paths.append(root/'api/ledger.sqlite')
    return {str(p):file_sha(p) for p in sorted(set(paths))}


def freeze_contract():
    manifest=prompts.verify_frozen()
    prior={version:read_json(root/'metrics.json') for version,root in [('v41',PRIOR41),('v42',PRIOR42)]}
    if any(p['api']['pending'] or p['api']['reserved_usd'] for p in prior.values()):
        raise ValueError('Prior smoke reservations must be settled')
    cost=sum(p['api']['confirmed_usd'] for p in prior.values())
    budget={'content_cumulative_cap_usd':50.,'prior_v41_and_v42_confirmed_usd':cost,
        'prior_metrics_sha256':{v:file_sha(p/'metrics.json') for v,p in [('v41',PRIOR41),('v42',PRIOR42)]},
        'v43_cap_usd':2.,'v43_ledger':str(ROOT/'api/ledger.sqlite'),
        'B2_remaining_cap_rule':'50 - settled v4.1 B1 - settled v4.2 B1 - v4.3 B1 confirmed/reserved'}
    immutable(ROOT/'budget_contract.json',budget)
    value={'version':'v4.3_B1_same20_smoke','base_commit':BASE_COMMIT,'prompts':manifest,
        'prompt_version':prompts.VERSION,'prompt_manifest_sha256':file_sha(prompts.FROZEN),
        'environment_version':ENVIRONMENT_VERSION,'environment_sha256':prompts.environment_hashes(),
        'sources':20,'attempts_per_source':1,'sample_sha256':file_sha(PRIOR41/'sample.json'),
        'previous_gate_sha256':file_sha(PRIOR42/'gate.json'),
        'budget_contract_sha256':file_sha(ROOT/'budget_contract.json'),
        'primary_gate':'environment-valid executed first actions / all returned action calls >=0.90',
        'all20_required':True,'teacher_only_first_action':True,'inference_multiple_actions_invalid':True,
        'trimmed_is_not_automatically_valid':True,'raw_responses_retained':True,
        'targets':'canonical single executed action only; observations/results/notices masked',
        'new_planner_calls':0,'teacher':'pinned Luna low; independent unseeded requests',
        'revision_steps_per_delegation':6,'korean_whole_essay_steps':14,
        'context_limit':8192,'output_limit':1024,'prefix_limit':7168,'workers':1,
        'SEARCH':'only delegated needs_search=yes; public aliases; same teacher/inference environment',
        'quality_judging':False,'training':False,'GPU_used':False,'scale_calls':False,
        'requires_root_preflight_review':True}
    immutable(ROOT/'contract.json',value)
    path=ROOT/'protected_files.json'
    if not path.exists(): atomic_new(path,prior_files())
    for name,digest in read_json(path).items():
        if file_sha(name)!=digest: raise ValueError('Prior protected artifact changed: '+name)
    return value


def prepare():
    constrain_cpu();freeze_contract()
    sample=read_json(PRIOR41/'sample.json')
    immutable(ROOT/'sample.json',{**sample,'paired_v41_path':str(PRIOR41/'sample.json'),
        'paired_v42_path':str(PRIOR42/'sample.json'),'version':prompts.VERSION,'resampled':False})
    files={};plans={}
    for source in sample['source_ids']:
        old=PRIOR42/'essays'/(safe_id(source)+'.json');row=deepcopy(read_json(old))
        if row['environment_version']!='v4.2_boundaries_20261010': raise ValueError('Wrong paired source version')
        # Boundary rules have not changed since v4.2. Reuse the exact normalized
        # text, IDs, profile, lineage and plan; no analyzer or planner is needed.
        prepared_document(row)
        row['environment_version']=ENVIRONMENT_VERSION
        row['normalization']['version']=ENVIRONMENT_VERSION
        row['v43_source_reuse']={'path':str(old),'sha256':file_sha(old),'new_Bareun_calls':0}
        path=ROOT/'essays'/old.name;immutable(path,row)
        files[source]={'path':str(path),'sha256':file_sha(path),'v42_path':str(old),'v42_sha256':file_sha(old)}
        oldplan=PRIOR42/'content/items'/(safe_id(source)+'.json');plan=deepcopy(read_json(oldplan))
        plan['v43_reuse']={'path':str(oldplan),'sha256':file_sha(oldplan),'items_byte_semantics_unchanged':True}
        target=ROOT/'content/items'/oldplan.name;immutable(target,plan)
        plans[source]={'path':str(target),'sha256':file_sha(target),'v42_path':str(oldplan),'v42_sha256':file_sha(oldplan)}
    immutable(ROOT/'source_files.json',files);immutable(ROOT/'planner_reuse.json',plans)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True)
    from .context_audit_v43 import audit
    context=audit(tokenizer,output_root=ROOT,system_prompts=prompts.PROMPTS)
    # These saved proposals are diagnostic replays only, not new teacher calls.
    mask=[]
    prior=read_json(PRIOR42/'metrics.json')
    for invalid in prior['invalid_records']:
        action=invalid['action'].get('value')
        if invalid['category']!='mask_guard' or not action or action.get('action')!='EDIT': continue
        attempt=read_json(PRIOR42/'content/attempts'/f'{safe_id(invalid["source_id"])}_a1.json')
        call=next(c for c in attempt['calls'] if c['role']==invalid['role'] and c['delegation']==invalid['delegation'] and c['turn']==invalid['turn'])
        observation=json.loads(call['public_messages'][1]['content'])
        before=next(s['text'] for p in observation['paragraphs'] for s in p['sentences'] if s['id']==action['sentence'])
        start=before.index(action['old']);end=start+len(action['old'])
        after=before[:start]+action['new']+before[end:]
        check_mask_edit(before,after,start,end,action['new'])
        mask.append({'source_id':invalid['source_id'],'role':invalid['role'],'turn':invalid['turn'],
            'sentence':action['sentence'],'old':action['old'],'new':action['new'],
            'v42_valid':False,'v43_mask_guard_passes':True,'actual_new_teacher_action':False})
    if len(mask)!=6: raise ValueError('Expected exactly six paired mask-preserving EDIT proposals')
    immutable(ROOT/'mask_regression.json',mask)
    result={'sources':20,'same_source_order':True,'same_v42_normalized_text_profile_IDs':True,
        'plans_reused':20,'tasks':sum(len(read_json(p['path'])['items']) for p in plans.values()),
        'needs_search_tasks':sum(i.get('needs_search','no')=='yes' for p in plans.values() for i in read_json(p['path'])['items']),
        'mask_preserving_rejections_replayed':len(mask),'new_Bareun_calls':0,'new_planner_calls':0,
        'context_histories_verified':context['histories'],'maximum_archived_prefix_tokens':context['maximum_prefix_tokens'],
        'context_audit_sha256':file_sha(ROOT/'context_audit.json'),
        'contract_sha256':file_sha(ROOT/'contract.json'),'paid_calls':0,'gpu_used':False}
    immutable(ROOT/'prepared.json',result);write_json(ROOT/'status.json',{'stage':'prepared',**result})
    return result


def accounting():
    from .smoke42 import accounting as original
    return bind(original,ROOT=ROOT)()


def api_for():
    from verak.v3.v2_ops.local import load_environment
    config=load_config();load_environment(config)
    luna=read_json(config['paths']['phase4_output']/'models.json')['luna_model'];phase='v43_B1_smoke'
    config[phase]={'model':luna,'max_cost_usd':2.,'max_concurrent_requests':1,'phase_api_ceiling':1200}
    config['paths'][phase+'_output']=ROOT
    api=VersionedAPI(config,1200,phase=phase);api.allowed_models={luna}
    return api


def require_review():
    approval=read_json(ROOT/'preflight_approval.json')
    if not approval.get('approved') or any(approval.get(key+'_sha256')!=file_sha(ROOT/(key+'.json'))
            for key in ('contract','prepared','context_audit')):
        raise ValueError('Root preflight review is absent or belongs to a different contract')


def run():
    constrain_cpu()
    with collection_lock(ROOT):
        if (ROOT/'complete.json').exists(): return read_json(ROOT/'complete.json')
        require_review();freeze_contract()
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True)
        luna=api_for()
        try:
            luna.settle_interrupted()
            sources=read_json(ROOT/'sample.json')['source_ids'];files=read_json(ROOT/'source_files.json')
            plans=read_json(ROOT/'planner_reuse.json')
            for n,source in enumerate(sources,1):
                for saved in (files[source],plans[source]):
                    if file_sha(saved['path'])!=saved['sha256']: raise ValueError('Frozen source/plan changed')
                attempt=teach(read_json(files[source]['path']),read_json(plans[source]['path']),1,luna,tokenizer,output_root=ROOT)
                write_json(ROOT/'status.json',{'stage':'smoke','processed':n,'planned':20,
                    'last_source':source,'last_status':attempt['status'],'api':accounting(),'at':time.time()})
            from .smoke43_report import report
            result=report(tokenizer);write_json(ROOT/'status.json',result);return result
        finally: luna.close()


def main():
    import argparse,os,subprocess,sys
    constrain_cpu();parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','prepare','run','launch','report'));args=parser.parse_args()
    if args.command=='freeze':
        from transformers import AutoTokenizer
        path=load_config()['paths']['policy_base']
        result=prompts.freeze(AutoTokenizer.from_pretrained(str(path),local_files_only=True),path)
    elif args.command=='prepare': result=prepare()
    elif args.command=='run': result=run()
    elif args.command=='report':
        from .smoke43_report import report
        from transformers import AutoTokenizer
        result=report(AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True))
    else:
        require_review()
        if (ROOT/'complete.json').exists(): result=read_json(ROOT/'complete.json')
        else:
            marker=ROOT/'launch.json'
            if marker.exists():
                old=read_json(marker);process=Path('/proc')/str(old['pid'])/'cmdline'
                if process.exists() and b'verak.v4.smoke43' in process.read_bytes().split(b'\0'):
                    print(json.dumps(old));return
            with (ROOT/'worker.log').open('a') as log:
                process=subprocess.Popen([sys.executable,'-m','verak.v4.smoke43','run'],cwd=Path(__file__).resolve().parents[2],
                    env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result={'pid':process.pid,'worktree':str(Path(__file__).resolve().parents[2]),
                'command':'python -m verak.v4.smoke43 run','gpu_used':False}
            write_json(marker,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
