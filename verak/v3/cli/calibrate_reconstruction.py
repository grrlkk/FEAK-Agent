"""Phase 4 only: shared-budget dual-source reconstruction calibration."""
import argparse
import re
from pathlib import Path

from verak.v3.common import DEFAULT_CONFIG,load_config,read_json
from verak.v3.phase2 import read_jsonl
from verak.v3.reconstruction_data import prepare_cases


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['prepare','local','api','calibrate'],required=True)
    p.add_argument('--max-api-calls',type=int,required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--extra-source',type=Path,action='append',default=[])
    args=p.parse_args()
    if not 0<=args.max_api_calls<=650:
        raise ValueError('Phase 4 API ceiling is 650')
    config=load_config()
    output=config['paths']['phase4_output']
    if args.stage=='prepare':
        cases,manifest=prepare_cases(config,output)
        print({k:manifest[k] for k in ('n','existing_unique_sites','unique_essays','genres')})
        return
    cases=read_jsonl(output/'cases.jsonl')
    if args.stage=='local':
        from verak.v3.reconstruction_local import generate_local
        generate_local(cases,config,output,limit=args.limit)
    elif args.stage=='api':
        from verak.v3.reconstruction_api import ReconstructionAPI,run_requests,summarize_api
        available=read_json(output/'models.json')['luna_model']
        if not available or available!=config['cheap_model']['model']:
            raise ValueError('Config must pin the verified available Luna ID')
        api=ReconstructionAPI(output,args.max_api_calls,cheap_model=available)
        local={r['item_id']:r for r in read_jsonl(output/'local_reconstructions.jsonl')}
        try:
            run_requests(cases,api,local,limit=args.limit)
        finally:
            summarize_api(output)
    else:
        from verak.v3.reward.similarity import calibrate
        reconstructions=read_jsonl(output/'reconstructions.jsonl')
        for path in args.extra_source:
            rows=read_jsonl(path)
            reconstructions.extend({**r,'source':r.get('source','gpt-5-mini')} for r in rows)
        result=calibrate(cases,reconstructions,output)
        # Leave all scorer settings and noise files intact.
        raw=DEFAULT_CONFIG.read_text()
        match=re.search(r'(?ms)^similarity:\n.*?(?=^\S|\Z)',raw)
        if match is None:
            raise ValueError('Missing similarity config block')
        block=match.group()
        for key,value in (('model',result['selected_model']),('tau',result['tau'])):
            block,count=re.subn(r'(?m)^  '+key+r':[^\n]*',f'  {key}: {value}',block)
            if count!=1:
                raise ValueError('Ambiguous similarity setting')
        DEFAULT_CONFIG.write_text(raw[:match.start()]+block+raw[match.end():])
        print({k:result[k] for k in ('selected_model','tau','pairs')})


if __name__=='__main__':
    main()
