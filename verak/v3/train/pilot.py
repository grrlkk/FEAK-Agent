"""Frozen 100-episode teacher pilot, isolated budget, no bulk-generation path."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import random

from ..common import read_json, write_json, file_sha, sha_text
from ..phase2 import read_jsonl
from ..view_data import load_episode_examples
from ..agent.runner import system_prompt, run_episode
from ..env import RevisionEnv
from ..eval.api import Phase6Teacher
from ..eval.resources import Resources

LEVELS = ('L1', 'L2', 'L3', 'L4')


def stratified_order(rows, seed=71):
    rng = random.Random(seed)
    pools = {level: sorted((r['episode_id'] for r in rows if r['level']==level)) for level in LEVELS}
    if sum(map(len,pools.values())) != len(rows):
        raise ValueError('Unexpected curriculum level')
    for pool in pools.values(): rng.shuffle(pool)
    result=[]
    while any(pools.values()):
        levels=[level for level in LEVELS if pools[level]]
        rng.shuffle(levels)
        result.extend(pools[level].pop() for level in levels)
    if len(set(result)) != len(result):
        raise ValueError('Duplicate episode IDs')
    return result


def prepare(config):
    if config['env']['enable_check'] or config['env']['mode'] != 'two_stage':
        raise ValueError('Pilot requires main two_stage without CHECK')
    if config['role_teacher'] != {'model':'gpt-6.1-sol','reasoning_effort':'low'}:
        raise ValueError('Teacher contract changed')
    path=config['paths']['active_corrupt']/'agent_train.jsonl'
    rows=read_jsonl(path)
    if len(rows)!=1573 or any(r['split']!='agent_train' for r in rows):
        raise ValueError('Use exactly the accepted recoverability-filtered agent_train corpus')
    ids=stratified_order(rows,config['phase7_pilot']['seed'])
    lookup={r['episode_id']:r for r in rows}
    selected=ids[:100]
    if Counter(lookup[i]['level'] for i in selected)!=Counter({level:25 for level in LEVELS}):
        raise ValueError('Pilot must contain 25 per level')
    examples={e.id:e for e in load_episode_examples(config,'agent_train')}
    for row in rows:
        if row['source_id'] not in examples:
            raise ValueError('Ineligible or wrong-split corruption source')
        for record in row['records']:
            if record['op']=='G_OFFTOPIC' and record['params']['donor']['source_id'] not in examples:
                raise ValueError('Wrong-split donor')
    report=config['paths']['repo']/'imple/reports/V3_PHASE_6_EXAMPLES.md'
    manifest={'seed':71,'method':'shuffle within level; shuffle active level order every round; no replacement',
        'corpus_path':str(path),'corpus_sha256':file_sha(path),'count':len(rows),
        'full_order':ids,'pilot_ids':selected,'pilot_counts':dict(Counter(lookup[i]['level'] for i in selected)),
        'prompt_sha256':{role:sha_text(system_prompt(role)) for role in ('global','korean')},
        'mode':'two_stage','enable_check':False,'teacher':config['role_teacher'],
        'policy_context_limit':config['policy']['context_limit'],
        'examples_report_sha256':file_sha(report),
        'contract_sha256':sha_text(json.dumps({k:config[k] for k in ('env','reward','similarity','role_teacher','phase7_pilot')},sort_keys=True))}
    target=config['paths']['phase7_pilot_output']/'design.json'
    if target.exists() and read_json(target)!=manifest:
        raise ValueError('Frozen pilot design changed; never silently reuse mismatched results')
    write_json(target,manifest)
    return manifest,lookup,examples


def safe_id(value):
    return value.replace(':','_')


def execute(config,api):
    design,rows,examples=prepare(config)
    output=config['paths']['phase7_pilot_output']
    resources=Resources(config,examples,output_key='phase7_pilot_output')
    def one(id):
        path=output/'episodes'/(safe_id(id)+'.json')
        if path.exists() and read_json(path).get('completed'):
            return read_json(path)
        row=rows[id]
        state=resources.worker()
        episode={k:row[k] for k in ('episode_id','source_id','genre','level','question','records',
                                    'corrupted_score','preexisting_spell_spans')}
        episode.update(source=resources.source(row['source_id']),document=resources.corrupted(row))
        env=RevisionEnv(config,mode='two_stage',analysis=state.analysis,scorer=resources,
                        similarity=resources.similarity,tokenizer=state.tokenizer)
        result=run_episode(env,episode,Phase6Teacher(api,'teacher_pilot'),
                            event_path=output/'events'/(safe_id(id)+'.jsonl'))
        write_json(path,result)
        print(json.dumps({'id':id,'completed':result['completed'],'steps':result['steps'],
              'cost':result['cost_usd'],'error':result.get('runtime_error'),'budget':api.accounting()},ensure_ascii=False),flush=True)
        if not result['completed']:
            raise RuntimeError(str(result['runtime_error']))
        return result
    errors,finished=[],[]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            iterator,pending=iter(design['pilot_ids']),{}
            def fill():
                while len(pending)<4 and not errors:
                    id=next(iterator,None)
                    if id is None:break
                    pending[pool.submit(one,id)]=id
            fill()
            while pending:
                done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    id=pending.pop(future)
                    try:
                        future.result()
                        finished.append(id)
                    except Exception as exc:
                        errors.append({'id':id,'type':type(exc).__name__,'message':str(exc)})
                write_json(output/'progress.json',{'requested':100,'finished':finished,'errors':errors,'budget':api.accounting()})
                fill()
    finally:
        resources.close()
        write_json(output/'run_status.json',{'requested':100,'finished':finished,'errors':errors,'budget':api.accounting()})
    if errors:raise RuntimeError('Pilot interrupted; inspect saved outputs and budget before resuming')
