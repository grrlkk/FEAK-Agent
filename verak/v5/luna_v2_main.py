"""Resumeable main experiment. Human preparation gate cannot self-approve."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sqlite3
import time
import random

from verak.v4.common import read_json, write_json, file_sha, REPO
from .luna_v2_data import ROOT, CONDITIONS
from .luna_v2_env import Environment
from .luna_v2_run import Episode, score, p1
from .luna_v2_eval import feedback_prompt, essay_prompt, edit_cases, anonymized, validate_feedback, validate_essay, validate_link
from .luna_v2_batch import Batch


def config():
    value = read_json(ROOT/'frozen_config.json')
    review = read_json(ROOT/'human_preparation_review.json')
    if review.get('evaluation_50_authorized') is False:
        raise PermissionError('The latest user instruction explicitly withholds evaluation50; preparation only.')
    review_file = REPO/'reports/luna_upper_bound_prep_review.md'
    if review.get('approved') is not True or review.get('artifact_sha256') != file_sha(review_file):
        raise ValueError('v2 §5.3 requires an actual human review of the complete episode')
    if value['missing_phi'] or value['delta_sample_count'] < 8 or value['modification_limit'] is None:
        raise ValueError('Preparation gates incomplete')
    if file_sha(ROOT/'selection.json') != value['selection_sha256']: raise ValueError('Frozen sample changed')
    # Require the preregistration to have been committed before evaluation.
    if not (ROOT/'preregistration_commit.json').exists(): raise ValueError('Commit frozen config and cohort before main inference')
    return value


def freeze():
    value = read_json(ROOT/'preparation.json')
    if value['missing_phi'] or value['delta_sample_count'] < 8 or value['modification_limit'] is None:
        raise ValueError('Preparation gates incomplete')
    selection = read_json(ROOT/'selection.json')
    value = {**value, 'selection_sha256':file_sha(ROOT/'selection.json'), 'api_contract_sha256':file_sha(ROOT/'api_contract.json'),
             'prompts_sha256':file_sha(Path(__file__).with_name('luna_v2_prompts.json')),
             'candidate_selection':'first passing candidate in generation order; D always first executable candidate',
             'C_contract':'same target selection and current-context rendering; only notices removed. Once accepted states diverge, later contexts may differ.',
             'noise_quantile_method':'numpy linear q=.9; accepted surface-only paraphrases',
             'score_basis':'frozen GPU scorer, mean of eight expected 1–9 scores',
             'modification_limit_scope':'rewritten paragraph; structure uses whole essay',
             'max_concurrent_essays':4, 'training':False, 'final_test_opened':False,
             'interpretation':'No equivalence claim from nonsignificant Wilcoxon; generation model does not establish an absolute capability upper bound.'}
    target = ROOT/'frozen_config.json'
    if target.exists() and read_json(target) != value: raise ValueError('Frozen config already differs')
    write_json(target,value)
    destination = Path(__file__).parent/'config'
    write_json(destination/'luna_v2_config.json',value)
    write_json(destination/'luna_v2_eval50.json',{'seed':selection['seed'],'training_exclusion_required':True,'cohort':selection['eval50']})
    write_json(REPO/'verak/v5/evaluation_holdouts/luna_upper_bound_v2.json',{'cohort':selection['eval50'],'exclude_from_training':True})
    return value


def row_for(meta):
    row = read_json(ROOT/'sources'/(meta['source_id'].replace(':','_')+'.json'))
    row['text'] = Environment().document(row['text']).text
    return row


def dispatch(name, provider, requests):
    folder = ROOT/'jobs'/name; folder.mkdir(parents=True,exist_ok=True)
    rows, mapping = anonymized(requests)
    write_json(folder/'mapping.json',mapping)
    client = Batch(provider,root=ROOT)
    manifest = folder/'dispatch.json'
    if manifest.exists():
        sent = read_json(read_json(manifest)['manifest'])
    else:
        sent = client.send(name,rows); write_json(manifest,{'manifest':sent['manifest']})
    done = client.poll(sent)
    if done['status'] != 'completed': client.close()
    return done


def submit_baselines():
    config(); requests = []
    for meta in read_json(ROOT/'selection.json')['eval50']:
        row = row_for(meta); scores = score(row['question'],row['text'])
        write_json(ROOT/'main_sources'/(row['source_id'].replace(':','_')+'.json'),row)
        for arm in ['A','A+']:
            requests.append({'key':[row['source_id'],arm], 'essay_id':row['source_id'],'condition':arm,
                             'prompt':p1(row,scores,arm), 'max_output':4096})
    return dispatch('baselines','luna',requests)


def agents(followup=False):
    cfg = config(); phase = 'main_N3' if followup else 'main'
    if followup:
        originals = [read_json(p) for p in (ROOT/'main/B').glob('*/result.json')]
        if len(originals) != 50: raise ValueError('Complete all B essays before N=3 trigger')
        items = [i for r in originals for i in r['items']]
        share = sum(i['kept_original'] for i in items)/len(items) if items else 0.
        write_json(ROOT/'followup_trigger.json',{'B_kept_original_share':share,'triggered':share>.5,'N':3,'shared_total_cap_usd':5.})
        if share <= .5: return
        cfg = {**cfg,'N':3}
    def run_one(meta):
        row = row_for(meta)
        for arm in ['B','C','D']:
            path = ROOT/phase/arm/row['source_id'].replace(':','_')/'result.json'
            if not path.exists():
                result = Episode(row,arm,cfg,phase).run()
                print(json.dumps({'phase':phase,'essay_id':row['source_id'],'condition':arm,'calls':len(result['calls'])}),flush=True)
    with ThreadPoolExecutor(max_workers=cfg['max_concurrent_essays']) as pool:
        futures = [pool.submit(run_one,meta) for meta in read_json(ROOT/'selection.json')['eval50']]
        for f in as_completed(futures): f.result()


def collect_baselines():
    config()
    job = ROOT/'jobs/baselines'
    saved = read_json(read_json(job/'dispatch.json')['manifest'])
    client = Batch('luna',root=ROOT); result = client.poll(saved)
    if result['status'] != 'completed':
        client.close(); return result['status']
    mapping = read_json(job/'mapping.json')
    for cid, value in result['outputs'].items():
        if value['status'] != 'completed': raise RuntimeError('Incomplete baseline: '+cid)
        meta = mapping[cid]; row = row_for({'source_id':meta['essay_id']})
        cases, aligned = edit_cases(row['text'],value['raw'].strip())
        write_json(ROOT/'main'/meta['condition']/meta['essay_id'].replace(':','_')/'result.json',
                   {'essay_id':meta['essay_id'],'condition':meta['condition'],'phase':'main','original':row['text'],'final':value['raw'].strip(),
                    'calls':[{'role':'one_pass','usage':value['usage'],'fingerprint':value['fingerprint'],'cost_usd':value['confirmed_usd']}],
                    'alignment':aligned,'wall_seconds':None,'batch_wall_seconds':result['wall_seconds'],
                    'latency_note':'Provider Batch exposes batch completion latency, not per-request wall-clock.'})
    return 'completed'


def feedback():
    config(); requests = []
    for meta in read_json(ROOT/'selection.json')['eval50']:
        row = row_for(meta)
        requests.append({'key':[row['source_id'],'problems'],'essay_id':row['source_id'],'prompt':feedback_prompt(row),'max_output':1400})
    result = dispatch('feedback','claude',requests)
    if result['status'] == 'completed':
        mapping = read_json(ROOT/'jobs/feedback/mapping.json')
        for cid, value in result['outputs'].items():
            if value['status'] != 'completed': raise ValueError('Incomplete feedback judgment')
            write_json(ROOT/'problems'/(mapping[cid]['essay_id'].replace(':','_')+'.json'),validate_feedback(value['raw']))
    return result['status']


def judgments(phase='main'):
    config(); requests = []
    expected = 5 if phase=='main' else 3
    arms = CONDITIONS if phase=='main' else ['B','C','D']
    for meta in read_json(ROOT/'selection.json')['eval50']:
        row = row_for(meta); problems = read_json(ROOT/'problems'/(row['source_id'].replace(':','_')+'.json'))
        for arm in arms:
            result = read_json(ROOT/phase/arm/row['source_id'].replace(':','_')/'result.json')
            requests.append({'key':[phase,row['source_id'],arm,'essay'],'essay_id':row['source_id'],'condition':arm,'kind':'essay',
                             'prompt':essay_prompt(row,problems,result['final']),'max_output':1500})
            cases,audit = edit_cases(result['original'],result['final'])
            write_json(ROOT/'edit_cases'/phase/arm/(row['source_id'].replace(':','_')+'.json'),{'cases':cases,'alignment':audit})
            for i, case in enumerate(cases):
                if case['judgable']:
                    requests.append({'key':[phase,row['source_id'],arm,'link',i],'essay_id':row['source_id'],'condition':arm,
                                     'kind':'link','case_index':i,'prompt':case['prompt'],'max_output':350})
    # Bound reservations per chunk; no paid request may exceed the experiment cap.
    # Each CLI invocation dispatches at most one new chunk, permitting status and
    # budget checks between provider batches.
    for begin in range(0,len(requests),60):
        name = f'{phase}_judge_{begin:05d}'
        done = dispatch(name,'claude',requests[begin:begin+60])
        if done['status'] != 'completed': return {'status':done['status'],'chunk':name,'total_requests':len(requests)}
        mapping = read_json(ROOT/'jobs'/name/'mapping.json')
        for cid,value in done['outputs'].items():
            if value['status'] != 'completed': raise ValueError('Incomplete Claude judgment '+cid)
            m = mapping[cid]; row = row_for({'source_id':m['essay_id']})
            result = read_json(ROOT/phase/m['condition']/m['essay_id'].replace(':','_')/'result.json')
            if m['kind']=='essay':
                problems = read_json(ROOT/'problems'/(m['essay_id'].replace(':','_')+'.json'))
                judged = validate_essay(value['raw'],problems,result['final'])
            else:
                case = read_json(ROOT/'edit_cases'/phase/m['condition']/(m['essay_id'].replace(':','_')+'.json'))['cases'][m['case_index']]
                judged = validate_link(value['raw'],case)
            write_json(ROOT/'judgments'/phase/(cid+'.json'),{**m,'judgment':judged,'api_fingerprint':value['fingerprint']})
    return {'status':'completed','total_requests':len(requests)}


def cost_probe():
    """Training-only judge pilot, before human/main gates; shared $5 ledger."""
    requests, candidates = [], []
    for meta in read_json(ROOT/'selection.json')['prep10']:
        row = row_for(meta)
        result = read_json(ROOT/'prep_discovery/B'/row['source_id'].replace(':','_')/'result.json')
        requests.append({'key':['cost_probe',row['source_id'],'problems'],'essay_id':row['source_id'],'kind':'problems',
                         'prompt':feedback_prompt(row),'max_output':1400})
        for i,case in enumerate(edit_cases(result['original'],result['final'])[0]):
            if case['judgable']:
                candidates.append({'key':['cost_probe',row['source_id'],'link',i],'essay_id':row['source_id'],'kind':'link',
                                   'prompt':case['prompt'],'max_output':350})
    random.Random(20261014).shuffle(candidates)
    requests.extend(candidates[:30])
    result = dispatch('prep_cost_probe','claude',requests)
    if result['status'] == 'completed':
        mapping = read_json(ROOT/'jobs/prep_cost_probe/mapping.json')
        grouped = {}
        for cid,value in result['outputs'].items():
            if value['status'] != 'completed': raise ValueError('Cost probe incomplete response')
            group = mapping[cid]['kind']; grouped.setdefault(group,[]).append(value['usage'])
        write_json(ROOT/'claude_cost_probe.json',{'training_only':True,'usage_by_kind':grouped,'batch_wall_seconds':result['wall_seconds']})
    return {'status':result['status'],'requests':len(requests)}


if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('part',choices=['freeze','baselines','collect_baselines','agents','followup','feedback','judge','judge_followup','cost_probe'])
    args = p.parse_args()
    actions = {'freeze':freeze,'baselines':submit_baselines,'collect_baselines':collect_baselines,'agents':agents,
               'followup':lambda:agents(True),'feedback':feedback,'judge':judgments,'judge_followup':lambda:judgments('main_N3'),'cost_probe':cost_probe}
    result = actions[args.part]()
    if result is not None: print(json.dumps({k:v for k,v in result.items() if k not in {'prepared','outputs','provider'}} if isinstance(result,dict) else result,ensure_ascii=False))
