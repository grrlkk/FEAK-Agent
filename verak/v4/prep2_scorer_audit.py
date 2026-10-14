"""File-only scorer-label provenance audit. Never load model weights or make calls."""
from pathlib import Path
from collections import Counter
import re

from .data import read_rows
from .prep2_common import REPO, ROOT, read_json, write_json, file_sha, load_config, FACT


def compare_row(row):
    first=row['assistant'].splitlines()[0].strip()
    if not re.fullmatch(r'[1-9](?:\s+[1-9]){7}',first):
        raise ValueError('Assistant first line is not eight 1–9 scores')
    assistant=[int(x) for x in first.split()]
    r1,r2=row['grader_1_scores'],row['grader_2_scores']
    if len(r1)!=8 or len(r2)!=8 or any(not isinstance(x,(int,float)) or x!=int(x) or not 1<=x<=5 for x in r1+r2):
        raise ValueError('Expected two eight-rubric human integer arrays on 1–5')
    mapped=[int(a+b-1) for a,b in zip(r1,r2)]
    return assistant,r1,r2,mapped


def run():
    adapter=Path(load_config()['paths']['scorer_adapter'])
    project=adapter.parents[1]
    inventory=[p.name for p in adapter.iterdir() if p.is_file()]
    files=[REPO/'verak/v3/config.yaml',REPO/'verak/v3/score/kanana.py',
        REPO/'imple/reports/V3_PHASE_0.md',adapter/'adapter_config.json',adapter/'README.md',
        project/'essay_scoring_llm/scaling.py',project/'essay_scoring_llm/dataset.py',
        project/'essay_scoring_llm/experiment.py',Path(load_config()['paths']['policy_base'])/'config.json']
    base_config=read_json(files[-1])
    results={}
    for split in ('train','valid','test'):
        path=REPO/'data/data_jsonl'/(split+'.jsonl')
        n=0; matches=[0]*8; all_match=0; invalid=[]; mismatch_examples=[]
        digits=Counter(); human=Counter()
        for number,row in read_rows(path):
            try:
                a,r1,r2,mapped=compare_row(row)
            except (ValueError,KeyError,IndexError,TypeError) as exc:
                invalid.append({'source_id':f'{split}:{number}','error':str(exc)})
                continue
            n+=1
            flags=[x==y for x,y in zip(a,mapped)]
            matches=[x+int(y) for x,y in zip(matches,flags)]
            all_match+=all(flags)
            digits.update(a); human.update(r1+r2)
            if not all(flags) and len(mismatch_examples)<5:
                mismatch_examples.append({'source_id':f'{split}:{number}','assistant':a,'grader_1':r1,
                    'grader_2':r2,'rater_sum_minus_one':mapped})
        results[split]={'path':str(path),'sha256':file_sha(path),'valid_rows':n,'invalid_rows':invalid,
            'all_eight_match_rater_sum_minus_one':all_match,'per_rubric_matches':matches,
            'matching_digits':sum(matches),'total_digits':8*n,'assistant_digit_distribution':dict(digits),
            'human_digit_distribution':dict(human),'mismatch_examples':mismatch_examples}
    result={'task':'PREP2 scorer labels','api_calls':0,'gpu_used':False,'model_loaded':False,
        'user_provenance_statement':FACT,'frozen_adapter':str(adapter),'adapter_files':sorted(inventory),
        'adapter_config':read_json(adapter/'adapter_config.json'),
        'evidence_files':{str(p):{'sha256':file_sha(p)} for p in files},
        'source_train_path_provenance':'User-confirmed in V3_PHASE_0.md; not an independently saved training run manifest',
        'training_recipe_locally_found':False,
        'training_target_construction_status':'user_confirmed_human_grader_aggregation; exact numerical mapping independently verified',
        'user_followup':'학습할 땐 교사 점수들을 합산해서 사용했네 (PREP2 follow-up)',
        'original_trainer_field_access':'not independently recoverable without original script; stored assistant digits equal mapped human labels',
        'base_architecture':base_config.get('architectures'),'base_model_type':base_config.get('model_type'),
        'separate_regression_head_in_current_loader':False,
        'loss_recipe_status':'CE Only vs CE+NTL vs CE+NTL+SAL unresolved; no original loss config/checkpoint provenance available',
        'inference_output':'eight single-digit 1–9 scores; raw nine-digit softmax expectations; no FEAK RF correction in v3 scorer',
        'human_scale_formula':'s = grader_1 + grader_2 - 1 = 2 * mean(grader_1, grader_2) - 1; mean = (s+1)/2',
        'formula_evidence':'essay_scoring_llm/scaling.py label_from_raters; this evaluation code is not itself evidence of LoRA training target construction',
        'data':results,
        'interpretation':'Assistant field provenance and original training recipe are separate questions. Exact numerical equality is a property of stored data, not proof of which fields the trainer read.'}
    write_json(ROOT/'scorer_labels.json',result)
    return result
