"""New D3 sources and the unchanged PREP2 map manifest; local Bareun only."""
from collections import Counter
import random
import time

from verak.v3.common import extract_question_essay
from verak.v3.corrupt.document import Document
from verak.v3.insertion_boost.resources import BoostParagraphs
from verak.v3.v2_ops.local import load_environment
from .data import read_rows, genre_of, normalized_source, feedback_parts
from .prep3_common import (REPO, ROOT, PREP1, PREP2, GENRES, read_json, write_json,
    file_sha, sha_text, safe_id, atomic_new, load_config, freeze_contract)


def freeze_content():
    freeze_contract()
    path=ROOT/'content/sample.json'
    if path.exists():
        saved=read_json(path)
        if file_sha(saved['input_path'])!=saved['input_sha256']:
            raise ValueError('Raw train input changed')
        return saved
    old=read_json(PREP1/'design.json'); previous=read_json(PREP2/'sample.json')
    forbidden_q=set(previous['test_question_hashes'])
    prior={**old['source_metadata'],**previous['source_metadata']}
    forbidden_norm={r['normalized_source_hash'] for r in prior.values()}
    pool={g:[] for g in GENRES}; excluded=Counter(); seen=set()
    rawpath=REPO/'data/data_jsonl/train.jsonl'
    for index,raw in read_rows(rawpath):
        source=f'train:{index}'; question,essay=extract_question_essay(raw)
        norm=normalized_source(essay); genre=genre_of(raw)
        if source in prior or norm in forbidden_norm:
            excluded['prior_prep_or_frozen_map_source']+=1; continue
        if sha_text(question) in forbidden_q:
            excluded['test_question']+=1; continue
        if norm in seen:
            excluded['duplicate_source']+=1; continue
        if genre not in GENRES:
            excluded['unknown_genre']+=1; continue
        scores=raw.get('grader_1_scores',[])+raw.get('grader_2_scores',[])
        if len(scores)!=16 or any(type(s) not in (int,float) or not 1<=s<=5 for s in scores):
            excluded['invalid_human_scores']+=1; continue
        seen.add(norm)
        pool[genre].append({'source_id':source,'genre':genre,'question_hash':sha_text(question),
            'essay_hash':sha_text(essay),'normalized_source_hash':norm,
            'human_mean':sum(scores)/16,'characters':len(essay)})
    chosen=[]; eligibility={}
    for offset,genre in enumerate(GENRES):
        values=sorted(pool[genre],key=lambda r:(r['human_mean'],r['source_id']))
        cutoff=values[(2*len(values)-1)//3]['human_mean']
        lowmid=[r for r in values if r['human_mean']<=cutoff]
        random.Random(251+offset).shuffle(lowmid)
        n=34 if offset==0 else 33
        if len(lowmid)<n: raise ValueError('Not enough new low/middle sources')
        chosen+=lowmid[:n]
        eligibility[genre]={'eligible':len(values),'low_middle_cutoff':cutoff,'low_middle_count':len(lowmid),'chosen':n}
    random.Random(257).shuffle(chosen)
    result={'version':'prep3_content','input_path':str(rawpath),'input_sha256':file_sha(rawpath),
        'source_ids':[r['source_id'] for r in chosen],'source_metadata':{r['source_id']:r for r in chosen},
        'test_question_hashes':sorted(forbidden_q),'eligibility':eligibility,'exclusions':dict(excluded),
        'excludes_prior_prep_and_frozen_2000_maps':True,'human_sampling':'raw grader_1/2 mean on 1–5',
        'sampling_seeds':[251,252,253,257],'feedback_origin':'strong_LLM'}
    atomic_new(path,result)
    return result


def materialize(group):
    freeze_contract()
    if group=='maps':
        sample=read_json(PREP2/'sample.json'); sources=sample['maps2000']
    elif group=='content':
        sample=freeze_content(); sources=sample['source_ids']
    else:
        raise ValueError('Unknown source cohort')
    if file_sha(sample['input_path'])!=sample['input_sha256']:
        raise ValueError('Raw source input changed')
    missing={s for s in sources if not (ROOT/'essays'/(safe_id(s)+'.json')).exists()}
    # Reuse exact existing profiles without touching their original artifacts.
    for source in list(missing):
        for base in (PREP2,PREP1):
            old=base/'essays'/(safe_id(source)+'.json')
            if old.exists():
                row=read_json(old)
                if row['essay_hash']!=sample['source_metadata'][source]['essay_hash']:
                    raise ValueError('Existing source profile hash mismatch')
                atomic_new(ROOT/'essays'/(safe_id(source)+'.json'),row)
                missing.remove(source); break
    if missing:
        config=load_config(); load_environment(config)
        analyzer=BoostParagraphs(config,cache_dir=ROOT/('bareun_'+group))
        for index,raw in read_rows(sample['input_path']):
            source=f'train:{index}'
            if source not in missing: continue
            question,text=extract_question_essay(raw)
            profile=analyzer.profile(text); document=Document.from_profile(text,profile)
            row={**sample['source_metadata'][source],'split':'train','question':question,'text':text,
                'profile':profile.to_dict(),'layout':document.snapshot(),
                'paragraphs':[{'id':p.pid,'sentences':[{'id':u.sid,'text':u.text} for u in p.units]} for p in document.paragraphs],
                'human_scores':[raw['grader_1_scores'],raw['grader_2_scores']],
                'feedback':feedback_parts(raw['assistant']),'feedback_origin':'strong_LLM','scorer_seen':True}
            if sha_text(text)!=row['essay_hash'] or len(row['feedback'])!=8:
                raise ValueError('Source materialization mismatch')
            atomic_new(ROOT/'essays'/(safe_id(source)+'.json'),row)
            missing.remove(source)
            write_json(ROOT/group/'materialization.json',{'remaining':len(missing),'planned':len(sources),'at':time.time()})
    if missing: raise ValueError('Missing frozen source IDs')
    manifest={s:{'path':str(ROOT/'essays'/(safe_id(s)+'.json')),
        'sha256':file_sha(ROOT/'essays'/(safe_id(s)+'.json'))} for s in sources}
    path=ROOT/group/'source_files.json'
    if path.exists() and read_json(path)!=manifest: raise ValueError('Frozen source files changed')
    if not path.exists(): atomic_new(path,manifest)
    return [read_json(manifest[s]['path']) for s in sources]


def materialize_maps():
    return materialize('maps')
