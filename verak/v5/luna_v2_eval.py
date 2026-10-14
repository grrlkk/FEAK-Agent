"""Blinded v2 judgments, explicit edit denominators and paired statistics."""
from collections import Counter
import json
import random

import numpy as np
from scipy.stats import rankdata, wilcoxon

from verak.v4.common import write_json, read_json, sha_text
from verak.v4.environment_v43 import sentence_spans
from .luna_v2_data import ROOT, SEED, paragraphs
from .luna_v2_env import alignment
from .luna_v2_run import render, parse_json


def units(text):
    return [(pi, p[a:b]) for pi,p in enumerate(paragraphs(text)) for a,b in sentence_spans(p)]


def edit_cases(original, final):
    old, new = units(original), units(final)
    audit = alignment([s for _,s in old], [s for _,s in new])
    pairs = sorted([(b['new'][0], b['old'][0]) for b in audit['blocks'] if len(b['old'])==len(b['new'])==1], key=lambda x:x[0])
    # Maximum increasing subsequence prevents insertions shifting every following
    # sentence from being mislabeled MOVE.
    best = []
    for j, i in pairs:
        prior = max((seq for seq in best if seq[-1][1] < i), key=len, default=[])
        best.append(prior+[(j,i)])
    stable = set(max(best, key=len, default=[]))
    cases = []
    for b in audit['blocks']:
        oi, nj = b['old'], b['new']
        if not oi: kind = 'INSERT'
        elif not nj: kind = 'DELETE'
        elif len(oi)>1: kind = 'MERGE'
        elif len(nj)>1: kind = 'SPLIT'
        elif (nj[0],oi[0]) not in stable: kind = 'MOVE'
        elif old[oi[0]][1] != new[nj[0]][1]: kind = 'REWRITE'
        else: continue
        targets = list(nj)
        if not targets:
            following = sorted((x for x in audit['blocks'] if x['old'] and x['new'] and min(x['old']) > max(oi)), key=lambda x:min(x['old']))
            if following: targets = [following[0]['new'][0]]
        if not targets:
            cases.append({'kind':kind,'old':oi,'new':nj,'judgable':False,'reason':'No surviving sentence after terminal deletion'})
            continue
        before_index = min(oi) if oi else min((i for j,i in pairs if j>=min(targets)), default=max(0,len(old)-1))
        before = '\n'.join(t for _,t in old[max(0,before_index-2):min(len(old),max(oi or [before_index])+3)])
        start, end = max(0,min(targets)-2), min(len(new),max(targets)+3)
        after = '\n'.join(('▶ ' if j in targets else '')+new[j][1] for j in range(start,end))
        prompt = render('P8', {'{앞 2문장 ~ 뒤 2문장}':before, '{앞 2문장 ~ 뒤 2문장, 대상 문장에 ▶}':after})
        cases.append({'kind':kind,'old':oi,'new':nj,'targets':targets,'judgable':True,'before':before,'after':after,'prompt':prompt,
                      'alignment_similarity':b['similarity']})
    return cases, audit


def feedback_prompt(row):
    # User approved this provenance correction; do not misrepresent the author.
    prompt = render('P7a', {'{문항}':row['question'], '[글]\n{…}':'[글]\n'+row['text'], '[피드백]\n{…}':'[피드백]\n'+row['feedback']})
    return prompt.replace('전문가가 쓴 피드백입니다.', '저장된 피드백입니다. 피드백 작성 주체는 확인되지 않았습니다.')


def essay_prompt(row, problems, final):
    fixed = [x for x in problems['items'] if x['fixable']]
    return render('P7b', {'{문항}':row['question'], '[원문]\n{…}':'[원문]\n'+row['text'],
                         'F1 {…}\nF2 {…}':'\n'.join(x['id']+' '+x['problem'] for x in fixed) or '(고칠 수 있는 문제 없음)',
                         '[고친 글]\n{…}':'[고친 글]\n'+final})


def validate_feedback(raw):
    value = parse_json(raw)
    items = value['items']
    if not isinstance(items,list) or len({x['id'] for x in items}) != len(items): raise ValueError('Invalid problem IDs')
    if any(type(x['fixable']) is not bool or not isinstance(x['problem'],str) for x in items): raise ValueError('Invalid problem schema')
    return value


def validate_essay(raw, problems, final):
    value = parse_json(raw)
    expected = {x['id'] for x in problems['items'] if x['fixable']}
    items = value['feedback']
    if len(items) != len(expected) or {x['id'] for x in items} != expected: raise ValueError('Missing/extra/duplicate feedback judgments')
    for item in items:
        if item['resolved'] not in {'yes','partial','no'}: raise ValueError('Invalid resolution label')
        evidence = item.get('evidence','')
        if item['resolved'] in {'yes','partial'} and (not evidence or evidence not in final):
            raise ValueError('Resolution evidence is not an exact quote in the revised essay')
    if type(value['thesis_kept']) is not bool or not isinstance(value['new_content'],list): raise ValueError('Invalid preservation judgment')
    return value


def validate_link(raw, case):
    value = parse_json(raw)
    if type(value.get('error')) is not bool: raise ValueError('Invalid link error boolean')
    if value.get('type') not in {'연결어미','접속어','지시','문체','없음'}: raise ValueError('Invalid link type')
    if value['error']:
        if value['type']=='없음' or not value.get('evidence') or value['evidence'] not in case['after']: raise ValueError('Unsupported error evidence')
    elif value['type']!='없음': raise ValueError('Inconsistent no-error label')
    return value


def paired(left, right):
    ids = sorted(set(left)&set(right))
    differences = np.array([left[i]-right[i] for i in ids],dtype=float)
    nonzero = differences[differences!=0]
    if not len(ids): return {'n':0, 'p':None, 'median_difference':None, 'rank_biserial':None}
    if not len(nonzero): p, effect = 1., 0.
    else:
        ranks = rankdata(abs(nonzero))
        effect = float(np.sum(ranks*np.sign(nonzero))/np.sum(ranks))
        p = float(wilcoxon(differences, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return {'n':len(ids),'nonzero_pairs':len(nonzero),'p':p,'median_difference':float(np.median(differences)),'rank_biserial':effect}


def anonymized(rows):
    """No condition names or essay IDs in provider-facing IDs or prompts."""
    output, mapping = [], {}
    for row in rows:
        custom = sha_text(json.dumps([SEED,row['key']],ensure_ascii=False))[:24]
        mapping[custom] = {k:v for k,v in row.items() if k not in {'prompt','max_output'}}
        output.append({'custom_id':custom,'prompt':row['prompt'],'max_output':row['max_output']})
    random.Random(SEED).shuffle(output)
    return output, mapping
