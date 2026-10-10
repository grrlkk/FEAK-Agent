"""Fixed 43-source validity gate followed by a separately frozen 2,000-source scale."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

from .maps import extraction_request, validate, consensus, _request_outcome
from .prep2_common import (ROOT, PREP1, GENRES, read_json, write_json, safe_id,
    file_sha, sha_text, atomic_new, collection_lock, contract, api_for, accounting)


def gate(pairs):
    if len(pairs) != 43:
        raise ValueError('Gate denominator must remain 43 original invalid sources')
    n = sum(p['status'] == 'valid' for p in pairs)
    return {'valid': n, 'total': 43, 'validity': n/43, 'passed': n/43 >= .90,
            'threshold': .90, 'required_valid_pairs': 39}


def extract(api, row, attempt, output, cohort):
    path = output/f'attempt_{attempt}'/(safe_id(row['source_id'])+'.json')
    messages, schema = extraction_request(row, long_only=True)
    identity = {'source_id':row['source_id'], 'attempt':attempt, 'genre':row['genre'],
        'request_sha256':sha_text(json.dumps([messages,schema],ensure_ascii=False,sort_keys=True))}
    if path.exists():
        saved = read_json(path)
        if any(saved.get(k) != v for k,v in identity.items()):
            raise ValueError('PREP2 extraction identity changed')
        return saved
    outcome = _request_outcome(api, messages, schema, stage=f'prep2_map_{cohort}_a{attempt}',
        source_id=row['source_id'], maximum=4096, check=lambda v: validate(v,row,long_only=True))
    result = {**identity, **outcome}
    atomic_new(path,result)
    return result


def join(row, output):
    name = safe_id(row['source_id'])+'.json'
    inputs = [output/f'attempt_{i}'/name for i in (1,2)]
    identity = {'source_id':row['source_id'], 'genre':row['genre'], 'attempt_sha256':[file_sha(p) for p in inputs]}
    path = output/'consensus'/name
    if path.exists():
        old = read_json(path)
        if any(old.get(k) != v for k,v in identity.items()):
            raise ValueError('Consensus inputs changed')
        return old
    attempts = [read_json(p) for p in inputs]
    result = {**identity, 'attempt_statuses':[a['status'] for a in attempts], 'status':'unavailable'}
    if all(a['status']=='valid' for a in attempts):
        result['map'],result['diagnostics'] = consensus(*(a['value'] for a in attempts),row,long_only=True)
        result['status'] = 'valid'
    atomic_new(path,result)
    return result


def collect(api, rows, output, cohort):
    frozen = {'sources':[r['source_id'] for r in rows], 'input_sha256':{
        r['source_id']:sha_text(json.dumps(extraction_request(r,long_only=True),ensure_ascii=False,sort_keys=True)) for r in rows},
        'workers':4, 'max_output':4096, 'attempts':2, 'sampling_seed':None}
    if (output/'design.json').exists():
        if read_json(output/'design.json') != frozen:
            raise ValueError('Frozen map cohort changed')
    else:
        atomic_new(output/'design.json',frozen)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(extract,api,row,a,output,cohort) for row in rows for a in (1,2)]
        for number,future in enumerate(as_completed(futures),1):
            value = future.result()
            write_json(ROOT/'C/status.json', {'stage':cohort,'saved_attempts':number,'planned_attempts':2*len(rows),
                'last_status':value['status'],'at':time.time(),'api':api.accounting()})
    pairs = [join(row,output) for row in rows]
    stats = summarize(rows,pairs,output)
    write_json(output/'metrics.json',stats)
    return pairs


def summarize(rows,pairs,output):
    statuses,errors = Counter(),Counter()
    relations = defaultdict(Counter)
    by_genre = {g:Counter() for g in GENRES}
    flags = Counter()
    for row,pair in zip(rows,pairs):
        by_genre[row['genre']]['sources'] += 1
        for a in (1,2):
            v = read_json(output/f'attempt_{a}'/(safe_id(row['source_id'])+'.json'))
            statuses[v['status']] += 1
            if v.get('error'):
                errors[v['error']] += 1
        if pair['status'] != 'valid':
            continue
        by_genre[row['genre']]['valid_pairs'] += 1
        diag = pair['diagnostics']
        for relation,counts in diag['relations'].items():
            for scope in ('all',row['genre']):
                for key in ('left','right','intersection','union'):
                    relations[scope+':'+relation][key] += counts[key]
        flags['final_advisory_sentences'] += len(pair['map']['off_topic'])
        flags['protected_overrides'] += len(diag['off_topic']['protected_override'])
    return {'sources':len(rows),'valid_pairs':sum(p['status']=='valid' for p in pairs),'single_run_statuses':dict(statuses),
        'invalid_reasons':dict(errors),'by_genre':{g:dict(v) for g,v in by_genre.items()},
        'agreement':{k:{**v,'jaccard':v['intersection']/v['union'] if v['union'] else None} for k,v in relations.items()},
        'off_topic':dict(flags),'off_topic_is_advisory':True,'semantic_accuracy_not_measured':True}


def run():
    import socket
    from .prep2_data import materialize
    with collection_lock(ROOT/'C'):
        # A sandbox DNS failure is not a model outcome or a validity observation.
        # Fail once before dispatch, rather than storing 86 artificial failures.
        socket.getaddrinfo('api.openai.com',443)
        contract()
        old = read_json(PREP1/'design.json')
        ids = [s for s in old['maps150'] if read_json(PREP1/'C/consensus'/(safe_id(s)+'.json'))['status']!='valid']
        if len(ids)!=43:
            raise ValueError('Original 43 invalid-pair cohort changed')
        rows = [read_json(PREP1/'essays'/(safe_id(s)+'.json')) for s in ids]
        api = api_for('C')
        try:
            api.settle_interrupted()
            prior = list((ROOT/'C/retry43').glob('attempt_*/*.json'))
            if prior and not api.accounting()['calls'] and all(
                read_json(p).get('requests') == [] and read_json(p).get('error','').startswith('gaierror:') for p in prior):
                archive = ROOT/'C'/('not_dispatched_dns_'+str(int(time.time())))
                archive.mkdir()
                for name in ('retry43','gate.json','complete.json','status.json'):
                    path = ROOT/'C'/name
                    if path.exists():
                        path.rename(archive/name)
            pairs = collect(api,rows,ROOT/'C/retry43','retry')
            outcome = gate(pairs)
            atomic_new(ROOT/'C/gate.json',outcome) if not (ROOT/'C/gate.json').exists() else None
            if outcome['passed']:
                write_json(ROOT/'C/status.json',{'stage':'materializing_2000','at':time.time(),'api':api.accounting()})
                scale_rows = materialize('maps2000')
                collect(api,scale_rows,ROOT/'C/scale2000','scale')
            write_json(ROOT/'C/complete.json',{'gate':outcome,'scale_started':outcome['passed'],
                'accounting':api.accounting(),'at':time.time()})
        finally:
            api.close()
