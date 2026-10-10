"""PREP3 edge filtering, saved-map gate, and conditional frozen-manifest collection."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import socket
import time

from . import maps
from .prep3_common import (ROOT, PREP1, PREP2, GENRES, read_json, write_json, file_sha, sha_text,
    safe_id, atomic_new, collection_lock, freeze_contract, api_for, accounting)

POLICY = {'version':'v4_edge_filter_1','outgoing_support_per_sentence':1,'outgoing_example_per_sentence':1,
    'cross_paragraph_anchors_only_when_paragraphs_above':6,
    'cardinality_tie_break':'first valid edge in provider order; invalid edges do not consume a slot',
    'edge_reason_order':['fields','non_string','type','source','target','self','cross_paragraph','duplicate','outgoing_limit'],
    'non_edge_metadata':'existing genre/paragraph-role/key/off_topic validation retained',
    'map_gate_population':'300 pilot extractions + 86 PREP2 retry extractions, including unavailable maps',
    'map_gate_min_validity':.95,'edge_gate_max_dropped_fraction':.15,
    'edge_gate_denominator':'all raw sentence and paragraph relation entries; edges of metadata-invalid maps count as discarded',
    'post_intersection_hierarchy_filter_reported_separately':True,'off_topic':'advisory; union two-hop protection is not proof of relevance'}


def freeze_policy():
    freeze_contract()
    path=ROOT/'maps/edge_policy.json'
    if path.exists() and read_json(path)!=POLICY:
        raise ValueError('Frozen edge policy changed')
    if not path.exists():
        atomic_new(path,POLICY)


def raw_edge_count(value):
    return sum(len(value.get(k,[])) for k in ('sentence_relations','paragraph_relations')
               if isinstance(value,dict) and isinstance(value.get(k),list))


def filter_edges(value,row):
    # Only edge validity is relaxed; malformed map metadata remains a visible failure.
    if not isinstance(value,dict) or any(not isinstance(value.get(k),list)
            for k in ('sentence_relations','paragraph_relations')):
        raise ValueError('Map relation fields must be arrays')
    clean=maps.validate({**value,'sentence_relations':[],'paragraph_relations':[]},row,long_only=True)
    sids,pids,members=maps.indexes(row)
    basic=[]; dropped=[]
    for field,ids,types in [('sentence_relations',set(sids),maps.SENTENCE_TYPES),
                           ('paragraph_relations',set(pids),maps.PARAGRAPH_TYPES)]:
        for index,edge in enumerate(value[field]):
            reason=None
            if not isinstance(edge,dict) or set(edge)!={'source','target','type'}:
                reason='invalid_fields'
            elif not all(isinstance(edge[k],str) for k in ('source','target','type')):
                reason='non_string_ID_or_type'
            elif edge['type'] not in types:
                reason='unknown_relation_type'
            elif edge['source'] not in ids:
                reason='unknown_source_ID'
            elif not (edge['target']=='Q' if field=='sentence_relations' and edge['type']=='main' else edge['target'] in ids):
                reason='invalid_target_or_direction'
            elif edge['source']==edge['target']:
                reason='self_relation'
            if reason:
                dropped.append({'field':field,'index':index,'edge':deepcopy(edge),'reason':reason})
            else:
                basic.append((field,index,edge))
    anchors={x['key_sentence'] for x in clean['paragraph_roles'] if x['key_sentence'] is not None}
    anchors.update(e['source'] for f,_,e in basic if f=='sentence_relations' and e['type']=='main')
    seen=set(); outgoing=Counter()
    for field,index,edge in basic:
        s,t,k=edge['source'],edge['target'],edge['type']; key=(field,s,t,k)
        reason=None
        if (field=='sentence_relations' and len(pids)>6 and t!='Q' and members[s]!=members[t]
                and (s not in anchors or t not in anchors)):
            reason='long_cross_paragraph_missing_main_or_key'
        elif key in seen:
            reason='duplicate_relation'
        elif field=='sentence_relations' and k in {'support','example'} and outgoing[s,k]>=1:
            reason='outgoing_'+k+'_limit'
        if reason:
            dropped.append({'field':field,'index':index,'edge':deepcopy(edge),'reason':reason})
            continue
        clean[field].append(deepcopy(edge)); seen.add(key)
        if field=='sentence_relations' and k in {'support','example'}:
            outgoing[s,k]+=1
    return {'map':clean,'raw_edges':raw_edge_count(value),'kept_edges':raw_edge_count(clean),
        'dropped_edges':sorted(dropped,key=lambda e:(e['field'],e['index'])),
        'dropped_by_reason':dict(Counter(e['reason'] for e in dropped)),
        'policy':POLICY['version']}


def extraction_request(row):
    messages,schema=maps.extraction_request(row,long_only=True)
    messages[0]['content']=messages[0]['content'].replace(
        '문장 하나에서 나가는 support와 example은 둘을 합쳐 최대 하나다.',
        '문장 하나에서 나가는 support는 최대 하나, example도 별도로 최대 하나다. 두 종류를 각각 하나씩 둘 수 있다.')
    return messages,schema


def revalidate_one(row,original,path):
    identity={'source_id':row['source_id'],'original_path':str(original),'original_sha256':file_sha(original)}
    if path.exists():
        saved=read_json(path)
        if any(saved.get(k)!=v for k,v in identity.items()):
            raise ValueError('Saved extraction changed')
        return saved
    saved=read_json(original)
    parsed=saved.get('parsed',saved.get('value'))
    # A raw completed response is permitted; never repair/truncate incomplete JSON.
    if parsed is None:
        requests=[r for r in saved.get('requests',[]) if r.get('status')=='completed' and r.get('path')]
        if requests:
            try:
                parsed=json.loads(read_json(requests[-1]['path'])['raw'])
            except (ValueError,KeyError):
                pass
    result={**identity,'genre':row['genre'],'original_status':saved['status'],
        'raw_edges':raw_edge_count(parsed),'status':'invalid'}
    try:
        result.update(status='valid',value=filter_edges(parsed,row))
    except (ValueError,KeyError,TypeError) as error:
        result.update(error=type(error).__name__+': '+str(error),
            dropped_by_reason={'map_metadata_or_parse_invalid':raw_edge_count(parsed)})
    atomic_new(path,result)
    return result


def join(row,output):
    name=safe_id(row['source_id'])+'.json'
    paths=[output/f'attempt_{n}'/name for n in (1,2)]
    identity={'source_id':row['source_id'],'genre':row['genre'],'attempt_sha256':[file_sha(p) for p in paths]}
    target=output/'consensus'/name
    if target.exists():
        saved=read_json(target)
        if any(saved.get(k)!=v for k,v in identity.items()):
            raise ValueError('Consensus input changed')
        return saved
    attempts=[read_json(p) for p in paths]
    result={**identity,'status':'unavailable','attempt_statuses':[a['status'] for a in attempts]}
    if all(a['status']=='valid' for a in attempts):
        value,diagnostics=maps.consensus(*(a['value']['map'] for a in attempts),row,strict=False)
        filtered=filter_edges(value,row)
        result.update(status='valid',map=filtered['map'],diagnostics=diagnostics,
            post_intersection_dropped_edges=filtered['dropped_edges'])
    atomic_new(target,result)
    return result


def summarize(rows,output):
    counts=Counter(); reasons=Counter(); invalid=Counter(); genres={g:Counter() for g in GENRES}
    intersection_drops=Counter(); relation_counts=defaultdict(Counter)
    for row in rows:
        pair=join(row,output)
        genres[row['genre']]['sources']+=1
        counts['pairs']+=1; counts['valid_pairs']+=pair['status']=='valid'
        for n in (1,2):
            a=read_json(output/f'attempt_{n}'/(safe_id(row['source_id'])+'.json'))
            counts['extractions']+=1; counts['valid_maps']+=a['status']=='valid'
            genres[row['genre']]['extractions']+=1
            genres[row['genre']]['valid_maps']+=a['status']=='valid'
            filtered=a.get('value',{}) if a['status']=='valid' else {}
            counts['raw_edges']+=filtered.get('raw_edges',a.get('raw_edges',raw_edge_count(a.get('parsed'))))
            counts['kept_edges']+=filtered.get('kept_edges',0)
            if a['status']=='valid':
                reasons.update(filtered['dropped_by_reason'])
                counts['empty_maps']+=filtered['kept_edges']==0
            else:
                reasons.update(a.get('dropped_by_reason',{'map_metadata_or_parse_invalid':raw_edge_count(a.get('parsed'))}))
                invalid[a.get('error',a['status'])]+=1
        if pair['status']=='valid':
            genres[row['genre']]['valid_pairs']+=1
            counts['final_edges']+=raw_edge_count(pair['map'])
            counts['off_topic_advisory']+=len(pair['map']['off_topic'])
            counts['protection_overrides']+=len(pair['diagnostics']['off_topic']['protected_override'])
            intersection_drops.update(e['reason'] for e in pair['post_intersection_dropped_edges'])
            for kind,nums in pair['diagnostics']['relations'].items():
                relation_counts[kind].update({k:nums[k] for k in ('left','right','intersection','union')})
    counts['dropped_edges']=sum(reasons.values())
    assert counts['raw_edges']==counts['kept_edges']+counts['dropped_edges']
    return {**counts,'map_validity':counts['valid_maps']/counts['extractions'] if counts['extractions'] else None,
        'dropped_fraction':counts['dropped_edges']/counts['raw_edges'] if counts['raw_edges'] else None,
        'dropped_by_reason':dict(reasons),'invalid_map_reasons':dict(invalid),
        'post_intersection_dropped_by_reason':dict(intersection_drops),
        'by_genre':{g:dict(v) for g,v in genres.items()},
        'agreement':{k:{**v,'jaccard':v['intersection']/v['union'] if v['union'] else None} for k,v in relation_counts.items()}}


def revalidate():
    freeze_policy()
    with collection_lock(ROOT/'maps/revalidation'):
        cohorts={}
        sources=read_json(PREP1/'design.json')['maps150']
        retry=read_json(PREP2/'C/retry43/design.json')['sources']
        for cohort,ids,base in [('pilot150',sources,PREP1/'C'),('retry43',retry,PREP2/'C/retry43')]:
            rows=[read_json(PREP1/'essays'/(safe_id(s)+'.json')) for s in ids]
            output=ROOT/'maps/revalidation'/cohort
            for row in rows:
                for n in (1,2):
                    name=safe_id(row['source_id'])+'.json'
                    revalidate_one(row,base/f'attempt_{n}'/name,output/f'attempt_{n}'/name)
            cohorts[cohort]=summarize(rows,output)
            write_json(output/'metrics.json',cohorts[cohort])
        sums={k:sum(c.get(k,0) for c in cohorts.values()) for k in
            ('extractions','valid_maps','raw_edges','kept_edges','dropped_edges','valid_pairs','pairs','empty_maps')}
        validity=sums['valid_maps']/sums['extractions']; drop=sums['dropped_edges']/sums['raw_edges']
        gate={**sums,'map_validity':validity,'dropped_fraction':drop,
            'passed':validity>=.95 and drop<=.15,'thresholds':{'map_validity':.95,'dropped_fraction':.15},
            'cohorts':cohorts,'new_api_calls':0,'policy_sha256':file_sha(ROOT/'maps/edge_policy.json')}
        path=ROOT/'maps/gate.json'
        if path.exists() and read_json(path)!=gate:
            raise ValueError('Saved-map gate changed')
        if not path.exists():
            atomic_new(path,gate)
        return gate


def extract(api,row,n,output):
    path=output/f'attempt_{n}'/(safe_id(row['source_id'])+'.json')
    messages,schema=extraction_request(row)
    identity={'source_id':row['source_id'],'genre':row['genre'],'attempt':n,
        'request_sha256':sha_text(json.dumps([messages,schema],ensure_ascii=False,sort_keys=True))}
    if path.exists():
        saved=read_json(path)
        if any(saved.get(k)!=v for k,v in identity.items()):
            raise ValueError('Frozen scale extraction changed')
        return saved
    result={**identity,**maps._request_outcome(api,messages,schema,stage=f'prep3_maps_a{n}',
        source_id=row['source_id'],maximum=4096,check=lambda v:filter_edges(v,row))}
    atomic_new(path,result)
    return result


def run():
    freeze_policy()
    gate=read_json(ROOT/'maps/gate.json')
    if not gate['passed']:
        write_json(ROOT/'maps/complete.json',{'scale_started':False,'gate':gate,'api_calls':0})
        return
    from .prep3_data import materialize_maps
    with collection_lock(ROOT/'maps'):
        socket.getaddrinfo('api.openai.com',443)
        rows=materialize_maps()
        output=ROOT/'maps/scale2000'
        design={'source_ids':[r['source_id'] for r in rows],'source_manifest_sha256':file_sha(PREP2/'sample.json'),
            'attempts':2,'policy_sha256':file_sha(ROOT/'maps/edge_policy.json'),
            'request_sha256':{r['source_id']:sha_text(json.dumps(extraction_request(r),ensure_ascii=False,sort_keys=True)) for r in rows}}
        path=output/'design.json'
        if path.exists() and read_json(path)!=design:
            raise ValueError('Scale design changed')
        if not path.exists(): atomic_new(path,design)
        api=api_for('maps')
        try:
            api.settle_interrupted()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(extract,api,row,n,output) for row in rows for n in (1,2)]
                for number,future in enumerate(as_completed(futures),1):
                    value=future.result()
                    write_json(ROOT/'maps/status.json',{'stage':'extracting','saved':number,'planned':4000,
                        'last_status':value['status'],'at':time.time(),'api':api.accounting()})
            metrics=summarize(rows,output)
            write_json(output/'metrics.json',metrics)
            write_json(ROOT/'maps/complete.json',{'scale_started':True,'metrics':metrics,'accounting':api.accounting(),'at':time.time()})
        finally:
            api.close()
