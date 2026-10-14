"""Six paired conditions on smoke20/dev10; no test input or training path."""
import argparse
from collections import Counter
import time

from .common import *
from . import detector
from .pipeline import initial_map, onepass, repair
from verak.v4.environment_v43 import BoundaryParagraphs, prepared_document
from verak.v4.scale2_projection import restore_document, save_document

CONDITIONS=('onepass','onepass_feak','luna','luna_selfcheck','luna_feak','luna_notice_repair')


def cases():
    manifest=verified(read_json(ROOT/'preflight.json')['C_design'])
    return [{**case,'cohort':name} for name,cohort in manifest['cohorts'].items() for case in cohort['cases']]


def directory(case):return ROOT/'C'/case['cohort']/safe_id(case['case_id'])


def notices_for(row,final,graph,path):
    values=detector.changes(prepared_document(row),restore_document(final),graph)
    value={'notices':values,'counts':dict(Counter(n['type'] for n in values)),
           'initial_rule_resolution':detector.resolved(values,restore_document(final)),
           'graph_sha256':sha_text(json.dumps(graph,sort_keys=True,ensure_ascii=False))}
    if path.exists():
        if read_json(path)!=value:raise ValueError('Frozen notice set changed')
    else:atomic_new(path,value)
    return value


def c_local():
    if (ROOT/'C/local_complete.json').exists():return
    started=time.monotonic();local_analysis_environment();tok=tokenizer()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'bareun');backend=Backend();score=ReferenceScores()
    try:
        for i,case in enumerate(cases(),1):
            row=public_row(verified(case['row']));out=directory(case)
            mapped=initial_map(row,backend,tok,analysis,score,out/'original')
            revised=onepass(row,backend,tok,analysis,out/'onepass')
            notice=notices_for(row,revised['final'],mapped['map'],out/'onepass_notices.json')
            fixed=repair(row,revised['final'],notice['notices'],mapped['map'],backend,tok,analysis,out/'onepass_feak')
            for name,value in (('original',save_document(prepared_document(row))),('onepass',revised['final']),('onepass_feak',fixed['final'])):
                path=out/(name+'_score.json')
                if not path.exists():atomic_new(path,score(row['question'],restore_document(value).text))
            write_json(ROOT/'progress.json',{'part':'C.local','done':i,'total':30,'at':time.time()})
            print(f'C.local {i}/30 {case["case_id"]}',flush=True)
        atomic_new(ROOT/'C/local_complete.json',{'episodes':30,'wall_seconds':time.monotonic()-started})
    finally:backend.close();score.close()


def c_api():
    if (ROOT/'C/api_complete.json').exists():return
    from .api import PaidAPI,LunaWorker
    from .reviser import edit_list
    started=time.monotonic();local_analysis_environment();tok=tokenizer()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'bareun');api=PaidAPI('luna')
    try:
        for i,case in enumerate(cases(),1):
            row=public_row(verified(case['row']));out=directory(case)
            mapped=read_json(out/'original/initial_map.json')
            revised=edit_list(row,api,analysis,out/'luna',scores=mapped['scores'])
            notice=notices_for(row,revised['final'],mapped['map'],out/'luna_notices.json')
            edit_list(row,api,analysis,out/'luna_selfcheck',initial_document=revised['final'],self_check=True)
            backend=LunaWorker(api,case['case_id'])
            repair(row,revised['final'],notice['notices'],mapped['map'],backend,tok,analysis,out/'luna_notice_repair',
                   lineage=revised['lineage'],declared_relations=revised['declared_relations'])
            write_json(ROOT/'progress.json',{'part':'C.api','done':i,'total':30,'at':time.time()})
            print(f'C.api {i}/30 {case["case_id"]}',flush=True)
        atomic_new(ROOT/'C/api_complete.json',{'episodes':30,'wall_seconds':time.monotonic()-started})
    finally:api.close()


def c_repair():
    if (ROOT/'C/repair_complete.json').exists():return
    started=time.monotonic();local_analysis_environment();tok=tokenizer()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'bareun');backend=Backend();score=ReferenceScores()
    try:
        for i,case in enumerate(cases(),1):
            row=public_row(verified(case['row']));out=directory(case)
            mapped=read_json(out/'original/initial_map.json');revised=read_json(out/'luna/edit_list.json')
            notice=read_json(out/'luna_notices.json')
            fixed=repair(row,revised['final'],notice['notices'],mapped['map'],backend,tok,analysis,out/'luna_feak',
                         lineage=revised['lineage'],declared_relations=revised['declared_relations'])
            variants={'luna':revised,'luna_feak':fixed,'luna_selfcheck':read_json(out/'luna_selfcheck/edit_list.json'),
                      'luna_notice_repair':read_json(out/'luna_notice_repair/repair.json')}
            for name,value in variants.items():
                path=out/(name+'_score.json')
                if not path.exists():atomic_new(path,score(row['question'],restore_document(value['final']).text))
            write_json(ROOT/'progress.json',{'part':'C.repair','done':i,'total':30,'at':time.time()})
            print(f'C.repair {i}/30 {case["case_id"]}',flush=True)
        atomic_new(ROOT/'C/repair_complete.json',{'episodes':30,'wall_seconds':time.monotonic()-started})
    finally:backend.close();score.close()


def outputs(case):
    out=directory(case)
    files={'onepass':'onepass/onepass.json','onepass_feak':'onepass_feak/repair.json',
           'luna':'luna/edit_list.json','luna_selfcheck':'luna_selfcheck/edit_list.json',
           'luna_feak':'luna_feak/repair.json','luna_notice_repair':'luna_notice_repair/repair.json'}
    return {name:read_json(out/path) for name,path in files.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['c_local','c_api','c_repair'])
    args=parser.parse_args();globals()[args.part]()
