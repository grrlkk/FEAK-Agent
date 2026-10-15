"""Frozen dev100 diagnosis; public full-system path has no privileged records."""
import argparse
from collections import Counter
import time

from .common import *


def prepare():
    from verak.v3.rft1.evaluate import inputs
    from verak.v3.train.sft_data import config_for
    from verak.v4.scale3_corruption import prepare_corruption
    local_analysis_environment()
    config = config_for(); design, corpus, examples = inputs(config)
    cases = []
    for eid in design['contract']['dev_ids']:
        prepared, path = prepare_corruption(corpus[eid], root=ROOT/'A/prepared', config=config, examples=examples)
        row_path = ROOT/'A/rows'/(safe_id(eid)+'.json')
        if not row_path.exists():
            atomic_new(row_path, public_row(prepared['policy_row']))
        cases.append({'case_id': eid, 'row': ref(row_path), 'prepared': ref(path)})
    value = {'cases': cases, 'operators': dict(Counter(r['op'] for eid in design['contract']['dev_ids']
                                                     for r in corpus[eid]['records']))}
    path = ROOT/'A/manifest.json'
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Prepared dev100 differs')
    else:
        atomic_new(path, value)
    return value


def a1():
    if (ROOT/'A/a1/complete.json').exists():return
    from verak.v4.environment_v43 import BoundaryParagraphs
    from verak.v4.fix1_infer import run_case
    started = time.monotonic()
    local_analysis_environment(); tok = tokenizer()
    analysis = BoundaryParagraphs(load_config(), cache_dir=ROOT/'bareun')
    score = ReferenceScores(); backend = Backend()
    try:
        cases = read_json(ROOT/'A/manifest.json')['cases']
        for index, case in enumerate(cases, 1):
            # a1 opens only the public row. Private prepared paths stay closed.
            row = public_row(verified(case['row']))
            result = run_case(row, case_id=case['case_id'], output=ROOT/'A/a1', backend=backend,
                              tok=tok, analysis=analysis, score=score)
            write_json(ROOT/'progress.json', {'part': 'A.a1', 'done': index, 'total': len(cases),
                'case_id': case['case_id'], 'at': time.time(), 'complete': result['complete']})
            print(f'A.a1 {index}/{len(cases)} {case["case_id"]} complete={result["complete"]}', flush=True)
        atomic_new(ROOT/'A/a1/complete.json', {'episodes': len(cases), 'wall_seconds': time.monotonic()-started,
                                              'private_records_opened_during_generation': False})
    finally:
        backend.close(); score.close()


def oracle_items(prepared,document):
    """Recorded corrupted sites + current positional predecessor, no answers.

    Paragraph aliases are exposed only when every paragraph member is a target
    of this record. Destination anchors are public current IDs anywhere in text.
    """
    aliases=prepared['stable_id_aliases']['public_to_private_sentences']
    inverse={v:k for k,v in aliases.items()}
    mapping=prepared['policy_row']['normalization']['sid_mapping']
    anns={a.sid:a for a in document.structure().annotations}
    templates={i['record_id']:i for i in prepared['plan']['items'] if i.get('record_id')}
    output=[]
    for record in prepared['corpus']['records']:
        template=templates[record['record_id']]
        targets={s for sid in record['sids'] for s in mapping.get(inverse.get(sid,sid),[])}
        targets &= anns.keys()
        scope=set(targets)|{anns[s].predecessor_id for s in targets if anns[s].predecessor_id in anns}
        move_targets=set(targets)
        if record['op']=='G_PARA_SWAP':
            for p in document.paragraphs:
                if p.units and {u.sid for u in p.units}<=targets:
                    scope.add(p.pid);move_targets.add(p.pid)
        item={k:template[k] for k in ('action','instruction','needs_search')}
        item['location']=sorted(scope)
        output.append({'record_id':record['record_id'],'op':record['op'],
            'role':template['owner'],'item':item,'move_targets':sorted(move_targets)})
    return output


def a2():
    if (ROOT/'A/a2/complete.json').exists():return
    from copy import deepcopy
    from verak.v4.environment_v43 import BoundaryParagraphs
    from verak.v4.scale2_projection import save_document
    from .pipeline import ScopedEnvironment, stage
    started=time.monotonic();local_analysis_environment();tok=tokenizer()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'bareun');backend=Backend()
    cases=read_json(ROOT/'A/manifest.json')['cases']
    try:
        for index,case in enumerate(cases,1):
            directory=ROOT/'A/a2'/safe_id(case['case_id']);path=directory/'attempt.json'
            if path.exists():continue
            prepared=verified(case['prepared']);row=public_row(verified(case['row']))
            env=ScopedEnvironment(row,analysis)
            full=read_json(ROOT/'A/a1/attempts'/(safe_id(case['case_id'])+'.json'))
            original_map=next((a['value']['map'] for a in full['orchestrator_actions']
                               if a.get('valid') and a.get('action')=='MAP'),None)
            env.shared_map=deepcopy(original_map)
            result={'case_id':case['case_id'],'source_id':row['source_id'],'initial':save_document(env.document),
                    'stages':[],'original_map_reused_from_a1':True,'plan_source':'record sites + worker SFT templates',
                    'original_forms_or_destination_revealed':False,'MOVE_destination_anchors':'whole_current_essay'}
            # Structural first, then form records, then the final whole-essay pass.
            order={r['record_id']:i for i,r in enumerate(prepared['corpus']['records'])}
            tasks=oracle_items(prepared,env.document)
            tasks.sort(key=lambda x:(x['role']=='korean',order[x['record_id']]))
            for number,t in enumerate(tasks,1):
                # Update current antecedents after preceding structural changes.
                current=next(x for x in oracle_items(prepared,env.document) if x['record_id']==t['record_id'])
                if not current['item']['location']:
                    result['stages'].append({'record_id':t['record_id'],'role':t['role'],
                        'ending':'target_absent','no_protocol_error':False,'all_done':False,'calls':[]});continue
                s=stage(env,t['role'],[current['item']],backend,tok,directory/f'd{number:02d}',
                        number=number,scope=current['item']['location'],
                        move_targets=set(current['move_targets']) if t['role']=='revision' else None)
                result['stages'].append({'record_id':t['record_id'],'operator':t['op'],**s})
            s=stage(env,'korean',[],backend,tok,directory/'whole_korean',number=len(tasks)+1)
            result['stages'].append({'final_whole_essay_pass':True,**s})
            result.update(final=save_document(env.document),lineage=deepcopy(env.lineage),actions=deepcopy(env.actions),
                          no_protocol_error=all(s['no_protocol_error'] for s in result['stages']),
                          all_done=all(s['all_done'] for s in result['stages']))
            atomic_new(path,result)
            write_json(ROOT/'progress.json',{'part':'A.a2','done':index,'total':len(cases),'at':time.time()})
            print(f'A.a2 {index}/{len(cases)} {case["case_id"]}',flush=True)
        atomic_new(ROOT/'A/a2/complete.json',{'episodes':len(cases),'wall_seconds':time.monotonic()-started})
    finally:backend.close()


def a3():
    if (ROOT/'A/a3/complete.json').exists():return
    from verak.v4.environment_v43 import BoundaryParagraphs
    from .pipeline import onepass
    started=time.monotonic();local_analysis_environment();tok=tokenizer()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'bareun');backend=Backend()
    cases=read_json(ROOT/'A/manifest.json')['cases']
    try:
        for index,case in enumerate(cases,1):
            row=public_row(verified(case['row']))
            onepass(row,backend,tok,analysis,ROOT/'A/a3'/safe_id(case['case_id']))
            write_json(ROOT/'progress.json',{'part':'A.a3','done':index,'total':len(cases),'at':time.time()})
            print(f'A.a3 {index}/{len(cases)} {case["case_id"]}',flush=True)
        atomic_new(ROOT/'A/a3/complete.json',{'episodes':len(cases),'wall_seconds':time.monotonic()-started})
        if all((ROOT/'A'/name/'complete.json').exists() for name in ('a1','a2')):
            from .measurement import report_a
            report_a()
    finally:backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['prepare', 'a1', 'a2', 'a3'])
    args = parser.parse_args()
    if args.part == 'prepare':
        print({k:v for k,v in prepare().items() if k != 'cases'})
    else:
        globals()[args.part]()
