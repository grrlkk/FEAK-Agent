"""Dev-only reward sanity audit using frozen Bareun observations; no model calls."""
from collections import Counter,defaultdict
from statistics import mean

from verak.v3.common import load_config,write_json,file_sha
from verak.v3.phase2 import read_jsonl
from verak.v3.view_data import load_episode_examples
from verak.v3.corrupt.document import BareunBank,Document,source_document
from verak.v3.reward.recovery import recover_records
from verak.v3.reward.overedit import overedit


class NoLiveAnalysis:
    def profile(self,text):
        raise RuntimeError('Reward audit requires cached Bareun observations')


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--corpus', type=Path, help='Defaults to the active dev corpus')
    args = parser.parse_args()
    config=load_config()
    output=args.out or config['paths'].get('phase5_output', config['paths']['phase4_output'])
    output.mkdir(parents=True, exist_ok=True)
    examples={e.id:e for e in load_episode_examples(config,'agent_dev')}
    bank=BareunBank(config,analyzer=NoLiveAnalysis(),cache_dir=output/'reward_profile_cache',
        read_cache_dirs=(config['paths']['phase7_pilot2_output']/'bareun_units',
                         config['paths']['phase3b_output']/'bareun_units',
                         config['paths']['phase3_output']/'bareun_units'))
    corpus=args.corpus or config['paths']['active_corrupt']/'agent_dev.jsonl'
    rows=read_jsonl(corpus)
    counts=Counter(); failures=[]; cache={}; unchanged=defaultdict(list); partial=[]
    source_ids={row['source_id'] for row in rows}
    source_ids.update(r['params']['donor']['source_id'] for row in rows for r in row['records']
                      if r['op']=='G_OFFTOPIC')
    for sid in sorted(source_ids):
        cache[sid]=source_document(config,examples[sid],bank)
    for row in rows:
        sid=row['source_id']
        source=cache[sid]
        corrupted=Document.restore(row['corrupted_layout'],bank)
        for result in recover_records(source,source,row['records'],corrupted=corrupted):
            counts[result['op']]+=1
            if result['recovery']!=1:
                failures.append({'episode_id':row['episode_id'],**result})
        if overedit(source,source,source,row['records'])['value']!=0:
            raise ValueError('Source has nonzero overedit')
        if overedit(source,source,corrupted,row['records'])['order']!=0:
            raise ValueError('Unchanged corrupted essay has nonzero order overedit')
        for result in recover_records(source,corrupted,row['records'],corrupted=corrupted):
            unchanged[result['op']].append(result['recovery'])
            if result['recovery']:
                partial.append({'episode_id':row['episode_id'],**result})
    result={'essays':len(rows),'records':sum(counts.values()),'operator_counts':dict(counts),
        'corpus_sha256':file_sha(corpus),'source_recovery_failures':failures,
        'source_overedit_failures':0,
        'unchanged_order_overedit_failures':0,
        'unchanged_recovery':{op:{'n':len(v),'mean':mean(v),'min':min(v),'max':max(v),
                                     'nonzero':sum(x>0 for x in v)} for op,v in unchanged.items()},
        'unchanged_partial_records':partial,'new_bareun_calls':0}
    write_json(output/'source_recovery_audit.json',result)
    print({'essays':len(rows),'records':sum(counts.values()),'source_failures':len(failures),
           'unchanged_nonzero':len(partial)})
    if failures or partial:
        raise SystemExit('Source recovery audit failed')


if __name__=='__main__':
    main()
