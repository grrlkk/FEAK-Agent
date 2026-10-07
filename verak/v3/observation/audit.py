"""Read-only integrity checks over finished observation trajectories."""
from collections import Counter
from types import SimpleNamespace
import json
import re

from ..agent.runner import fit_history, split_handoff
from ..common import read_json, sha_text, write_json
from ..train.pilot import safe_id
from .environment import prompt
from .experiment import PHASE, graph_path
from .markers import episode_files


def audit(config,tokenizer):
    root=config['paths'][PHASE+'_output']
    failures=[];checks=0;contexts=0;graph_views=0
    def check(ok,what):
        nonlocal checks
        checks+=1
        if not ok:failures.append(what)
    for setting,cohort,row,_ in episode_files(config):
        name=setting+':'+row['corpus_episode_id']
        check(row['mode']=='two_stage' and row['model']=='gpt-6-luna',name+': model/mode')
        check(not any(row['checks'].values()),name+': CHECK disabled')
        for role,history in row['messages_by_role'].items():
            calls=[c for c in row['calls'] if c['role']==role]
            positions=[i for i,m in enumerate(history) if m['role']=='assistant']
            check(len(calls)==len(positions),name+': saved turns')
            check(history[0]['content']==prompt(role,setting),name+': prompt')
            for c,index in zip(calls,positions):
                turn=int(c['turn'].split(':')[0])
                replay,compacted=fit_history(history[:index],SimpleNamespace(context_limit=8192,generation_reserve=1024),
                    tokenizer,row['actions_by_role'][role][:turn-1])
                contexts+=1
                check(replay==c['messages'] and compacted==c['history_compacted'],name+': context '+c['turn'])
                check(c['max_output_tokens']==1024 and c['reasoning_effort']=='low',name+': generation '+c['turn'])
            for m in history:
                if m['role']!='user':continue
                if setting in {'text_only','graph'}:
                    check('Korean document profile' not in m['content'] and 'marker-change notices' not in m['content'],name+': no hidden profile/notices')
                if setting=='text_only' and '[GLOBAL 인계:' in m['content']:
                    handoff,_=split_handoff(m['content'])
                    check(set(json.loads(handoff.split('\n',1)[1]))=={'actions'},name+': action-only handoff')
        if setting!='graph':continue
        saved=read_json(graph_path(config,row['corpus_episode_id']))
        snap=row['initial_layout']
        text=''.join(g+''.join(u['leading']+u['text'] for u in p['units']) for g,p in zip(snap['gaps'],snap['paragraphs']))+snap['tail']
        check(sha_text(text)==saved['input_text_hash'],name+': graph from observed input')
        path=root/'graph/graph_history'/(safe_id(row['corpus_episode_id'])+'.json')
        if not path.exists():continue
        disc=saved['discourse']
        for v in read_json(path)['views']:
            graph_views+=1;g=v['graph'];sids={s['id'] for s in g['sentences']}
            expected=[e for e in disc['sentence_edges'] if e['source'] in sids and e['target'] in sids]
            check(g['sentence_edges']==expected,name+f": frozen sentence edges {v['role']}:{v['step']}")
            check(g['paragraph_edges']==disc['paragraph_edges'],name+f": paragraph nodes survive emptying {v['role']}:{v['step']}")
            # Marker predecessors follow actual current text order, never the source order.
            order=[s['id'] for s in g['sentences']]
            check(all(s['predecessor']==(order[i-1] if i else None) for i,s in enumerate(g['sentences'])),
                  name+': current predecessors')
    result={'passed':not failures,'checks':checks,'contexts_replayed':contexts,'graph_views':graph_views,
            'failures':failures}
    write_json(root/'integrity_audit.json',result)
    return result
