"""Public-input sentence alignment: monotone matching plus conservative moves.

No corruption record or clean reference can enter this API. Ambiguous matches,
possible split/merge boundaries and unassigned paragraphs remain explicit.
"""
from collections import Counter
from difflib import SequenceMatcher
import re

from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v4.environment_v43 import sentence_spans, normalize_render


def key(text):
    return re.sub(r'\s+', '', text)


def similarity(a, b):
    return SequenceMatcher(None, key(a), key(b), autojunk=False).ratio()


def monotone(left, right, *, threshold=.35):
    """Maximum total positive lexical-match weight, allowing gaps on either side."""
    n,m = len(left),len(right)
    dp = [[0.]*(m+1) for _ in range(n+1)]
    trace = {}
    for i in range(1,n+1):
        for j in range(1,m+1):
            s = similarity(left[i-1],right[j-1])
            choices = [(dp[i-1][j], 'left'), (dp[i][j-1], 'right')]
            if s >= threshold:
                choices.append((dp[i-1][j-1]+s-threshold+.001,'pair'))
            dp[i][j],trace[i,j] = max(choices,key=lambda x:(x[0],x[1]=='pair'))
    i,j = n,m; pairs=[]
    while i and j:
        which=trace[i,j]
        if which=='pair':
            pairs.append((i-1,j-1,similarity(left[i-1],right[j-1])));i-=1;j-=1
        elif which=='left': i-=1
        else: j-=1
    return list(reversed(pairs))


def align(original, revised_text, analysis):
    if not revised_text.strip():
        raise ValueError('One-pass returned an empty essay')
    raw = [p.strip() for p in re.split(r'\n+',revised_text.strip()) if p.strip()]
    sentences = [(pi,p[a:b]) for pi,p in enumerate(raw) for a,b in sentence_spans(p)]
    old = original.units; left=[u.text for u in old]; right=[s for _,s in sentences]
    baseline=monotone(left,right)
    # Distinct identical sentences anchor IDs even when their order changed.
    lc,rc=Counter(map(key,left)),Counter(map(key,right))
    rlookup={key(s):j for j,s in enumerate(right)}
    exact={i:rlookup[key(s)] for i,s in enumerate(left) if lc[key(s)]==rc[key(s)]==1}
    pairs={i:(j,1.,'unique_exact') for i,j in exact.items()}
    free_l=[i for i in range(len(left)) if i not in pairs]
    used={j for j,_,_ in pairs.values()};free_r=[j for j in range(len(right)) if j not in used]
    for i,j,score in monotone([left[i] for i in free_l],[right[j] for j in free_r]):
        pairs[free_l[i]]=(free_r[j],score,'monotone')
    # Out-of-order edited moves: mutual, unique high-confidence unmatched pair.
    free_l=[i for i in range(len(left)) if i not in pairs]
    used={j for j,_,_ in pairs.values()};free_r=[j for j in range(len(right)) if j not in used]
    for i in free_l:
        ranked=sorted(((similarity(left[i],right[j]),j) for j in free_r),reverse=True)
        if not ranked or ranked[0][0]<.65: continue
        s,j=ranked[0]
        back=sorted((similarity(left[k],right[j]),k) for k in free_l)
        if back[-1][1]!=i or (len(ranked)>1 and s-ranked[1][0]<.10) or (len(back)>1 and s-back[-2][0]<.10): continue
        if j in {p[0] for p in pairs.values()}: continue
        pairs[i]=(j,s,'edited_move')
    inverse={j:(i,s,method) for i,(j,s,method) in pairs.items()}
    failures=[];matches=[]
    for i,(j,score,method) in sorted(pairs.items()):
        rivals=[similarity(left[k],right[j]) for k in range(len(left)) if k!=i]
        ambiguous=bool(rivals and max(rivals)>=score-.03)
        if ambiguous:
            failures.append({'type':'ambiguous_sentence_match','original':old[i].sid,'revised_index':j,
                             'score':score,'best_alternative':max(rivals)})
        if score < .50:
            failures.append({'type':'low_confidence_sentence_match','original':old[i].sid,'revised_index':j,'score':score})
        moved=any((i<k)!=(j<p[0]) for k,p in pairs.items() if k!=i)
        matches.append({'original':old[i].sid,'original_index':i,'revised_index':j,
                        'similarity':score,'method':method,'order_inversion':moved,'ambiguous':ambiguous})
    removed=[i for i in range(len(left)) if i not in pairs]
    added=[j for j in range(len(right)) if j not in inverse]
    # Boundary changes have no reliable one-to-one lineage: never invent certainty.
    for j in added:
        neighbors=[k for k in (j-1,j+1) if k in inverse and sentences[k][0]==sentences[j][0]]
        for k in neighbors:
            i,score,_=inverse[k]
            fused=' '.join(right[t] for t in sorted((j,k)))
            if similarity(left[i],fused) > max(.70,score+.10):
                failures.append({'type':'possible_sentence_split','original':old[i].sid,'revised_indices':sorted((j,k))})
    for i in removed:
        for k in (i-1,i+1):
            if k not in pairs: continue
            j,score,_=pairs[k]
            fused=' '.join(left[t] for t in sorted((i,k)))
            if similarity(fused,right[j]) > max(.70,score+.10):
                failures.append({'type':'possible_sentence_merge','originals':[old[t].sid for t in sorted((i,k))],
                                 'revised_index':j})
    # Assign paragraph identity from aligned membership, not original order.
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    old_pi={u.sid:pi for pi,p in enumerate(original.paragraphs) for u in p.units}
    weights=np.zeros((len(raw),len(original.paragraphs)))
    for j,(i,s,_) in inverse.items(): weights[sentences[j][0],old_pi[old[i].sid]]+=s
    ri,ci=linear_sum_assignment(-weights)
    para={int(a):original.paragraphs[int(b)].pid for a,b in zip(ri,ci) if weights[a,b]>0}
    for a,b in zip(ri,ci):
        if weights[a,b] and sum(weights[a,k]==weights[a,b] for k in range(weights.shape[1]))>1:
            failures.append({'type':'ambiguous_paragraph_match','revised_paragraph':int(a)})
    serial=max((int(p.pid[1:]) for p in original.paragraphs),default=0)
    for pi in range(len(raw)):
        if pi not in para:
            serial+=1;para[pi]='P'+str(serial)
            failures.append({'type':'unmatched_paragraph','revised_paragraph':pi,'assigned_id':para[pi]})
    paragraphs=[Paragraph(para[pi],[]) for pi in range(len(raw))]
    existing={u.sid for u in old};inserted=[]
    for j,(pi,text) in enumerate(sentences):
        if j in inverse: sid=old[inverse[j][0]].sid
        else:
            serial=j+1;sid='V5N'+str(serial)
            while sid in existing: serial+=1;sid='V5N'+str(serial)
            inserted.append(sid);existing.add(sid)
        paragraphs[pi].units.append(Unit(sid,text,[],''))
    doc=Document(paragraphs,['']+['\n\n']*(len(paragraphs)-1),'',original.analyzer_version)
    normalize_render(doc);analysis.refresh(doc,{p.pid for p in doc.paragraphs})
    if key(doc.text)!=key(revised_text): raise ValueError('Alignment changed non-whitespace output content')
    return doc, {'method':'monotone lexical alignment + unique exact and mutual edited move detection',
        'initial_monotone_pairs':baseline,'matches':matches,'deleted_or_unmatched_originals':[old[i].sid for i in removed],
        'inserted_or_unmatched_revisions':inserted,'failures':failures,'alignment_failed':bool(failures),
        'no_private_source_or_record':True, 'nonwhitespace_text_preserved':True}
