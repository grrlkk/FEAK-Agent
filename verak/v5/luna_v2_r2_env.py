"""Human-reviewed preparation revision: boundary-only, native POS-based notices."""
from collections import Counter
from difflib import SequenceMatcher

from verak.v3.ko.final_style import quote_spans
from verak.v3.ko.coarse import CLOSED_CONJUNCTIONS
from .luna_v2_env import Environment, alignment

RELATION_KO = {'CONDITION':'조건', 'CAUSE':'이유', 'ADVERSATIVE':'대조·양보', 'PURPOSE':'목적'}
CONTENT = {'NNG','NNP','NNB','NP','NR','VV','VA','VX','VCP','VCN','SN','SL','SH'}
NOMINAL = {'NNG','NNP','NNB','NP','NR','SL','SH','SN','XSN'}


def josa(word, pair=('이','가')):
    text = str(word).rstrip("'\".,!? ")
    last = text[-1:] or '가'
    jong = (ord(last)-0xAC00)%28 if '가' <= last <= '힣' else None
    if jong is None:
        # Korean reading of final digits; other scripts use a neutral quoted label.
        final = {'0':True,'1':True,'2':False,'3':True,'4':False,'5':False,'6':True,'7':True,'8':True,'9':False}
        return pair[0] if final.get(last,False) else pair[1]
    if pair == ('으로','로') and jong == 8: return '로'
    return pair[0] if jong else pair[1]


def references(annotation):
    """Detect references from Bareun tags, retaining lossless source surfaces."""
    tokens = annotation.tokens
    quoted = quote_spans(annotation.text, annotation.start)
    found = []
    for i,t in enumerate(tokens):
        if any(a <= t.start < b for a,b in quoted): continue
        determiner = t.tag in {'MM','MMD'} and t.form in {'이','그','저','이런','그런','저런','이러한','그러한','저러한'}
        pronoun = t.tag in {'NP','NNB'} and t.form in {'이','그','저','이것','그것','저것','이들','그들','저들'}
        adverb = t.tag in {'MAG','MAJ'} and t.form in {'이렇게','그렇게','저렇게','이처럼','그처럼','저처럼','이에','이로'}
        derived = t.form in {'이렇','그렇','저렇','이러하','그러하','저러하'} and t.tag in {'VA','VV','VX'}
        if not (determiner or pronoun or adverb or derived): continue
        end = t.end
        # Include particles and the explicit 이로 인해 construction.
        k = i+1
        while k < len(tokens) and tokens[k].tag.startswith('J'):
            end = tokens[k].end; k += 1
        if k < len(tokens) and tokens[k].form == '인하' and tokens[k].tag in {'VV','VX'}:
            end = tokens[k].end; k += 1
            if k<len(tokens) and tokens[k].tag=='EC': end=tokens[k].end
        if derived and k<len(tokens) and tokens[k].tag in {'EC','ETM'}: end=tokens[k].end
        surface = annotation.text[t.start-annotation.start:end-annotation.start]
        found.append({'surface':surface,'span':[t.start,end],'token_form':t.form,'tag':t.tag})
    return found


def content_words(a):
    return Counter(t.form for t in a.tokens if t.tag in CONTENT)


def conjunction(a):
    if a is None: return None
    lexical=[t for t in a.tokens if t.tag not in {'SF','SP','SS','SE','SO','SW'}]
    if not lexical:return None
    t=lexical[0]
    if t.tag=='MAJ' or (t.tag=='MAG' and t.form in CLOSED_CONJUNCTIONS):
        return {'form':t.form,'tag':t.tag,'relation':CLOSED_CONJUNCTIONS.get(t.form)}
    # Native analysis can split a fixed conjunctive predicate into VA+EC.
    candidate=a.initial_conj
    if candidate and t.form in {'그렇','그러하','그러'} and any(x.tag in {'EC','ETN'} for x in lexical[:3]):
        return {'form':candidate['form'],'tag':'predicate+ending','relation':candidate.get('coarse_class')}
    return None


def trivial_predecessor_change(a,b):
    return bool(a and b and SequenceMatcher(None,a.text,b.text,autojunk=False).ratio() >= .9 and content_words(a)==content_words(b))


def generic_obligation(a):
    tokens = a.tokens
    return any((t.form.endswith('야') and t.tag in {'EC','JX'}) and any(x.form=='하' and x.tag in {'VV','VX'} for x in tokens[i+1:]) for i,t in enumerate(tokens))


def main_topic(a):
    if a is None: return None
    ts = a.tokens; quoted = quote_spans(a.text,a.start)
    # This is a conservative surface heuristic, not a dependency parser.
    clause_start = max((i+1 for i,t in enumerate(ts) if t.tag=='EC' and t.form not in {'어','아','여','게','지'}),default=0)
    candidates = []
    for i,t in enumerate(ts):
        if any(x<=t.start<y for x,y in quoted): continue
        topical = t.tag=='JX' and t.form in {'은','는','만','도'}
        subject = t.tag=='JKS' and t.form in {'이','가'} and i>=clause_start
        if not (topical or subject) or not i: continue
        k=i-1
        while k>=0 and ts[k].tag=='JX' and ts[k].form in {'은','는','만','도'}: k-=1
        if k<0 or ts[k].tag not in NOMINAL: continue
        left=k
        while left>0 and ts[left-1].tag in NOMINAL|{'XPN'} and ts[left-1].end==ts[left].start: left-=1
        surface=a.text[ts[left].start-a.start:ts[k].end-a.start]
        rank = 0 if topical and i>=clause_start else 1 if topical else 2
        candidates.append((rank, i, {'surface':surface,'marker':t.form,'span':[ts[left].start,t.end],
                                    'evidence':'주절 화제 표지 우선; 종속절 주격 표지 제외', 'heuristic':True}))
    return min(candidates,key=lambda x:(x[0],x[1]))[2] if candidates else None


def character_edits(before, after):
    """Levenshtein characters / max(len(before),len(after)); spaces included."""
    # Myers bit-vector Levenshtein, arbitrary-precision Python integers. This
    # avoids a new dependency and quadratic matrices for whole-essay structure.
    a,b=(before,after) if len(before)<=len(after) else (after,before)
    distance=len(a)
    if not a: distance=len(b)
    else:
        equality={}
        for i,c in enumerate(a):equality[c]=equality.get(c,0)|(1<<i)
        mask=(1<<len(a))-1;high=1<<(len(a)-1);positive=mask;negative=0
        for c in b:
            equal=equality.get(c,0);vertical=equal|negative
            horizontal=(((equal&positive)+positive)^positive)|equal
            plus=negative|~(horizontal|positive);minus=positive&horizontal
            if plus&high:distance+=1
            if minus&high:distance-=1
            plus=(plus<<1)|1;minus<<=1
            positive=(minus|~(vertical|plus))&mask;negative=(plus&vertical)&mask
    denominator=max(len(before),len(after),1)
    return {'distance':distance,'denominator':denominator,'ratio':distance/denominator,
            'unit':'Unicode code point; whitespace included'}


class BoundaryEnvironment(Environment):
    @staticmethod
    def affected(doc,pids):
        ids=set()
        for i,p in enumerate(doc.paragraphs):
            if p.pid not in pids: continue
            if i: ids.add(doc.paragraphs[i-1].units[-1].sid)
            if i+1<len(doc.paragraphs): ids.add(doc.paragraphs[i+1].units[0].sid)
        return ids

    def structure(self,doc,action):
        if len(action.get('deleted',[])) > .30*len(doc.units):
            raise ValueError('구성 단계 삭제는 원문 문장 수의 30% 이하여야 합니다.')
        out,lineage,_=super().structure(doc,action)
        old_positions={u.sid:(p.pid,i) for p in doc.paragraphs for i,u in enumerate(p.units)}
        # LCS-derived move detection does not count index shifts after deletion.
        surviving=[u.sid for u in doc.units if u.sid not in action['deleted']]
        new=[u.sid for u in out.units]
        ordered=set()
        for block in SequenceMatcher(None,surviving,new,autojunk=False).get_matching_blocks():
            ordered.update(new[block.b:block.b+block.size])
        moved={u.sid for p in out.paragraphs for u in p.units if u.sid not in ordered or old_positions[u.sid][0]!=p.pid}
        # Untouched neighbors of the paragraphs altered by structure also form seams.
        changed_pids={p.pid for p in out.paragraphs if [u.sid for u in p.units] != next(([u.sid for u in q.units] for q in doc.paragraphs if q.pid==p.pid),[])}
        seams=self.affected(out,changed_pids)
        return out,lineage,moved|seams

    @staticmethod
    def notices(before,after,lineage,affected,style):
        old={a.sid:a for a in before.structure().annotations}
        new={a.sid:a for a in after.structure().annotations}
        old_order=list(old);new_order=list(new)
        notices=[]
        for a in new.values():
            if a.sid not in affected: continue
            ancestors=[old[s] for s in lineage.get(a.sid,[]) if s in old]
            b=ancestors[0] if ancestors else None
            op=old.get(b.predecessor_id) if b else None; np=new.get(a.predecessor_id)
            old_prev=op.text if op else '없음'; new_prev=np.text if np else '없음'
            cause=list(dict.fromkeys(([op.sid] if op else []) + (lineage.get(np.sid,[np.sid]) if np else [])))
            def add(phi,event,message,source_sids=None,**extra):
                notices.append({'phi':phi,'event':event,'sid':a.sid,'message':message,
                                'source_sids':cause if source_sids is None else source_sids,
                                'target_source_sids':[x.sid for x in ancestors],**extra})
            def ecs(xs): return [(c['form'],RELATION_KO.get(c['coarse_class'],'관계 불확실')) for x in xs for c in x.connectives if c.get('eligible')]
            ec0,ec1=ecs(ancestors),ecs([a])
            if ec0!=ec1:
                fmt=lambda rows:', '.join(f'{x}({r})' for x,r in rows) or '없음'
                add('phi_c','ending_changed',f"이 문장의 연결어미가 '{fmt(ec0)}'에서 '{fmt(ec1)}'로 바뀌었습니다.",source_sids=[x.sid for x in ancestors])
            c0,c1=conjunction(b),conjunction(a)
            cj0=c0['form'] if c0 else None
            cj1=c1['form'] if c1 else None
            if cj0!=cj1:
                event='conjunction_added' if not cj0 else 'conjunction_removed' if not cj1 else 'conjunction_replaced'
                add('phi_j',event,f"문두 접속어가 '{cj0 or '없음'}'에서 '{cj1 or '없음'}'로 바뀌었습니다.",source_sids=[x.sid for x in ancestors])
            context_changed=old_prev!=new_prev
            if context_changed and cj1:
                add('phi_j','predecessor_changed',f"이 문장의 '{cj1}'{josa(cj1)} 받는 앞 문장이 \"{old_prev}\"에서 \"{new_prev}\"로 바뀌었습니다.")
            if context_changed and a.subject_omitted and not generic_obligation(a):
                t0,t1=main_topic(op),main_topic(np)
                if not t0 or not t1 or (t0['surface'],t0['marker'])!=(t1['surface'],t1['marker']):
                    if t1:
                        name=t1['surface']; previous=t0['surface'] if t0 else '추정 불가'
                        msg=f"이 문장은 주어가 생략되어 있습니다. 앞 문장의 주제어가 '{previous}'에서 '{name}'{josa(name,('으로','로'))} 바뀌어, '{name}'{josa(name)} 생략된 주어로 읽힐 수 있습니다."
                    else: msg='이 문장은 주어가 생략되어 있고 바로 앞 문장이 바뀌었습니다. 생략된 주어가 무엇으로 읽히는지 확인해 주세요.'
                    add('phi_z_subject','predecessor_topic_changed',msg,certainty='확실' if t1 else '불확실',before=t0,after=t1,antecedent_identity_asserted=False)
            refs=references(a)
            if context_changed and refs and not trivial_predecessor_change(op,np):
                display=', '.join(x['surface'] for x in refs)
                add('phi_z_reference','predecessor_changed',f"이 문장의 지시 표현 '{display}'{josa(display)} 가리킬 수 있는 앞 문장이 \"{old_prev}\"에서 \"{new_prev}\"로 바뀌었습니다.",references=refs)
            # Boundary text remains untouched. A newly changed neighboring style
            # can still disturb its register; don't flag a pre-existing mismatch.
            styles0=(b.style if b else 'unknown',op.style if op else 'unknown')
            styles1=(a.style,np.style if np else 'unknown')
            known=lambda x:x in {'한다','해','해요','합니다'}
            if styles0!=styles1 and all(known(x) for x in styles1) and styles1[0]!=styles1[1]:
                display={'한다':'평서체','해':'해체','해요':'해요체','합니다':'높임체'}
                ending=(a.final_ending or {}).get('form','없음')
                add('phi_r','neighbor_style_changed',f"이 문장의 종결어미 '{ending}'({display[a.style]})와 바뀐 앞 문장의 문체({display[np.style]})가 서로 다릅니다.",before=styles0,after=styles1)
            ni=new_order.index(a.sid);oi=old_order.index(b.sid) if b else -1
            next0=old[old_order[oi+1]] if b and oi+1<len(old_order) else None
            next1=new[new_order[ni+1]] if ni+1<len(new_order) else None
            pair0=(b.style if b else 'unknown',next0.style if next0 else 'unknown')
            pair1=(a.style,next1.style if next1 else 'unknown')
            if pair0!=pair1 and all(known(x) for x in pair1) and pair1[0]!=pair1[1]:
                display={'한다':'평서체','해':'해체','해요':'해요체','합니다':'높임체'}
                ending=(a.final_ending or {}).get('form','없음')
                causes=list(dict.fromkeys(([next0.sid] if next0 else [])+lineage.get(next1.sid,[next1.sid])))
                add('phi_r','following_style_changed',f"이 문장의 종결어미 '{ending}'({display[a.style]})와 바뀐 뒤 문장의 문체({display[next1.style]})가 서로 다릅니다.",source_sids=causes,before=pair0,after=pair1)
        return notices

    @staticmethod
    def repair_scope(before,after,sid,essay):
        old=before.locate(sid)[2]; new=after.locate(sid)[2]
        annotations0={a.sid:a for a in before.structure().annotations}
        annotations1={a.sid:a for a in after.structure().annotations}
        a,b=annotations0[sid],annotations1[sid]
        words={t.form for u in essay.units for t in u.tokens if t.tag in CONTENT}
        subject_spans0=[e['span'] for e in a.subject_evidence]
        subject_spans1=[e['span'] for e in b.subject_evidence]
        ref0,ref1=references(a),references(b)
        def allowed(t,ann,subjects,refs,other):
            if t.tag in {'EC','EF'} or t.tag.startswith('J'): return True
            if t.tag in {'SF','SP'}: return True
            if ann.initial_conj and t.form==ann.initial_conj['form'] and t.tag in {'MAG','MAJ'}: return True
            if any(x['span'][0]<=t.start<x['span'][1] for x in refs): return True
            if t.tag in NOMINAL and any(x<=t.start<y for x,y in subjects):
                return t.form in words and (other.subject_omitted or other.subject_evidence or references(other))
            return False
        oldkeys=[(t.form,t.tag) for t in a.tokens]; newkeys=[(t.form,t.tag) for t in b.tokens]
        for op,i,j,k,l in SequenceMatcher(None,oldkeys,newkeys,autojunk=False).get_opcodes():
            if op=='equal': continue
            changed=[(t,a,subject_spans0,ref0,b) for t in a.tokens[i:j]]+[(t,b,subject_spans1,ref1,a) for t in b.tokens[k:l]]
            if any(not allowed(*x) for x in changed):
                return '바뀐 형태소가 연결어미·접속어·주어·지시어·조사·종결어미의 허용 범위를 넘었습니다.'
        return None
