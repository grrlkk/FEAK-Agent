"""Rule-only phi differences. Positional antecedents are hints, not coreference."""
from copy import deepcopy
from difflib import SequenceMatcher
import json
import hashlib


def fingerprint(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def phi(document):
    structure=document.structure(); rows={}
    units={u.sid:u.text for u in document.units}
    for a in structure.annotations:
        pred=a.predecessor_id
        antecedent={'sid':pred,'text_sha256':fingerprint(units[pred]) if pred in units else None,
                    'interpretation':'positional_predecessor_not_resolved_coreference'}
        rows[a.sid]={'ec_categories':[c['coarse_class'] for c in a.connectives if c['eligible']],
            'conjunction_category':a.initial_conj['coarse_class'] if a.initial_conj and a.initial_conj['eligible'] else None,
            'antecedent':antecedent,'subject_omitted':a.subject_omitted,
            'omission_uncertain':a.omission_uncertain,'style':a.style,
            'majority_style':structure.dominant_style,'off_style':a.off_style}
    return rows


def changes(before,after,graph=None):
    old,new=phi(before),phi(after);notices=[]
    edges=(graph or {}).get('sentence_relations',[])
    def add(kind,sid,a,b,**detail):
        if a==b: return
        prior=old.get(sid,{});current=new[sid]
        antecedents=[x.get('sid') for x in (prior.get('antecedent',{}),current['antecedent'])]
        live={u.sid for u in after.units}
        scope=list(dict.fromkeys([sid]+[s for s in antecedents if s in live and s!=sid]))
        notice={'id':f'N{len(notices)+1}','type':kind,'sid':sid,'before':deepcopy(a),'after':deepcopy(b),
            'inserted':sid not in old,'scope':scope,'antecedent_kind':'positional_hint',
            'original_graph_relations':[deepcopy(e) for e in edges if e['source']==sid or e['target']==sid],
            'error_verdict':False}
        notice.update(detail)
        notice['message']=f'{sid} {kind}: '+json.dumps(a,ensure_ascii=False)+' → '+json.dumps(b,ensure_ascii=False)
        notices.append(notice)
    def ec_changes(sid,before,after):
        # Preserve unchanged occurrences; one notice per added, removed or
        # replaced category. Indexes are observations, not semantic judgments.
        for tag,a0,a1,b0,b1 in SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
            if tag=='equal':continue
            for offset in range(max(a1-a0,b1-b0)):
                ai=a0+offset if a0+offset<a1 else None
                bi=b0+offset if b0+offset<b1 else None
                add('ec_category',sid,before[ai] if ai is not None else None,after[bi] if bi is not None else None,
                    ec_index_before=ai,ec_index_after=bi,sentence_ec_before=deepcopy(before))
    for sid,b in new.items():
        a=old.get(sid)
        if a is None:
            ec_changes(sid,[],b['ec_categories'])
            if b['conjunction_category']:
                add('conjunction_category',sid,None,b['conjunction_category'])
                add('conjunction_antecedent',sid,None,b['antecedent'])
            add('subject_omission',sid,None,b['subject_omitted'])
            if b['subject_omitted']: add('omitted_subject_antecedent',sid,None,b['antecedent'])
            add('style_vs_majority',sid,None,{k:b[k] for k in ('style','majority_style','off_style')})
            continue
        ec_changes(sid,a['ec_categories'],b['ec_categories'])
        add('conjunction_category',sid,a['conjunction_category'],b['conjunction_category'])
        if a['conjunction_category'] or b['conjunction_category']:
            add('conjunction_antecedent',sid,a['antecedent'],b['antecedent'])
        add('subject_omission',sid,a['subject_omitted'],b['subject_omitted'])
        if a['subject_omitted'] and b['subject_omitted']:
            add('omitted_subject_antecedent',sid,a['antecedent'],b['antecedent'])
        add('style_vs_majority',sid,{k:a[k] for k in ('style','majority_style','off_style')},
            {k:b[k] for k in ('style','majority_style','off_style')})
    return notices


def resolved(notices,final):
    current=phi(final);results=[]
    for n in notices:
        a=current.get(n['sid']);status=None;rule=None
        if a is not None:
            if n['type']=='style_vs_majority':
                if a['style'] not in {'unknown','mixed',None} and a['majority_style'] not in {'unknown','mixed',None}:
                    status=a['style']==a['majority_style'];rule='style_equals_current_essay_majority'
            elif n['type']=='ec_category':
                if a['ec_categories']==n['sentence_ec_before']:
                    status=True;rule='original_EC_category_sequence_restored_not_semantic_correctness'
            elif n['before'] is not None:
                keys={'ec_category':'ec_categories','conjunction_category':'conjunction_category',
                      'subject_omission':'subject_omitted','conjunction_antecedent':'antecedent',
                      'omitted_subject_antecedent':'antecedent'}
                value=a[keys[n['type']]]
                # Matching the original marker/anchor is observable. A different
                # legitimate new relation cannot be certified by these rules.
                if value==n['before']:
                    status=True;rule='original_phi_value_restored_not_semantic_correctness'
                elif n['type']=='omitted_subject_antecedent' and not a['subject_omitted'] and not a['omission_uncertain']:
                    status=True;rule='explicit_subject_observed_not_coreference_correctness'
        results.append({'notice_id':n['id'],'type':n['type'],'resolved_by_rule':status,'rule':rule,
                        'unknown_is_not_failure_or_success':True,'sentence_survives':a is not None})
    return results
