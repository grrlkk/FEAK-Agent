"""Observable Phase 2c changes; no claims about antecedent identity or correctness."""
from ..ko.structural import CHANGE_LEVELS

PRIORITY = {'polarity_change': 0, 'modality_change': 0, 'ec_relation_change': 1,
            'conjunction_change': 1, 'dependency_change': 2, 'subject_omission_change': 2,
            'focus_change': 3, 'style_change': 4}


def cohesion_facts(before, after, changed_sids):
    old = {a.sid: a for a in before.annotations}
    new = {a.sid: a for a in after.annotations}
    affected = set(changed_sids)
    for structure in (before, after):
        ids = [a.sid for a in structure.annotations]
        for i, sid in enumerate(ids):
            if sid in changed_sids:
                affected.update(ids[max(0, i-1):i+2])
        for edge in structure.edges:
            if edge['src'] in changed_sids or edge['dst'] in changed_sids:
                affected.update((edge['src'], edge['dst']))
    result = []
    def add(kind, sid, a, b, label):
        if a != b:
            result.append({'type': kind, 'level': CHANGE_LEVELS[kind], 'sid': sid,
                           'before': a, 'after': b, 'message': f'{sid} {label}: {a} → {b}'})
    def ec(ann):
        return [(c['form'], c['coarse_class']) for c in ann.connectives if c['eligible']]
    def conjunction(ann):
        c = ann.initial_conj
        return (c['form'], c['visible_relation'], ann.predecessor_id) if c else None
    for sid in affected:
        a, b = old.get(sid), new.get(sid)
        if b is None:
            continue
        if a:
            for kind, name, label in [('polarity_change', 'polarity', '극성 표지'),
                ('modality_change', 'modality', '양태 표지'), ('style_change', 'style', '종결 문체'),
                ('subject_omission_change', 'subject_omitted', '주어 생략 관찰')]:
                add(kind, sid, getattr(a, name), getattr(b, name), label)
            add('ec_relation_change', sid, ec(a), ec(b), '연결어미 관계 후보')
            add('conjunction_change', sid, conjunction(a), conjunction(b), '접속 표현·관계·앞 문장')
            if a.subject_omitted and b.subject_omitted:
                add('dependency_change', sid, a.predecessor_id, b.predecessor_id, '주어 생략: 앞 문장')
            add('focus_change', sid, [(p['form'], p['function']) for p in a.focus_particles],
                [(p['form'], p['function']) for p in b.focus_particles], '화제·초점 표지')
        else:
            add('style_change', sid, None, b.style, '추가 문장 종결 문체')
    return sorted(result, key=lambda r: (PRIORITY[r['type']], r['sid'], r['type']))
