"""Luna-v2 surface alignment and factual notices over native Bareun features."""
from collections import Counter
from difflib import SequenceMatcher
import re

from verak.v3.corrupt.document import Document, Paragraph, Unit, MARKER
from verak.v4.environment_v43 import BoundaryParagraphs, sentence_spans
from verak.v4.common import load_config
from .luna_v2_data import ROOT, paragraphs

PHI = ('phi_c', 'phi_j', 'phi_z_subject', 'phi_z_reference', 'phi_r')
REFERENCES = re.compile(r'(?<![가-힣])(?:이것|그것|이런|그런|이러한|그러한|이처럼|그처럼|이때|그때|이들|그들|이를|그를|이|그)(?=$|\s|[은는이가을를의도,])')


def alignment(old, new):
    """Exact moves plus monotone 1:1 / 1:2 / 2:1 / 1:3 / 3:1 blocks.

    Ambiguous group matches are exposed, never silently called exact lineage.
    This same function defines the modification ratio in every condition.
    """
    old_counts, new_counts = Counter(old), Counter(new)
    exact = {j: old.index(s) for j, s in enumerate(new) if old_counts[s] == new_counts[s] == 1}
    oi = [i for i in range(len(old)) if i not in exact.values()]
    nj = [j for j in range(len(new)) if j not in exact]
    n, m = len(oi), len(nj)
    dp = {(0, 0): (0., [])}
    for i in range(n+1):
        for j in range(m+1):
            if (i, j) not in dp: continue
            cost, path = dp[i, j]
            options = []
            if i < n: options.append((1, 0, .7, 0.))
            if j < m: options.append((0, 1, .7, 0.))
            for a, b in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)):
                if i+a > n or j+b > m: continue
                ids, jds = oi[i:i+a], nj[j:j+b]
                if any(y != x+1 for x, y in zip(ids, ids[1:])) or any(y != x+1 for x, y in zip(jds, jds[1:])): continue
                similarity = SequenceMatcher(None, ''.join(old[x] for x in ids), ''.join(new[x] for x in jds), autojunk=False).ratio()
                if similarity < (.30 if a == b == 1 else .65): continue
                options.append((a, b, 1-similarity+.14*(a+b-2), similarity))
            for a, b, extra, similarity in options:
                key = i+a, j+b
                if key not in dp or cost+extra < dp[key][0]:
                    dp[key] = cost+extra, path+[{'old': oi[i:i+a], 'new': nj[j:j+b], 'similarity': similarity}]
    matches = [{'old': [i], 'new': [j], 'similarity': 1.} for j, i in exact.items()] + dp[n, m][1]
    unchanged = {i for x in matches if len(x['old']) == len(x['new']) == 1 and old[x['old'][0]] == new[x['new'][0]] for i in x['old']}
    return {'blocks': sorted(matches, key=lambda x: (x['new'] or [m+1])[0]),
            'changed_original': len(old)-len(unchanged), 'original_sentences': len(old),
            'changed_sentence_ratio': (len(old)-len(unchanged))/max(1, len(old)),
            'uncertain_blocks': [x for x in matches if x['old'] and x['new'] and x['similarity'] < .5]}


def numbered(doc):
    return '\n\n'.join(p.pid+'\n'+'\n'.join(u.sid+' '+u.text for u in p.units) for p in doc.paragraphs)


def paragraph_text(p):
    return ' '.join(u.text for u in p.units)


class Environment:
    def __init__(self, analysis=None):
        from verak.v4.dryrun_common import local_analysis_environment
        local_analysis_environment()
        self.analysis = analysis or BoundaryParagraphs(load_config(), cache_dir=ROOT/'bareun_cache')

    def document(self, text):
        ps, serial = [], 0
        for i, p in enumerate(paragraphs(text), 1):
            units = []
            for surface, tokens, leading in self.analysis.pieces(p):
                serial += 1; units.append(Unit(f'S{serial}', surface, tokens, leading))
            ps.append(Paragraph(f'P{i}', units))
        return Document(ps, ['']+['\n\n']*(len(ps)-1), '', 'native-bareun')

    def rewrite(self, doc, pid, text, serial):
        if len(paragraphs(text)) != 1: raise ValueError('한 문단만 출력해야 합니다.')
        out = doc.clone(); p = next(p for p in out.paragraphs if p.pid == pid)
        pieces = self.analysis.pieces(text.strip())
        audit = alignment([u.text for u in p.units], [x[0] for x in pieces])
        parents = {j: [p.units[i].sid for i in b['old']] for b in audit['blocks'] for j in b['new']}
        used, lineage, units = set(), {u.sid: [u.sid] for u in out.units}, []
        for j, (surface, tokens, leading) in enumerate(pieces):
            ancestors = parents.get(j, [])
            sid = ancestors[0] if ancestors and ancestors[0] not in used else f'N{serial}_{j}'
            used.add(sid); lineage[sid] = ancestors
            units.append(Unit(sid, surface, tokens, leading))
        p.units = units
        affected = self.affected(out, {pid})
        self.analysis.refresh(out, {pid} | {out.paragraphs[out.locate(s)[0]].pid for s in affected})
        return out, lineage, affected, audit

    @staticmethod
    def affected(doc, pids):
        ids = set()
        for i, p in enumerate(doc.paragraphs):
            if p.pid not in pids: continue
            ids.update(u.sid for u in p.units)
            if i: ids.add(doc.paragraphs[i-1].units[-1].sid)
            if i+1 < len(doc.paragraphs): ids.add(doc.paragraphs[i+1].units[0].sid)
        return ids

    def structure(self, doc, action):
        ids = [s for p in action['paragraphs'] for s in p] + action['deleted']
        if Counter(ids) != Counter(u.sid for u in doc.units): raise ValueError('문장 번호는 정확히 한 번씩 나와야 합니다.')
        if not action['paragraphs'] or any(not p for p in action['paragraphs']): raise ValueError('빈 문단은 허용하지 않습니다.')
        bank = {u.sid: u for u in doc.clone().units}
        out = Document([Paragraph(f'P{i+1}', [bank[s] for s in p]) for i, p in enumerate(action['paragraphs'])], ['']+['\n\n']*(len(action['paragraphs'])-1))
        self.analysis.refresh(out, {p.pid for p in out.paragraphs})
        return out, {u.sid: [u.sid] for u in out.units}, {u.sid for u in out.units}

    def repair(self, doc, sid, text):
        if '\n' in text or len(sentence_spans(text)) != 1: raise ValueError('한국어 수리는 대상 한 문장만 고쳐야 합니다.')
        out = doc.clone(); pi, si, unit = out.locate(sid)
        surface, tokens, _ = self.analysis.pieces(text)[0]
        unit.text = surface; unit.tokens = tokens
        self.analysis.refresh(out, {out.paragraphs[pi].pid})
        return out

    @staticmethod
    def repair_scope(before, after, sid, essay):
        """Conservative lexical guard; native omitted-subject/reference context.

        Predicates/content nouns must survive in order. Extra nouns are permitted
        only at a dependent subject/reference site and must occur in the essay.
        """
        old, new = before.locate(sid)[2], after.locate(sid)[2]
        content = lambda u: [t.form for t in u.tokens if t.tag in {'NNG', 'NNP', 'VV', 'VA'}]
        a, b = content(old), content(new)
        annotation = next(x for x in before.structure().annotations if x.sid == sid)
        grounded = {t.form for u in essay.units for t in u.tokens if t.tag in {'NNG','NNP','VV','VA'}}
        for op, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == 'equal': continue
            if op != 'insert': return '대상 문장의 기존 내용어가 삭제되거나 교체되었습니다.'
            if not (annotation.subject_omitted or REFERENCES.search(old.text)):
                return '주어·지시어 수리에 해당하지 않는 내용어가 추가되었습니다.'
            if any(word not in grounded for word in b[k:l]): return '원문에 없는 내용어가 추가되었습니다.'
            added_predicates = {t.form for t in new.tokens if t.tag in {'VV','VA'}} - {t.form for t in old.tokens if t.tag in {'VV','VA'}}
            if added_predicates: return '새 서술어가 추가되었습니다.'
        return None

    @staticmethod
    def notices(before, after, lineage, affected, style):
        old = {a.sid: a for a in before.structure().annotations}
        new = {a.sid: a for a in after.structure().annotations}
        notices = []
        for a in new.values():
            if a.sid not in affected: continue
            sources = [old[s] for s in lineage.get(a.sid, []) if s in old]
            b = sources[0] if sources else None
            previous_old = old.get(b.predecessor_id) if b else None
            previous_new = new.get(a.predecessor_id)
            old_prev = previous_old.text if previous_old else '없음'
            new_prev = previous_new.text if previous_new else '없음'
            context_changed = old_prev != new_prev
            def add(kind, message, **extra):
                notices.append({'phi': kind, 'sid': a.sid, 'message': message,
                                'source_sids': [s.sid for s in sources], **extra})
            def ecs(anns):
                return [(c['form'], c['coarse_class']) for x in anns for c in x.connectives if c.get('eligible')]
            ec_old, ec_new = ecs(sources), ecs([a])
            if ec_old != ec_new:
                add('phi_c', f"이 문장의 연결어미가 '{ec_old or '없음'}'에서 '{ec_new or '없음'}'로 바뀌었습니다.", before=ec_old, after=ec_new)
            if context_changed and a.initial_conj and a.initial_conj.get('eligible'):
                add('phi_j', f"이 문장의 '{a.initial_conj['form']}'가 받는 앞 문장이 \"{old_prev}\"에서 \"{new_prev}\"로 바뀌었습니다.", before=old_prev, after=new_prev)
            if context_changed and a.subject_omitted:
                topics = [e['surface'] for e in previous_new.subject_evidence] if previous_new else []
                old_topics = [e['surface'] for e in previous_old.subject_evidence] if previous_old else []
                certainty = '확실' if topics else '불확실'
                msg = (f"이 문장은 주어가 생략되어 있습니다. 바로 앞 문장의 주제어가 '{', '.join(old_topics) or '추정 불가'}'에서 '{', '.join(topics)}'로 바뀌어, 생략된 주어가 '{', '.join(topics)}'로 읽힐 수 있습니다." if topics else '이 문장은 주어가 생략되어 있고 바로 앞 문장이 바뀌었습니다. 생략된 주어가 무엇으로 읽히는지 확인해 주세요.')
                add('phi_z_subject', msg, certainty=certainty, before=old_topics, after=topics, antecedent_identity_asserted=False)
            refs = sorted(set(REFERENCES.findall(a.text)))
            if context_changed and refs:
                add('phi_z_reference', f"이 문장의 '{', '.join(refs)}'가 가리킬 수 있는 바로 앞 문장이 \"{old_prev}\"에서 \"{new_prev}\"로 바뀌었습니다.", before=old_prev, after=new_prev)
            allowed = {'합니다', '해요'} if style == '높임체' else {'한다', '해'}
            mismatch = a.style != 'unknown' and a.style not in allowed
            changed_style = b and b.style != a.style and a.style != 'unknown'
            if changed_style or (mismatch and (not b or a.text != b.text)):
                ending = (a.final_ending or {}).get('form', '없음')
                msg = f"이 문장의 종결어미 '{ending}'가 글 전체의 문체({style})와 다릅니다." if mismatch else f"이 문장의 종결어미 문체가 '{b.style}'에서 '{a.style}'로 바뀌었습니다. 글 전체의 문체는 {style}입니다."
                add('phi_r', msg, before=b.style if b else None, after=a.style, mismatch=mismatch)
        return notices

    @staticmethod
    def safety(original, candidate, upper):
        reasons = []
        if Counter(MARKER.findall(original.text)) != Counter(MARKER.findall(candidate.text)):
            reasons.append(('anonymization', '가림 표시의 문자열과 개수가 바뀌었습니다.'))
        def numbers(s): return set(re.findall(r'\d+(?:[.,]\d+)*(?:%|퍼센트)?', MARKER.sub('', s)))
        fresh = numbers(candidate.text)-numbers(original.text)
        if fresh: reasons.append(('new_number', '원문에 없는 숫자가 생겼습니다: '+', '.join(sorted(fresh))))
        names = lambda d: {t.form for u in d.units for t in u.tokens if t.tag == 'NNP' and t.form not in MARKER.findall(d.text)}
        new_names = names(candidate)-names(original)
        # An existing surface may be retagged NNG→NNP after rewriting; it is not new content.
        new_names = {n for n in new_names if n not in original.text}
        if new_names: reasons.append(('new_proper_noun', '원문에 없는 고유명사가 생겼습니다: '+', '.join(sorted(new_names))))
        audit = alignment([u.text for u in original.units], [u.text for u in candidate.units])
        if audit['changed_sentence_ratio'] > upper:
            reasons.append(('modification_limit', f"원래 문장 {audit['original_sentences']}개 중 {audit['changed_original']}개를 바꿨습니다. 고칠 점과 관계없는 문장은 그대로 두세요."))
        return reasons, audit
