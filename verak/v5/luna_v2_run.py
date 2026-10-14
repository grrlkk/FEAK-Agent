"""Prompt-only Luna v2. Durable episodes, preparation gates, no training."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time

import httpx
import numpy as np

from verak.v4.common import read_json, write_json, sha_text, file_sha
from .api import PaidAPI
from .luna_v2_data import ROOT, SPEC, CONDITIONS, prepare, initialize_contract
from .luna_v2_env import Environment, PHI, numbered, paragraph_text, alignment

TEMPLATES = read_json(Path(__file__).with_name('luna_v2_prompts.json'))
_score_lock = threading.Lock()


def render(name, replacements):
    text = TEMPLATES[name]
    # Replacement strings are not recursively interpolated.
    if not replacements: return text
    return re.sub('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)),
                  lambda m: str(replacements[m.group()]), text)


def parse_json(raw):
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    value = json.loads(raw)
    if not isinstance(value, dict): raise ValueError('JSON object required')
    return value


def score(question, text):
    with _score_lock:
        r = httpx.post('http://127.0.0.1:8064', json={'question': question, 'text': text}, timeout=600)
    if r.status_code != 200: raise RuntimeError('Reference scorer: '+r.text[:1000])
    value = r.json()
    return {**value, 'q': value['mean']}


def score_text(result):
    # The frozen scorer's positions use these labels for every genre.
    labels = ['과제충실성', '설명명료성', '설명구체성', '설명적절성', '문장연결성', '글통일성', '어휘적절성', '어법적절성']
    return '\n'.join(f'{k}: {v:.4f}' for k, v in zip(labels, result['expected']))


def p1(row, scores, condition):
    text = render('P1', {'{문항}': row['question'], '{만점}': 9, '{루브릭: 점수}': score_text(scores), '{글. 문단은 빈 줄로 구분}': row['text']})
    if condition == 'A+': text = text.replace('- 문단은 빈 줄로 구분하고', TEMPLATES['Aplus']+'\n- 문단은 빈 줄로 구분하고')
    return text


def korean_prompt(doc, sid, plan, notices, condition):
    i = next(i for i, u in enumerate(doc.units) if u.sid == sid)
    template = TEMPLATES['P5']
    # The §2 observation table requires current paragraph; §10's skeleton omitted it.
    template = template.replace('[앞 문장]', '[현재 문단] {현재 문단}\n\n[앞 문장]', 1)
    if condition == 'C':
        template = template.replace(' 글의 다른 부분이 고쳐지면서 [문장]과 앞뒤 문장의 연결이 바뀌었을 수 있습니다.', '')
        template = template.replace('[바뀐 점]\n- {알림 1}\n- {알림 2}\n\n', '')
    replacements = {'[앞 문장] {…}': '[앞 문장] '+(doc.units[i-1].text if i else '없음'),
                    '[문장] {…}': '[문장] '+doc.units[i].text,
                    '[뒤 문장] {…}': '[뒤 문장] '+(doc.units[i+1].text if i+1 < len(doc.units) else '없음'),
                    '{현재 문단}': paragraph_text(doc.paragraphs[doc.locate(sid)[0]]),
                    '{style}': plan['style'], '{referents}': ', '.join(plan['referents']),
                    '- {알림 1}\n- {알림 2}': '\n'.join('- '+n['message'] for n in notices)}
    return re.sub('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)), lambda m: replacements[m.group()], template)


class Episode:
    def __init__(self, row, condition, config, phase='prep'):
        self.row, self.condition, self.config, self.phase = row, condition, config, phase
        self.env = Environment(); self.api = PaidAPI('luna', root=ROOT)
        self.path = ROOT/phase/condition/row['source_id'].replace(':', '_')
        self.path.mkdir(parents=True, exist_ok=True)
        self.events, self.calls, self.serial = [], [], 0
        self.started = time.monotonic()

    def log(self, kind, **values):
        event = {'kind': kind, 'elapsed_s': time.monotonic()-self.started, **values}
        self.events.append(event)
        with (self.path/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False)+'\n'); stream.flush()

    def call(self, role, prompt, max_output=2048, nonce=''):
        # Reuse identical observations at temperature 0 across arms. Logical calls
        # remain counted; cache replays and actually billed calls are distinct.
        identity = self.row['source_id']+':'+role+':'+sha_text(prompt)+':'+nonce
        before = time.monotonic()
        result = self.api.request([{'role': 'user', 'content': prompt}], item_id=identity, max_output=max_output)
        call = {'role': role, 'observation': prompt, 'action': result['raw'],
                'fingerprint': result['fingerprint'], 'usage': result['usage'], 'cost_usd': result['confirmed_usd'],
                'elapsed_s': time.monotonic()-before, 'original_dispatch_s': result['elapsed_s']}
        self.calls.append(call); self.log('call', **call)
        return result['raw']

    def plan(self, doc, scores):
        prompt = render('P2', {'{문항}': self.row['question'], '{만점}': 9,
                              '{루브릭: 점수}': score_text(scores), 'P1\nS1 {…}\nS2 {…}\n\nP2\nS3 {…}\n…': numbered(doc)})
        raw = self.call('orchestrator_plan', prompt)
        try:
            plan = parse_json(raw)
            if not isinstance(plan.get('thesis'), str) or plan.get('style') not in {'평서체', '높임체'}: raise ValueError('Invalid profile')
            if not isinstance(plan.get('referents'), list) or not all(isinstance(x, str) for x in plan['referents']): raise ValueError('Invalid referents')
            pids = [p.pid for p in doc.paragraphs]
            if not isinstance(plan.get('roles'), dict) or set(plan['roles']) != set(pids): raise ValueError('Missing paragraph roles')
            if not 1 <= len(plan.get('targets', [])) <= 3: raise ValueError('Expected 1–3 targets')
            targets = [t['id'] for t in plan['targets']]
            if len(set(targets)) != len(targets) or any(t not in pids for t in targets): raise ValueError('Unknown/duplicate target')
            if any(not isinstance(t.get('instruction'), str) for t in plan['targets']): raise ValueError('Invalid instruction')
            if plan.get('structure') is not None and not isinstance(plan['structure'], str): raise ValueError('Invalid structure')
            self.log('plan_parse', passed=True); return plan
        except (ValueError, KeyError, TypeError) as exc:
            self.log('plan_parse', passed=False, error=str(exc)); raise

    def rewrite_prompt(self, doc, pid, plan, instruction, reasons):
        i = next(i for i, p in enumerate(doc.paragraphs) if p.pid == pid)
        prompt = render('P4', {'{문항}': self.row['question'], '{thesis}': plan['thesis'], '{role}': plan['roles'].get(pid, '원래 문단의 역할'),
                              '{instruction}': instruction, '{style}': plan['style'], '[고칠 문단] {…}': '[고칠 문단] '+paragraph_text(doc.paragraphs[i]),
                              '[앞 문단 마지막 문장] {… 없으면 "없음"}': '[앞 문단 마지막 문장] '+(doc.paragraphs[i-1].units[-1].text if i else '없음'),
                              '[다음 문단 첫 문장] {… 없으면 "없음"}': '[다음 문단 첫 문장] '+(doc.paragraphs[i+1].units[0].text if i+1 < len(doc.paragraphs) else '없음')})
        if reasons:
            prompt = prompt.replace('이 문단을 다시 써 주세요.', '[이전 시도가 받아들여지지 않은 이유]\n'+'\n'.join('- '+r for r in reasons[:2])+'\n이 문제를 피해서 다시 써 주세요.\n\n이 문단을 다시 써 주세요.')
        return prompt

    def meaning(self, old_text, new_text, plan, role):
        prompt = render('P6', {'{thesis}': plan['thesis'], '{role}': role,
                              '[원래 문단] {…}': '[원래 문단] '+old_text, '[고친 문단] {…}': '[고친 문단] '+new_text})
        value = parse_json(self.call('orchestrator_meaning', prompt, 1024))
        if type(value.get('thesis_kept')) is not bool or type(value.get('role_kept')) is not bool or not isinstance(value.get('new_content'), list):
            raise ValueError('Invalid meaning judgment')
        return value

    def modification(self,old,candidate,pid):
        if pid == 'structure': return alignment([u.text for u in old.units],[u.text for u in candidate.units])
        return alignment([u.text for p in old.paragraphs if p.pid==pid for u in p.units],
                         [u.text for p in candidate.paragraphs if p.pid==pid for u in p.units])

    def inspect(self, old, candidate, lineage, affected, plan, pid, *, discovery=False):
        unchecked = self.condition == 'D' or discovery
        reasons, audit = self.env.safety(old, candidate, 1.)
        audit = self.modification(old,candidate,pid)
        if audit.get('modification_ratio',audit['changed_sentence_ratio']) > self.config['modification_limit']:
            reasons.append(('modification_limit', audit.get('message',f"원래 문장 {audit['original_sentences']}개 중 {audit['changed_original']}개를 바꿨습니다. 고칠 점과 관계없는 문장은 그대로 두세요.")))
        self.log('safety', reasons=reasons, alignment=audit, bypassed=unchecked)
        if reasons and not unchecked: return None, reasons, None
        notices = self.env.notices(old, candidate, lineage, affected, plan['style'])
        self.log('notices', notices=notices, candidate=candidate.text, affected=sorted(affected))
        targets = [u.sid for u in candidate.units if any(n['sid'] == u.sid for n in notices)]
        for sid in targets:
            notice = [n for n in notices if n['sid'] == sid]
            raw = self.call('korean', korean_prompt(candidate, sid, plan, notice, self.condition), 768)
            match = re.fullmatch(r'(고침|유지|막힘)\s*:\s*(.+)', raw.strip(), re.S)
            if not match:
                self.log('korean_parse', passed=False, sid=sid)
                # Unexecutable actions are protocol errors even in D.
                reasons.append(('korean_protocol', '한국어 에이전트의 행동 형식이 올바르지 않습니다.')); continue
            action, content = match.groups(); before = candidate.locate(sid)[2].text
            requested_action, requested_content = action, content
            if action=='고침' and content.strip()==before:
                action,content='유지','고침 출력이 현재 문장과 같아 유지로 정규화했습니다.'
            if action == '막힘': reasons.append(('korean_blocked', '이 문장만 고쳐서는 연결을 해결할 수 없습니다: '+content))
            if action == '고침':
                try:
                    repaired = self.env.repair(candidate, sid, content.strip())
                    scope_error = self.env.repair_scope(candidate, repaired, sid, old)
                    # v2 constrains the action to the dependent expression. New
                    # lexical material must be grounded in this essay.
                    newunit = repaired.locate(sid)[2]
                    newwords = {t.form for t in newunit.tokens if t.tag in {'NNG', 'NNP', 'VV', 'VA'}}
                    essaywords = {t.form for u in old.units for t in u.tokens if t.tag in {'NNG', 'NNP', 'VV', 'VA'}}
                    if (newwords-essaywords or scope_error) and (not unchecked or self.config.get('enforce_repair_scope',False)):
                        action,content='막힘',scope_error or '한국어 수리에 원문에 없는 내용어가 생겼습니다.'
                        reasons.append(('korean_blocked', content))
                    else: candidate = repaired
                except ValueError as exc:
                    action,content='막힘',str(exc); reasons.append(('korean_protocol', str(exc)))
            self.log('korean_decision',sid=sid,action=action,content=content,before=before,
                     after=candidate.locate(sid)[2].text,requested_action=requested_action,
                     requested_content=requested_content,notice=notice)
        post, _ = self.env.safety(old, candidate, 1.)
        reasons.extend(r for r in post if r not in reasons)
        post_amount=self.modification(old,candidate,pid)
        if post_amount.get('modification_ratio',post_amount['changed_sentence_ratio'])>self.config['modification_limit'] and not any(k=='modification_limit' for k,_ in reasons):
            reasons.append(('modification_limit',post_amount.get('message','한국어 수리 후 수정량 상한을 넘었습니다.')))
        if reasons and not unchecked: return None, reasons, None
        meaning = None
        if self.condition != 'D' or discovery:
            old_parts = {p.pid: paragraph_text(p) for p in old.paragraphs}
            new_parts = {p.pid: paragraph_text(p) for p in candidate.paragraphs}
            changed = [p for p in new_parts if new_parts[p] != old_parts.get(p)]
            judgements = []
            for part in (['whole'] if pid == 'structure' else changed):
                m = self.meaning(old.text if part == 'whole' else old_parts.get(part, ''), candidate.text if part == 'whole' else new_parts[part], plan, '글 전체 구성' if part == 'whole' else plan['roles'].get(part, '이음새 문단'))
                judgements.append(m)
                if not (m['thesis_kept'] and m['role_kept'] and not m['new_content']):
                    reasons.append(('meaning', '의미 또는 문단 역할이 보존되지 않았습니다: '+m.get('reason', '')))
            meaning = judgements
            if reasons and not unchecked: return None, reasons, meaning
            q0, q1 = score(self.row['question'], old.text), score(self.row['question'], candidate.text)
            delta = q1['q']-q0['q']
            self.log('quality', before=q0['q'], after=q1['q'], delta=delta, threshold=self.config['delta'], bypassed=unchecked)
            if delta <= self.config['delta']: reasons.append(('score', '채점 점수가 잡음 범위를 넘게 오르지 않았습니다.'))
        else:
            self.log('checks_bypassed', checks=['meaning', 'score_rejection', 'retry'])
        return (candidate if unchecked or not reasons else None), reasons, meaning

    def run(self, *, discovery=False):
        original = self.env.document(self.row['text']); current = original
        scores = score(self.row['question'], original.text)
        plan = self.plan(original, scores); self.log('profile', profile=plan, original=original.text, scores=scores)
        items = []
        if plan.get('structure'):
            prompt = render('P3', {'{문항}': self.row['question'], '{번호 붙은 글}': numbered(current), '{thesis}': plan['thesis'], '{structure}': plan['structure']})
            action = parse_json(self.call('revision_structure', prompt))
            try:
                candidate, lineage, affected = self.env.structure(current, action)
                accepted, reasons, _ = self.inspect(current, candidate, lineage, affected, plan, 'structure', discovery=discovery)
                self.log('structure_result', action=action, accepted=accepted is not None, reasons=reasons)
                if accepted is not None:
                    # Rebind plans by original sentence membership after reordering.
                    mapped, roles = {}, {}
                    for p in accepted.paragraphs:
                        members = Counter(original.locate(u.sid)[0] for u in p.units if u.sid in {v.sid for v in original.units})
                        if members:
                            oldpid = original.paragraphs[members.most_common(1)[0][0]].pid
                            mapped.setdefault(oldpid, p.pid); roles[p.pid] = plan['roles'][oldpid]
                        else: roles[p.pid] = '근거'
                    plan = {**plan, 'roles': roles, 'targets': [{**t, 'id': mapped[t['id']]} for t in plan['targets'] if t['id'] in mapped]}
                    current = accepted
            except ValueError as exc: self.log('structure_protocol', error=str(exc))
        targets = sorted(plan['targets'], key=lambda t: next(i for i,p in enumerate(current.paragraphs) if p.pid == t['id']))
        for item in targets:
            pid, reasons, accepted = item['id'], [], None
            before = paragraph_text(next(p for p in current.paragraphs if p.pid==pid))
            for attempt in range(1 if self.condition == 'D' or discovery else self.config['k']+1):
                prompt = self.rewrite_prompt(current, pid, plan, item['instruction'], [r[1] for r in reasons])
                count = self.config.get('N', 1)
                candidates = [self.call('revision_paragraph', prompt, 3072, nonce=f'N{count}:{j}' if count > 1 else '') for j in range(count)]
                if count > 1: self.log('candidate_diversity', pid=pid, attempt=attempt, generated=count, unique=len(set(candidates)))
                for j, raw in enumerate(candidates[:1] if self.condition == 'D' else candidates):
                    self.serial += 1
                    try:
                        candidate, lineage, affected, audit = self.env.rewrite(current, pid, raw, self.serial)
                        audit = self.modification(current,candidate,pid)
                        accepted, reasons, meaning = self.inspect(current, candidate, lineage, affected, plan, pid, discovery=discovery)
                    except (ValueError, KeyError) as exc:
                        accepted, reasons, meaning, audit = None, [('protocol', str(exc))], None, None
                    self.log('attempt', pid=pid, attempt=attempt, candidate_index=j, candidate=raw, accepted=accepted is not None, reasons=reasons, meaning=meaning, alignment=audit)
                    if accepted is not None: break
                if accepted is not None: current = accepted; break
            items.append({'pid': pid, 'attempts': attempt+1, 'kept_original': paragraph_text(next(p for p in current.paragraphs if p.pid==pid)) == before, 'reasons': reasons})
        final_score = score(self.row['question'], current.text)
        final_structure = current.structure()
        self.log('final_check', score=final_score, dominant_style=final_structure.dominant_style,
                 styles={a.sid: a.style for a in final_structure.annotations}, safety=self.env.safety(original, current, 1.)[0])
        result = {'essay_id': self.row['source_id'], 'condition': self.condition, 'phase': self.phase,
                  'original': original.text, 'final': current.text, 'plan': plan, 'items': items,
                  'scores_before': scores, 'scores_after': final_score, 'calls': self.calls,
                  'notices_by_type': dict(Counter(n['phi'] for e in self.events if e['kind']=='notices' for n in e['notices'])),
                  'korean_actions': dict(Counter(e['action'] for e in self.events if e['kind']=='korean_decision')),
                  'alignment': alignment([u.text for u in original.units], [u.text for u in current.units]),
                  'wall_seconds': time.monotonic()-self.started}
        write_json(self.path/'result.json', result); self.api.close(); return result


def probe():
    env = Environment()
    a = env.document('철수는 비가 와서 집에 갔다. 그래서 쉬었다. 이것은 좋은 선택이다.\n\n기분이 좋다.')
    b, lineage, affected, _ = env.rewrite(a, 'P1', '영희는 비가 와도 밖에 갔습니다. 그래서 푹 쉬었다. 이것은 좋은 선택이다.', 1)
    notices = env.notices(a, b, lineage, affected, '평서체')
    write_json(ROOT/'environment_probe.json', {'synthetic_fixture_not_prep_data': True, 'notices': notices})
    r = read_json(ROOT/'sources'/ (read_json(ROOT/'selection.json')['prep10'][0]['source_id'].replace(':', '_')+'.json'))
    result = score(r['question'], r['text'])
    print(json.dumps({'phi': dict(Counter(n['phi'] for n in notices)), 'score': result}, ensure_ascii=False))


def preparation():
    selection = prepare(); initialize_contract()
    started = time.monotonic()
    records, calibration = [], []
    for meta in selection['prep10']:
        row = read_json(ROOT/'sources'/(meta['source_id'].replace(':', '_')+'.json'))
        path = ROOT/'prep_discovery'/'B'/meta['source_id'].replace(':', '_')/'result.json'
        if path.exists(): result = read_json(path)
        else:
            result = Episode(row, 'B', {'delta': 0., 'modification_limit': 1., 'k': 0}, 'prep_discovery').run(discovery=True)
        records.append(result)
        print(json.dumps({'done': meta['source_id'], 'notices': result['notices_by_type'], 'calls': len(result['calls'])}), flush=True)
        cpath = ROOT/'calibration'/(meta['source_id'].replace(':', '_')+'.json')
        if cpath.exists(): calibration.append(read_json(cpath)); continue
        ep = Episode(row, 'B', {}, 'calibration')
        doc = ep.env.document(row['text']); plan = result['plan']
        # Select by position before seeing candidate scores, not by score gain.
        p = doc.paragraphs[len(doc.paragraphs)//2]
        prompt = '다음 문단의 뜻, 사실, 입장, 문장 수, 문체를 그대로 두고 표현만 가볍게 바꿔 주세요. 새로운 설명이나 이유를 보태지 말고 바꾼 문단만 출력하세요.\n\n'+paragraph_text(p)
        raw = ep.call('noise_surface_paraphrase', prompt, 3072)
        candidate, _, _, audit = ep.env.rewrite(doc, p.pid, raw, 0)
        meaning = ep.meaning(paragraph_text(p), raw, plan, plan['roles'].get(p.pid, '원래 역할'))
        safety, _ = ep.env.safety(doc, candidate, 1.)
        same_count = len(p.units) == len(candidate.paragraphs[doc.paragraphs.index(p)].units)
        good = meaning['thesis_kept'] and meaning['role_kept'] and not meaning['new_content'] and not safety and same_count
        delta = abs(score(row['question'], candidate.text)['q']-result['scores_before']['q']) if good else None
        c = {'source_id': row['source_id'], 'meaning': meaning, 'safety': safety, 'same_sentence_count': same_count,
             'accepted': good, 'abs_delta': delta, 'alignment': audit, 'original_paragraph': paragraph_text(p), 'candidate': raw}
        write_json(cpath, c); calibration.append(c); ep.api.close()
    accepted_noise = [c['abs_delta'] for c in calibration if c['accepted']]
    ratios = []
    for result in records:
        path = ROOT/'prep_discovery'/'B'/result['essay_id'].replace(':', '_')/'events.jsonl'
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['kind']=='attempt' and e.get('alignment') and e.get('meaning') and all(m['thesis_kept'] and m['role_kept'] and not m['new_content'] for m in e['meaning']):
                # Upper limit uses local paragraph rewrite ratio; whole-document
                # safety is no looser than this measured conservative bound.
                ratios.append(e['alignment']['changed_sentence_ratio'])
    types = Counter()
    for r in records: types.update(r['notices_by_type'])
    config = {'version': 'luna-v2', 'spec_sha256': file_sha(SPEC), 'conditions': CONDITIONS, 'seed': selection['seed'],
              'N': 1, 'k': 1, 'max_targets': 3,
              'delta': float(np.quantile(accepted_noise, .9)) if accepted_noise else None,
              'delta_sample_count': len(accepted_noise), 'modification_limit': float(np.quantile(ratios, .9)) if ratios else None,
              'modification_calibration_samples': len(ratios), 'phi_counts': dict(types),
              'missing_phi': [k for k in PHI if not types[k]], 'human_episode_review': 'pending',
              'feedback_provenance': 'stored assistant field; user approved unverified authorship',
              'followup': {'trigger': 'B kept-original paragraph share > 0.50', 'N': 3, 'conditions': ['B','C','D'], 'total_cap_usd': 5.},
              'wall_seconds': time.monotonic()-started,
              'deviations': ['Sonnet 5.5 temperature omitted with user approval',
                            'P5 includes current paragraph required by observation table',
                            'phi_r also reports changed final style when already consistent; facts are not error judgments',
                            'D blocked decisions are logged; executable first candidate retained without quality veto']}
    write_json(ROOT/'preparation.json', config)
    print(json.dumps(config, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('part', choices=['probe', 'prepare'])
    args = parser.parse_args()
    (probe if args.part=='probe' else preparation)()
