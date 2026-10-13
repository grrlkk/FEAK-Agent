"""Opt-in FAST1 execution constraints; frozen worker prompts/weights are unchanged."""
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import math
import re
import json

from verak.v3.env.protocol import ActionError
from verak.v4.common import sha_text
from verak.v4.content_env import dumps, parse_json
from verak.v4.environment_v45 import match_span, whitespace_view
from verak.v4.environment_v46 import TerminalExecutor
from verak.v4.render_v43 import assert_rendered, render
from .cooperate_env import CooperationEnvironment

VERSION = 'FAST1_DECIDE2_20261014'
NUMBERS = re.compile(r'(?<!\d)\d+(?:[.,]\d+)*(?!\d)')
WORDS = re.compile(r'[\w]+', re.UNICODE)


def lenient_span(text, old, threshold=.85):
    """Search within one named sentence; never silently choose tied spans."""
    exact, diagnostic = match_span(text, old)
    if exact is not None:
        return exact, diagnostic
    if not isinstance(old, str) or not old.strip():
        return None, diagnostic
    source, offsets = whitespace_view(text)
    target, _ = whitespace_view(old)
    ratio = threshold / (2-threshold)
    low = max(1, math.ceil(len(target)*ratio))
    high = min(len(source), math.floor(len(target)/ratio))
    # All word-boundary spans plus character windows suggested by matching blocks.
    boundaries = {0, len(source)}
    for match in re.finditer(r'\S+', source):
        boundaries.update(match.span())
    spans = {(a,b) for a in boundaries for b in boundaries if low <= b-a <= high}
    matcher = SequenceMatcher(None, target, source, autojunk=False)
    starts = set()
    radius = max(1, len(target)-low)
    for block in matcher.get_matching_blocks():
        if block.size:
            anchor = block.b-block.a
            starts.update(range(max(0,anchor-radius), min(len(source),anchor+radius+1)))
    for start in starts:
        for length in range(low, high+1):
            if start+length <= len(source):
                spans.add((start,start+length))
    scored = []
    for start,end in spans:
        score = SequenceMatcher(None,target,source[start:end],autojunk=False).ratio()
        if score >= threshold:
            scored.append((score,start,end))
    scored.sort(reverse=True)
    best = scored[0][0] if scored else None
    winners = [(a,b) for score,a,b in scored if score == best]
    diagnostic = {'mode':'lenient', 'threshold':threshold, 'similarity':best,
                  'best_matches':len(winners), 'matched':len(winners)==1,
                  'requested_old':old}
    if len(winners) != 1:
        return None, diagnostic
    start,end = winners[0]
    span = offsets[start][0], offsets[end-1][1]
    diagnostic.update(original_span=list(span), matched_text=text[span[0]:span[1]])
    return span, diagnostic


class LenientExecutor(TerminalExecutor):
    def edit(self, document, args, role):
        target = args['target']
        if ':' not in target or target.startswith(('before:', 'after:')):
            return super().edit(document,args,role)
        public, _, old = target.partition(':')
        unit = document.locate(self.sid(public,document))[2]
        span, diagnostic = lenient_span(unit.text,old)
        self.last_span_match = diagnostic
        if span is None:
            raise ActionError('substring','문장 안 유일한 구절 일치가 없습니다 (similarity>=0.85).')
        actual = unit.text[slice(*span)]
        if unit.text.count(actual) != 1:
            raise ActionError('substring','동일한 구절이 여러 곳에 있어 위치를 확정할 수 없습니다.')
        try:
            return super().edit(document,{**args,'target':public+':'+actual},role)
        finally:
            self.last_span_match = diagnostic


def repeated_words(text):
    words = WORDS.findall(text)
    return sorted({a for a,b in zip(words,words[1:]) if a==b})


def proper_nouns(document):
    return {t.form for u in document.units for t in u.tokens if t.tag=='NNP'}


class SafetyGate:
    """Deterministic feature gates calibrated once on saved, evaluated edits."""
    def __init__(self, original, ppl, *, max_log_ppl_rise=math.log(2)):
        self.original = original.clone()
        self.original_numbers = set(NUMBERS.findall(original.text))
        self.original_names = proper_nouns(original)
        self.ppl = ppl
        self.max_log_ppl_rise = max_log_ppl_rise

    def assess(self, before, after, changed):
        old = {u.sid:u for u in before.units}
        new = {u.sid:u for u in after.units}
        structure = before.structure()
        counts = Counter(a.style for a in structure.annotations if a.style not in {'unknown','mixed'})
        dominant = counts.most_common(1)[0][0] if counts else 'unknown'
        # A plurality with <=50% is not an essay majority.
        if counts and counts[dominant]*2 <= sum(counts.values()):
            dominant = 'unknown'
        annotations = {a.sid:a for a in after.structure().annotations}
        features = []
        for sid in sorted(changed):
            if sid not in new or sid in old and old[sid].text==new[sid].text:
                continue
            unit = new[sid]
            parent = sid
            while parent not in old and parent[-1:].isalpha():
                parent = parent[:-1]
            previous = old[parent].text if parent in old else before.text
            repetitions = repeated_words(unit.text)
            # "Adds" excludes unchanged material inherited from a saved
            # earlier edit. Every newly introduced number/name must still
            # occur in the raw original. Live accepted edits cannot seed an
            # unsupported fact, because its first introduction is rejected.
            inherited = old[parent].text if parent in old else ''
            numbers = sorted(set(NUMBERS.findall(unit.text))-self.original_numbers
                             -set(NUMBERS.findall(inherited)))
            names = sorted({t.form for t in unit.tokens if t.tag=='NNP'}-self.original_names)
            names = [n for n in names if n not in self.original.text and n not in inherited]
            style = annotations[sid].style
            mismatch = dominant != 'unknown' and style not in {dominant,'unknown','mixed'}
            before_ppl = self.ppl(previous)
            after_ppl = self.ppl(unit.text)
            rise = after_ppl['mean_nll']-before_ppl['mean_nll']
            reasons = []
            if repetitions: reasons.append('immediate_repeated_word')
            if numbers: reasons.append('new_number')
            if names: reasons.append('new_proper_noun')
            if mismatch: reasons.append('majority_style_mismatch')
            if rise > self.max_log_ppl_rise: reasons.append('sharp_base_perplexity_rise')
            features.append({'sid':sid,'before':previous,'after':unit.text,
                'repeated_words':repetitions,'new_numbers':numbers,'new_proper_nouns':names,
                'majority_style':dominant,'after_style':style,'style_mismatch':mismatch,
                'base_before':before_ppl,'base_after':after_ppl,'log_ppl_rise':rise,'reasons':reasons})
        return {'features':features,'reasons':sorted({r for f in features for r in f['reasons']}),
                'threshold_log_ppl_rise':self.max_log_ppl_rise}


def notice_sids(notices):
    result = set()
    for n in notices:
        if n.get('sid'): result.add(n['sid'])
        result.update(n.get('result_sids',[]))
    return result


class DecideEnvironment(CooperationEnvironment):
    def __init__(self,*args,safety=None,**kwargs):
        super().__init__(*args,search_enabled=False,**kwargs)
        self.executor = LenientExecutor(self.analysis,self.executor.sentences,self.executor.paragraphs)
        self.safety = safety
        self.current_notices = []
        self.korean_reason = None

    def delegate(self,items,number):
        super().delegate(items,number)
        self.step_limit = 4

    def start_korean(self,items=(),*,scope=(),notices=(),reason='after_revision'):
        super().start_korean(items)
        self.step_limit = 4
        self.scope = set(scope) & {u.sid for u in self.document.units}
        self.current_notices = deepcopy(list(notices))
        self.korean_reason = reason
        self.handoff['marker_change_notices'] = deepcopy(self.current_notices)
        if not self.scope:
            self.closed = True

    def public_messages(self,role,steps_left=None):
        messages = super().public_messages(role,steps_left)
        payload = assert_rendered(messages)
        payload['fast1'] = {'version':VERSION,'actions_per_item':4,
            'no_op_EDIT_ends_item_done':True,'lenient_EDIT_min_similarity':.85,
            'search_available':False,'new_facts':'writer_notes only'}
        if role == 'korean':
            payload['scope'] = self.korean_reason
            payload['whole_essay_typo_spacing_pass'] = False
            payload['allowed_scope'] = sorted(self.scope)
            payload['notices'] = deepcopy(self.current_notices)
            payload['task'] = 'allowed_scope의 문장만 형식과 연결을 검토한다. 변화 알림은 오류 판정이 아니다. 다른 문장은 문맥이다. 필요 없으면 STOP done한다.'
        return render(payload,messages[0]['content'])

    def reject(self,value,role,reason):
        self.steps_used += 1
        self.rejected_in_a_row += 1
        digest = sha_text(self.document.text)
        record = {'action':value.get('action','INVALID'),'value':deepcopy(value),
            'valid':False,'text_changed':False,'before_hash':digest,'after_hash':digest,
            'changed_sids':[],'marker_change_notices':[],'error':reason}
        self.actions[role].append(record)
        self.track_rejection(record)
        self.last_result = deepcopy(record)
        return record

    def track_rejection(self,record):
        canonical=json.dumps(record.get('value'),sort_keys=True,ensure_ascii=False)
        self.stage_rejections.setdefault((self.state_key(),canonical),{
            'action':deepcopy(record.get('value')),'error':record.get('error','rejected')})
        if self.rejected_in_a_row>=3:
            self.closed=True
            self.controller_stop={'action':'STOP','status':'blocked','controller_generated':True,
                'reason':'three_consecutive_rejections','summary':'연속 거절3회로 항목 종료',
                'issues':[record.get('error','rejected')],'failure':True}
            record['controller_stop']=deepcopy(self.controller_stop)

    def step(self,raw,role):
        if self.closed or role != self.active_role or not self.remaining:
            raise RuntimeError('No active worker item')
        try: value = parse_json(raw)
        except (ValueError,TypeError): value = None
        if isinstance(value,dict) and value.get('action')=='SEARCH':
            return self.reject(value,role,'search_disabled')
        if role=='korean' and isinstance(value,dict) and value.get('action')=='EDIT' and value.get('sentence') not in self.scope:
            return self.reject(value,role,'outside_notified_or_final_changed_sentence_scope')
        # Preserve every state side effect so a rejected safety candidate never leaks.
        keys = ('document','relations','undo','notices','citation_undo','lineage','scope','insert_count')
        prior = {k:deepcopy(getattr(self,k)) for k in keys}
        aliases, serial = deepcopy(self.executor.sentences), self.executor.serial
        search = self.search.snapshot()
        rejected = self.rejected_in_a_row
        record = super().step(raw,role)
        # Existing role/scope/mask/span checks must pass before accepting a no-op.
        if (not record['valid'] and record['action']=='EDIT'
                and '바꾸지 않는 행동' in record.get('error','')):
            record.update(valid=True,noop_EDIT=True,item_done=True,text_changed=False,
                          no_op_reason=record.pop('error'))
            self.closed = True
            self.controller_stop = None
            record.pop('controller_stop',None)
            self.rejected_in_a_row = 0
        if record['valid'] and record['text_changed'] and self.safety:
            try:
                diagnostic = self.safety.assess(prior['document'],self.document,record['changed_sids'])
            except Exception as exc:
                diagnostic = {'features':[],'reasons':['safety_measurement_unavailable'],
                              'error':type(exc).__name__+': '+str(exc)}
            record['safety'] = diagnostic
            if diagnostic['reasons']:
                record['candidate_text'] = self.document.text
                record['candidate_changed_sids'] = record['changed_sids']
                for k,v in prior.items(): setattr(self,k,v)
                self.executor.sentences, self.executor.serial = aliases, serial
                self.search.restore(search)
                self.rejected_in_a_row = rejected+1
                record.update(valid=False,text_changed=False,changed_sids=[],marker_change_notices=[],
                    after_hash=record['before_hash'],error='safety:'+','.join(diagnostic['reasons']),
                    rejection_reasons=diagnostic['reasons'])
                self.closed = False
                self.controller_stop = None
                self.track_rejection(record)
        self.last_result = {k:deepcopy(v) for k,v in record.items()
                            if k not in {'candidate_text','before_hash','after_hash','value'}}
        return record
