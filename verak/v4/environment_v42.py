"""v4.2-only terminal boundaries and role-specific EDIT, with frozen v4 untouched."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import re

from verak.src.schemas import Profile,Sentence
from verak.v3.corrupt.document import Document,Unit,MARKER,protected
from verak.v3.env.actions import ActionExecutor,check_markers
from verak.v3.env.protocol import ActionError
from verak.v3.insertion_boost.resources import BoostParagraphs
from verak.v3.phase2 import restore_profile
from .content_env import dumps
from .policy_env import V4Environment

ENVIRONMENT_VERSION='v4.2_boundaries_20261010'
RULES={
    'terminal_characters':'.?!',
    'nonterminal_protection':'whole #@...# markers; decimal dots between digits; URL/domain-internal punctuation',
    'runs':'consecutive .?! form one boundary; attach closing quotes/brackets; no empty pieces',
    'rendering':'one space between sentence units; retain paragraph order and paragraph gaps',
    'korean_EDIT':'no added unprotected terminal characters and no new terminal boundary; analyzer sentence count ignored',
    'revision_EDIT':'at most two deterministic sentence pieces; first ID retained, additional derived ID and notice',
    'normalization':'initial source segmentation and whitespace normalization are baseline preparation, never agent actions',
}
URL=re.compile(r'(?:https?://|www\.)[^\s<>"\'“”()\[\]{}]+|(?<![\w@])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s<>"\'“”()\[\]{}]*)?')
CLOSERS='\"\'”’)]}»'


def protected_positions(text):
    positions=set()
    for match in MARKER.finditer(text): positions.update(range(match.start(),match.end()))
    for match in URL.finditer(text):
        end=match.end()
        while end>match.start() and text[end-1] in '.?!,;:': end-=1
        positions.update(range(match.start(),end))
    for i,c in enumerate(text):
        if c=='.' and 0<i<len(text)-1 and text[i-1].isdigit() and text[i+1].isdigit(): positions.add(i)
    return positions


def terminal_positions(text):
    ignored=protected_positions(text)
    return [i for i,c in enumerate(text) if c in '.?!' and i not in ignored]


def sentence_spans(text):
    """Deterministic original-character ranges, independent of Bareun segmentation."""
    if '\n' in text or '\r' in text: raise ValueError('A sentence unit cannot span paragraphs')
    terminals=set(terminal_positions(text));spans=[];start=0;i=0
    def append(end):
        nonlocal start
        left=start
        while left<end and text[left].isspace(): left+=1
        while end>left and text[end-1].isspace(): end-=1
        if end>left: spans.append((left,end))
    while i<len(text):
        if i not in terminals: i+=1;continue
        end=i+1
        while end<len(text) and end in terminals: end+=1
        while end<len(text) and text[end] in CLOSERS: end+=1
        append(end);start=end;i=end
    append(len(text))
    return spans


def normalize_render(document):
    for paragraph in document.paragraphs:
        for index,unit in enumerate(paragraph.units): unit.leading='' if index==0 else ' '
    return document


def normalize_document(document,analysis):
    original=document.snapshot();mapping={};lineage={};known={u.sid for u in document.units}
    for paragraph in document.paragraphs:
        units=[]
        for unit in paragraph.units:
            spans=sentence_spans(unit.text)
            if not spans: raise ValueError('Empty original sentence unit')
            pieces=[]
            for index,(start,end) in enumerate(spans):
                sid=unit.sid
                if index:
                    serial=index
                    while True:
                        n=serial;suffix=''
                        while n: n,r=divmod(n-1,26);suffix=chr(97+r)+suffix
                        sid=unit.sid+suffix
                        if sid not in known: break
                        serial+=1
                    known.add(sid)
                pieces.append(sid)
                lineage[sid]={'parent_sid':unit.sid,'source_start':start,'source_end':end,'initial':True}
                units.append(Unit(sid,unit.text[start:end],[],''))
            mapping[unit.sid]=pieces
        paragraph.units=units
    normalize_render(document)
    analysis.refresh(document,{p.pid for p in document.paragraphs})
    return {'version':ENVIRONMENT_VERSION,'original_layout':original,'normalized_layout':document.snapshot(),
        'sid_mapping':mapping,'lineage':lineage,'source_characters_preserved_except_whitespace':
        ''.join(''.join(u['text'].split()) for p in original['paragraphs'] for u in p['units'])==
        ''.join(''.join(u.text.split()) for u in document.units)}


def document_profile(document):
    sentences=[];offset=0
    for gap,paragraph in zip(document.gaps,document.paragraphs):
        offset+=len(gap)
        for unit in paragraph.units:
            offset+=len(unit.leading)
            sentences.append(Sentence(unit.sid,int(paragraph.pid[1:]),offset,offset+len(unit.text),unit.text,
                [replace(t,start=t.start+offset,end=t.end+offset) for t in unit.tokens]))
            offset+=len(unit.text)
    return Profile(sentences,[],'bareun',document.analyzer_version)


def prepared_document(row):
    profile=restore_profile(row['profile']);by_sid={s.id:s for s in profile.sentences}
    state=row['layout'];doc=Document([],list(state['gaps']),state['tail'],state['analyzer_version'])
    from verak.v3.corrupt.document import Paragraph
    for p in state['paragraphs']:
        units=[]
        for u in p['units']:
            s=by_sid[u['sid']]
            if s.text!=u['text']: raise ValueError('Prepared sentence/profile mismatch')
            units.append(Unit(u['sid'],u['text'],[replace(t,start=t.start-s.start,end=t.end-s.start) for t in s.tokens],u['leading']))
        doc.paragraphs.append(Paragraph(p['pid'],units))
    if doc.text!=row['text']: raise ValueError('Prepared v4.2 document does not round-trip')
    return doc


def normalize_row(row,analysis,*,document=None):
    if row.get('environment_version')==ENVIRONMENT_VERSION:
        prepared_document(row)
        return deepcopy(row)
    doc=document.clone() if document is not None else Document.from_profile(row['text'],restore_profile(row['profile']))
    provenance=normalize_document(doc,analysis)
    if not provenance['source_characters_preserved_except_whitespace']:
        raise ValueError('Initial normalization changed non-whitespace source content')
    return {**deepcopy(row),'original_text':row['text'],'text':doc.text,'profile':document_profile(doc).to_dict(),
        'layout':doc.snapshot(),'environment_version':ENVIRONMENT_VERSION,'normalization':provenance,
        'paragraphs':[{'id':p.pid,'sentences':[{'id':u.sid,'text':u.text} for u in p.units]} for p in doc.paragraphs]}


class BoundaryParagraphs(BoostParagraphs):
    def pieces(self,text):
        spans=sentence_spans(text)
        if not spans: raise ValueError('No nonempty sentence in EDIT')
        texts=[text[a:b] for a,b in spans];canonical=' '.join(texts)
        profile=self.profile(canonical)
        tokens=[t for s in profile.sentences for t in s.tokens]
        pieces=[];offset=0
        for index,part in enumerate(texts):
            if index: offset+=1
            end=offset+len(part)
            if any(t.start<offset<t.end or t.start<end<t.end for t in tokens):
                raise ValueError('Bareun token crosses a canonical terminal boundary')
            pieces.append((part,[replace(t,start=t.start-offset,end=t.end-offset) for t in tokens
                if offset<=t.start and t.end<=end],'' if index==0 else ' '))
            offset=end
        return pieces

    def refresh(self,document,paragraph_ids):
        normalize_render(document)
        return super().refresh(document,paragraph_ids)


class BoundaryExecutor(ActionExecutor):
    editor_role=None

    def edit(self,document,args,role):
        target,new=args['target'],args['new_text']
        if target.startswith(('before:','after:')) or ':' not in target:
            result=super().edit(document,args,role);normalize_render(document);return result
        public,_,old=target.partition(':');sid=self.sid(public,document);pi,si,unit=document.locate(sid)
        if not old or unit.text.count(old)!=1:
            raise ActionError('substring','부분 문자열이 해당 문장에 정확히 한 번 있어야 합니다.')
        start=unit.text.index(old)
        if protected(unit.text,start,start+len(old)):
            raise ActionError('anonymization','익명화 표지 내부는 편집할 수 없습니다.')
        replacement=unit.text[:start]+new+unit.text[start+len(old):]
        if not replacement.strip(): raise ActionError('role_forbidden','EDIT는 문장을 삭제할 수 없습니다.')
        check_markers(unit.text,replacement)
        spans=sentence_spans(replacement)
        if self.editor_role=='korean':
            before=Counter(unit.text[i] for i in terminal_positions(unit.text))
            after=Counter(replacement[i] for i in terminal_positions(replacement))
            if after-before or len(spans)>1:
                raise ActionError('new_boundary','KOREAN EDIT는 종결부호나 문장 경계를 추가할 수 없습니다.')
        elif self.editor_role=='revision':
            if len(spans)>2: raise ActionError('split_limit','Revision EDIT는 두 문장까지만 나눌 수 있습니다.')
        else: raise ValueError('v4.2 EDIT needs an explicit editor role')
        pieces=self.analysis.pieces(replacement)
        rewritten=[Unit(sid if i==0 else self.fresh(sid),text,tokens,unit.leading if i==0 else leading)
                   for i,(text,tokens,leading) in enumerate(pieces)]
        document.paragraphs[pi].units[si:si+1]=rewritten
        normalize_render(document)
        return {u.sid for u in rewritten},{document.paragraphs[pi].pid}

    def move(self,document,args,role):
        result=super().move(document,args,role);normalize_render(document);return result


class V42Environment(V4Environment):
    def __init__(self,row,analysis,*,document=None):
        from .policy_prompts_v42 import verify_frozen
        verify_frozen();super().__init__(row,analysis)
        if document is not None:
            self.document=document.clone();self.normalization=normalize_document(self.document,analysis)
        elif row.get('environment_version')==ENVIRONMENT_VERSION:
            self.document=prepared_document(row);self.normalization=deepcopy(row['normalization'])
        else:
            self.normalization=normalize_document(self.document,analysis)
        self.lineage=deepcopy(self.normalization['lineage'])
        self.executor=BoundaryExecutor(analysis,{u.sid:u.sid for u in self.document.units},
            {p.pid:p.pid for p in self.document.paragraphs})

    def public_messages(self,role,steps_left=None):
        from .policy_prompts_v42 import prompt_for,assert_messages
        messages=super().public_messages(role,steps_left)
        messages[0]={'role':'system','content':prompt_for(role)}
        payload=json.loads(messages[1]['content'])
        payload['sentence_lineage']={sid:data for sid,data in self.lineage.items()
            if sid!=data['parent_sid'] and any(u.sid==sid for u in self.document.units)}
        messages[1]['content']=dumps(payload);assert_messages(messages,role)
        return messages

    def step(self,raw,role):
        self.executor.editor_role=role
        before={u.sid for u in self.document.units}
        result=super().step(raw,role)
        if result['valid'] and result['action']=='EDIT':
            added=[u.sid for u in self.document.units if u.sid not in before]
            if added:
                parent=result['value']['sentence'];self.scope.update(added)
                result['created_sids']=added
                notice={'kind':'sentence_split','source_sid':parent,'result_sids':[parent]+added,
                    'environment_version':ENVIRONMENT_VERSION}
                result['split_notice']=notice;result['marker_change_notices'].append(notice);self.notices.append(deepcopy(notice))
                for sid in added: self.lineage[sid]={'parent_sid':parent,'initial':False,'action_index':len(self.actions[role])}
                self.last_result={k:v for k,v in result.items() if k not in {'value','before_hash','after_hash'}}
        return result
