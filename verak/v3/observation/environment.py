"""Observation-only interventions; actions, analyzer and reward stay inherited."""
from copy import deepcopy
import json
import re

from ..env import RevisionEnv
from ..agent.runner import system_prompt
from .graph import build, render, changes


def prompt(role, setting):
    original=system_prompt(role)
    if setting=='current':
        return original
    if setting not in {'text_only','graph'}:
        raise ValueError('Unknown observation setting')
    # Preserve action/content rules verbatim; replace only observational instructions.
    lines=[line for line in original.splitlines() if not any(v in line for v in
        ('Korean document profile','marker-change notices','주문체는','관계 후보·접속','관찰의 ?'))]
    if setting=='text_only':
        lines=[line.replace('인계된 행동 기록과 누적 변화','인계된 행동 기록')
               for line in lines if not line.startswith('- GLOBAL의 이동 뒤 바뀐 접속')]
        lines.append('각 단계 시작과 매 5번째 행동 뒤 전체 글을 문장 ID와 함께 읽는다. 그 사이에는 바뀐 문단과 양쪽 이웃 한 문장이 갱신된다. 표시되지 않은 문장은 이전 상태를 유지한다. KOREAN은 GLOBAL의 행동 기록을 인계받는다.')
    else:
        lines.append('각 단계 시작과 매 5번째 행동 뒤 전체 글과 essay graph가 나온다. 그 사이에는 바뀐 문단과 양쪽 이웃 한 문장의 글·graph 및 graph-change notices가 갱신된다. KOREAN은 GLOBAL 행동 및 graph 변화 기록을 인계받는다.')
        lines.append('담화 edge와 node role은 입력 글에서 한 번 추출한 관찰이며 정답이 아니다. MOVE 뒤에도 같은 노드에 남으며 표현을 고쳐도 다시 추론하지 않는다. ⚠는 서로 다른 문단에 있다는 위치 사실, isolated는 추출된 담화 edge 없음이다. 둘 다 수정 필요성의 판정이 아니다. DEP와 주어 생략은 위치 힌트다. clauses는 바른의 어미·내포 경계에 따른 표면 구간이다. 인용/명사/관형절의 완전한 구문 분석은 아니다.')
    return '\n'.join(lines)+'\n'


def replace_section(text, header, replacement=''):
    pattern=r'(?m)^'+re.escape(header)+r'\n.*?(?=\n\[|\Z)'
    return re.sub(pattern, lambda _:replacement, text, flags=re.S).strip()


class ObservationEnv(RevisionEnv):
    def __init__(self, *args, setting='current', discourse=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.setting=setting
        self.discourse=deepcopy(discourse)
        self.graph_history=[]
        self.view_history=[]

    def observe(self, **kwargs):
        text=super().observe(**kwargs)
        if self.setting=='current':
            return text
        text=text.split('\n[Korean document profile:',1)[0]
        text=replace_section(text,'[marker-change notices: 표지 변화 알림]')
        if self.setting=='text_only':
            if kwargs.get('handoff'):
                text=replace_section(text,'[GLOBAL 인계: 행동 및 누적 marker-change notices]',
                    '[GLOBAL 인계: 행동]\n'+json.dumps({'actions':self.handoff['actions']},ensure_ascii=False,sort_keys=True))
            return text
        if self.discourse is None:
            raise ValueError('Graph condition needs an input-only discourse extraction')
        graph=build(self.public_structure(),self.discourse,
                    paragraph_ids=[self.paragraph_ids[p.pid] for p in self.document.paragraphs])
        notices=[]
        before=kwargs.get('before')
        last_action=next((a for role in reversed(self.roles) for a in self.actions[role][-1:]),{})
        if before is not None:
            old=build(self.public_structure(before),self.discourse,
                      paragraph_ids=[self.paragraph_ids[p.pid] for p in before.paragraphs])
            notices=changes(old,graph,last_action)
            self.graph_history.append({'role':last_action.get('role',self.role),
                't':last_action.get('t'), 'messages':notices})
        if kwargs.get('handoff'):
            text=replace_section(text,'[GLOBAL 인계: 행동 및 누적 marker-change notices]',
                '[GLOBAL 인계: 행동 및 graph-change notices]\n'+json.dumps({
                    'actions':self.handoff['actions'],'graph_change_notices':[
                        msg for item in self.graph_history if item['role']=='global' for msg in item['messages']]},
                    ensure_ascii=False,sort_keys=True))
        notice='[graph-change notices]\n'+('\n'.join(notices) or '없음')
        # Keep text snapshot markers exactly as the shared history selector expects.
        idx=text.find('\n[전체 갱신]')
        if idx<0:
            idx=text.find('\n[부분 갱신:')
        text=text[:idx]+'\n'+notice+text[idx:] if idx>=0 else text+'\n'+notice
        selected=re.findall(r'(?m)^((?:S\d+[a-z]*|N\d+)) \| ',text.split('[글]\n',1)[-1])
        if selected:
            text+='\n'+render(graph,self.role,selected)
        self.view_history.append({'role':self.role,'step':self.steps[self.role],
                                  'graph':graph,'view':text,'notices':notices})
        return text
