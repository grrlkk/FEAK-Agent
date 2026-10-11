"""Inference-only v4.4 guards. Frozen v4.3 teachers and exports are untouched."""
from copy import deepcopy

from verak.v3.env.feedback import cohesion_facts
from .common import sha_text
from .content_env import ContentEnv, parse_json
from .environment_v43 import V43Environment

VERSION = 'v4.4_inference_20261011'
CONTEXT = {'revision': 8192, 'korean': 16384, 'orchestrator': 8192}
OUTPUT = {'revision': 1024, 'korean': 1024, 'orchestrator': 4096}
NOOP_NOTICE = '고칠 것이 없으면 STOP 한다.'
REJECT_LIMIT = 3


class WholeSentenceDelete(ContentEnv):
    """An MRO hook for DELETE only; all other actions keep the v4.3 executor.

    The surrounding DelegatedEnv still enforces scope and role, and V43Environment
    still maintains citations/undo. Removing an entire sentence is not editing a
    mask's bytes. This hook never permits an EDIT to alter or remove a mask.
    """
    def step(self, raw, role):
        try:
            value = parse_json(raw)
        except (ValueError, TypeError):
            return super().step(raw, role)
        if not (isinstance(value, dict) and value.get('action') == 'DELETE' and role == 'revision'
                and set(value) == {'action', 'sentence'}):
            return super().step(raw, role)
        before = self.document.clone()
        relations = deepcopy(self.relations)
        record = {'action': 'DELETE', 'value': value, 'valid': False,
                  'before_hash': sha_text(before.text), 'changed_sids': [], 'marker_change_notices': []}
        aliases, serial = deepcopy(self.executor.sentences), self.executor.serial
        try:
            work = before.clone()
            # Resolve one sentence only. Paragraph deletion is not a worker tool.
            sid = self.executor.sid(value['sentence'], work)
            pi, si, unit = work.locate(sid)
            if len(work.units) <= 1:
                raise ValueError('글 전체를 삭제할 수 없습니다.')
            pid = work.paragraphs[pi].pid
            del work.paragraphs[pi].units[si]
            self.analysis.refresh(work, {pid})
            self.document = work
            self.relations = {s: r for s, r in relations.items() if s != sid and r['target'] != sid}
            self.undo.append((before, relations))
            record.update(valid=True, changed_sids=[sid], whole_sentence_mask_deletion=True,
                          marker_change_notices=cohesion_facts(before.structure(), work.structure(), {sid}))
        except Exception as exc:
            self.document, self.relations = before, relations
            self.executor.sentences, self.executor.serial = aliases, serial
            record['error'] = type(exc).__name__ + ': ' + str(exc)
        record.update(after_hash=sha_text(self.document.text), text_changed=before.text != self.document.text)
        self.actions[role].append(record)
        self.notices.extend(record['marker_change_notices'])
        self.last_result = {k: v for k, v in record.items() if k not in {'value', 'before_hash', 'after_hash'}}
        return record


class V44Environment(V43Environment, WholeSentenceDelete):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rejected_in_a_row = 0
        self.controller_stop = None
        self.shared_map = None

    def delegate(self, items, number):
        super().delegate(items, number)
        self.rejected_in_a_row = 0
        self.controller_stop = None

    def start_korean(self, items=()):
        super().start_korean(items)
        self.rejected_in_a_row = 0
        self.controller_stop = None

    def public_messages(self, role, steps_left=None):
        messages = super().public_messages(role, steps_left)
        if self.shared_map is not None:
            from .render_v43 import assert_rendered, render
            payload = assert_rendered(messages)
            ids = {u.sid for u in self.document.units}
            payload['shared_map'] = deepcopy(self.shared_map)
            payload['shared_map']['sentence_relations'] = [r for r in self.shared_map['sentence_relations']
                if r['source'] in ids and r['target'] in ids | {'Q'}]
            payload['shared_map']['off_topic'] = [s for s in self.shared_map['off_topic'] if s in ids]
            messages = render(payload, messages[0]['content'])
        return messages

    def step(self, raw, role):
        record = super().step(raw, role)
        if not record['valid'] and record['action'] == 'EDIT' and '바꾸지 않는 행동' in record.get('error', ''):
            record['error'] += ' ' + NOOP_NOTICE
            self.last_result['error'] = record['error']
        self.rejected_in_a_row = 0 if record['valid'] else self.rejected_in_a_row + 1
        if self.rejected_in_a_row >= REJECT_LIMIT:
            self.closed = True
            self.controller_stop = {'action': 'STOP', 'status': 'blocked',
                'summary': '연속 거절 3회로 단계를 종료했습니다.', 'issues': [record.get('error', 'rejected_action')],
                'controller_generated': True, 'reason': 'three_consecutive_rejections', 'failure': True}
            record['controller_stop'] = deepcopy(self.controller_stop)
            # This is a controller event, never an accepted model action or a
            # training target. The third rejected action retains valid=False.
            self.last_result['controller_stop'] = deepcopy(self.controller_stop)
        return record
