"""Canonical v4 hard constraints; legacy pilots and running v1 stay unchanged."""
import json

from .prep2_content import DelegatedEnv
from .content_env import dumps
from .policy_prompts import prompt_for, assert_messages


class V4Environment(DelegatedEnv):
    def __init__(self, row, analysis):
        super().__init__(row, analysis)
        self.active_role = None
        self.step_limit = 0
        self.steps_used = 0
        self.closed = True

    @property
    def remaining(self):
        return max(0, self.step_limit-self.steps_used)

    def delegate(self, items, number):
        super().delegate(items, number)
        self.active_role, self.step_limit, self.steps_used, self.closed = 'revision', 6, 0, False

    def start_korean(self, items=()):
        super().start_korean(items)
        # The mandatory whole-essay form pass is itself an explicit delegation.
        self.scope = {p.pid for p in self.document.paragraphs} | {u.sid for u in self.document.units}
        self.active_role, self.step_limit, self.steps_used, self.closed = 'korean', 14, 0, False

    def public_messages(self, role, steps_left=None):
        if role != self.active_role or self.closed or not self.remaining:
            raise RuntimeError('No active delegation for this role')
        if steps_left is not None and steps_left != self.remaining:
            raise ValueError('The environment owns the remaining-step count')
        messages = super().public_messages(role, self.remaining)
        messages[0]['content'] = prompt_for(role)
        payload = json.loads(messages[1]['content'])
        payload['allowed_scope'] = sorted(self.scope)
        payload['step_limit'] = self.step_limit
        messages[1]['content'] = dumps(payload)
        assert_messages(messages, role)
        return messages

    def step(self, raw, role):
        if role != self.active_role or self.closed or not self.remaining:
            raise RuntimeError('Delegation ended; another action cannot be applied')
        self.steps_used += 1  # rejected actions also consume one step
        result = super().step(raw, role)
        if result['valid'] and result['action'] == 'STOP':
            self.closed = True
        return result
