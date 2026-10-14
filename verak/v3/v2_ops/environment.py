"""Version-isolated two-stage environment; no changes to the running v1 implementation."""
import time

from ..common import sha_text
from ..env.revision import RevisionEnv
from ..env.feedback import cohesion_facts
from ..env.protocol import ActionError, ActionParseError, parse_action
from .protocol import validate_action
from .actions import V2ActionExecutor
from .config import require_v2


class V2RevisionEnv(RevisionEnv):
    def __init__(self, config, *, defer_rewards=True, **kwargs):
        require_v2(config)
        super().__init__(config, **kwargs)
        self.defer_rewards = defer_rewards

    def reset(self, episode):
        observation = super().reset(episode)
        self.executor = V2ActionExecutor(self.analysis, self.sentence_ids, self.paragraph_ids)
        return observation

    def observe(self, **kwargs):
        return super().observe(**kwargs).replace('[역할] GLOBAL', '[역할] 글 수정 에이전트 (GLOBAL)', 1)

    def reward(self):
        if self.defer_rewards or self.records is None:
            return None
        raise RuntimeError('v2 rewards are computed after the GPU1 queue gate, from saved trajectories')

    def step(self, raw):
        if self.done:
            raise RuntimeError('Episode already terminated')
        started = time.monotonic()
        role = self.role
        self.steps[role] += 1
        before = self.document.clone()
        old_structure = self.public_structure(before)
        facts, check_result, error, error_code, changed = [], None, None, None, set()
        value = {'thought': '', 'action': 'INVALID', 'args': {}}
        old_undo = list(self.undo_stack)
        try:
            value = parse_action(raw)
            validate_action(value, allow_check=self.check_enabled)
            action, args = value['action'], value['args']
            if action in ('EDIT', 'MOVE', 'INSERT', 'SPLIT'):
                candidate = self.document.clone()
                changed, paragraphs = getattr(self.executor, action.lower())(candidate, args, role)
                self.analysis.refresh(candidate, paragraphs)
                self.document = candidate
                self.undo_stack.append(before)
            elif action == 'UNDO':
                if not self.undo_stack:
                    raise ActionError('undo_empty', '이 단계에서 취소할 EDIT/MOVE/INSERT/SPLIT이 없습니다.')
                candidate = self.undo_stack.pop()
                changed = {u.sid for u in before.units + candidate.units}
                self.document = candidate
            elif action == 'CHECK':
                if not self.check_enabled:
                    raise ActionError('check_disabled', '현재 행동 공간은 EDIT/MOVE/UNDO/STOP이며 역할별 제한을 따릅니다.')
                if self.checks[role] >= self.config['env'][role]['max_checks']:
                    raise ActionError('check_budget', 'CHECK 예산이 없습니다.')
                self.checks[role] += 1
                check_result = self.check()
            if changed:
                public_changed = {self.sentence_ids[sid] for sid in changed}
                facts = cohesion_facts(old_structure, self.public_structure(), public_changed)
            self.errors = 0
        except (ActionError, ActionParseError, KeyError, ValueError, RuntimeError) as exc:
            self.document = before
            self.undo_stack = old_undo
            error_code = getattr(exc, 'code', 'parse_error' if isinstance(exc, ActionParseError) else 'tool_or_argument_error')
            error = str(exc)
            self.errors += 1
            changed = set()
        record = {'t': self.steps[role], 'role': role, 'thought': value.get('thought', ''),
            'action': value.get('action', 'INVALID'), 'args': value.get('args', {}), 'raw': raw,
            'valid': error is None, 'error': error, 'error_code': error_code,
            'changed_sids': sorted(changed), 'changed_public_sids': sorted(self.sentence_ids[sid] for sid in changed),
            'cohesion_changes': facts, 'check': check_result, 'elapsed_s': time.monotonic()-started,
            'before_hash': sha_text(before.text), 'after_hash': sha_text(self.document.text)}
        record['created_sids'] = sorted({u.sid for u in self.document.units} - {u.sid for u in before.units})
        record['removed_sids'] = sorted({u.sid for u in before.units} - {u.sid for u in self.document.units})
        self.actions[role].append(record)
        self.notices[role].extend(facts)
        reason = None
        if error is None and value['action'] == 'STOP':
            reason = 'STOP'
        elif self.errors >= self.config['env'].get('max_consecutive_errors', 3):
            reason = 'errors'
        elif self.steps[role] >= self.config['env'][role]['max_steps']:
            reason = 'max_steps'
        elif self.check_enabled and self.checks[role] >= self.config['env'][role]['max_checks']:
            reason = 'max_checks'
        handoff = self._finish(reason) if reason else False
        observation = self.observe(facts=facts, error=error, check=check_result, handoff=handoff, before=before)
        return observation, self.done, {'action': record, 'role': role, 'next_role': self.role,
            'handoff': handoff, 'stage_termination': reason,
            'stage_terminal_observation': '[단계 종료] ' + str(reason)}
