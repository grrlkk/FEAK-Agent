"""Opt-in integration repairs; frozen prompts and v4.6 defaults stay unchanged."""
from copy import deepcopy
import json

from verak.v4.common import sha_text
from verak.v4.content_env import dumps, parse_json
from verak.v4.environment_v46 import V46Environment, REJECT_LIMIT
from verak.v4.render_v43 import assert_rendered, render

VERSION = 'v5_cooperate1_20261013'


def search_unavailable(query):
    raise ValueError('SEARCH is disabled; the requested evidence belongs in writer_notes')


def snapshot(env):
    return sha_text(dumps([[[p.pid, [[u.sid, u.text] for u in p.units]]
                            for p in env.document.paragraphs], env.relations]))


def worker_report(entry):
    """Transmit actual reports and unresolved reasons, without private targets."""
    keys = ('role', 'delegation', 'automatic', 'ending', 'stop_status', 'steps',
            'failure', 'report', 'assigned_tasks', 'changed_sids',
            'initial_snapshot', 'final_snapshot', 'error')
    result = {k: deepcopy(entry[k]) for k in keys if k in entry}
    result['report_is_model_claim'] = True
    result['controller_ended_stage'] = entry.get('ending') == 'controller_STOP_blocked'
    return result


def orchestrator_observation(messages, env, reports, *, final_review_pending):
    messages = deepcopy(messages)
    payload = json.loads(messages[1]['content'])
    payload['integration'] = {
        'version': VERSION, 'current_snapshot': snapshot(env),
        'final_review_pending': final_review_pending,
        'finish_rule': '마지막 한국어 점검의 보고와 현재 글을 확인한 뒤 FINISH한다. 미해결 일은 writer_notes에 남긴다.',
        'delegation_rule': '현재 ID로 작업한다. MOVE의 location에는 이동 대상과 목적지 앵커를 모두 넣는다. 끝난 일은 반복하지 않는다.',
        'marker_rule': '지도와 변화 알림은 참고 관찰이며 오류의 확정 판정이 아니다.',
        'declared_relations': deepcopy(env.relations),
    }
    # orch.messages already receives these reports; do not duplicate them.
    payload['worker_reports'] = deepcopy(reports)
    if not env.search_enabled:
        payload['integration']['search_available'] = False
        payload['integration']['deferred_search_tasks'] = deepcopy(env.deferred_search_tasks)
    messages[1]['content'] = dumps(payload)
    return messages


class CooperationEnvironment(V46Environment):
    """Keep full Korean scope; never blacklist a sentence after a failed action."""
    def __init__(self, *args, search_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_rejections = {}
        self.worker_reports = []
        self.search_enabled = search_enabled
        self.deferred_search_tasks = []
        if not search_enabled:
            self.search.backend = search_unavailable

    def route_revision_items(self, items):
        """Defer evidence tasks before dispatch; never fabricate a worker STOP."""
        if self.search_enabled:
            return deepcopy(items), []
        ready, deferred = [], []
        for item in items:
            if item['needs_search'] == 'no':
                ready.append(deepcopy(item))
                continue
            entry = {'task': deepcopy(item), 'status': 'deferred_to_writer',
                     'reason': 'search_disabled', 'origin': 'controller',
                     'note': '추가 근거 확인 필요 ('+', '.join(item['location'])+'): '+item['instruction']}
            deferred.append(entry)
            if entry not in self.deferred_search_tasks:
                self.deferred_search_tasks.append(deepcopy(entry))
        return ready, deferred

    def writer_notes(self, existing=()):
        return list(dict.fromkeys([*existing, *(x['note'] for x in self.deferred_search_tasks)]))

    def delegate(self, items, number):
        if not self.search_enabled and any(x.get('needs_search') == 'yes' for x in items):
            raise ValueError('Defer SEARCH tasks to writer_notes before worker dispatch')
        super().delegate(items, number)
        self.stage_rejections = {}

    def start_korean(self, items=()):
        super().start_korean(items)
        self.stage_rejections = {}

    def state_key(self):
        # Steps/last error are not document changes. Scope and retrieved evidence
        # are: a rejected action in another delegation is not automatically banned.
        return sha_text(dumps([snapshot(self), self.active_role, sorted(self.scope),
                               self.search.observation(self.document.units)]))

    def public_messages(self, role, steps_left=None):
        messages = super().public_messages(role, steps_left)
        payload = assert_rendered(messages)
        key = self.state_key()
        payload['cooperation'] = {
            'version': VERSION, 'current_snapshot': snapshot(self),
            'previous_worker_reports': deepcopy(self.worker_reports),
            'observations_are_error_verdicts': False,
            'rejected_actions_in_current_state': [deepcopy(v) for (s, _), v
                                                  in self.stage_rejections.items() if s == key],
            'retry_rule': '같은 상태에서 이미 거절된 똑같은 행동을 반복하지 않는다. 다른 유효한 수정은 가능하다. 할 일이 없으면 STOP done, 못하면 이유를 적어 STOP blocked한다.',
        }
        if not self.search_enabled:
            payload['cooperation']['search_available'] = False
        if role == 'korean':
            # This is the original whole-essay form scope, NOT a per-notice queue.
            payload['scope'] = 'whole_essay_form_pass'
            payload['task'] = ('배정 항목과 글 전체의 오타·띄어쓰기를 함께 점검한다. '
                               '변화 알림의 연결은 현재 문맥에서 실제로 잘못된 경우만 고친다. '
                               '앞선 타당한 내용 수정을 보존하고, 구조 수정이 필요하면 issues에 보고한다. '
                               '이 단계의 검토가 끝나면 STOP한다.')
        return render(payload, messages[0]['content'])

    def step(self, raw, role):
        if self.closed or role != self.active_role or not self.remaining:
            raise RuntimeError('Delegation ended; another action cannot be applied')
        try:
            value = parse_json(raw)
        except (TypeError, ValueError):
            value = None
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, dict) else raw
        state = self.state_key()
        prior = self.stage_rejections.get((state, canonical))
        if prior is None:
            record = super().step(raw, role)
        else:
            # No model action is fabricated; the duplicate remains a rejection,
            # consumes its request, and retains the original three-reject limit.
            before = sha_text(self.document.text)
            self.steps_used += 1
            self.rejected_in_a_row += 1
            record = {'action': value.get('action', 'INVALID') if isinstance(value, dict) else 'INVALID',
                      'value': deepcopy(value), 'valid': False, 'text_changed': False,
                      'before_hash': before, 'after_hash': before, 'changed_sids': [],
                      'marker_change_notices': [], 'duplicate_rejection': True,
                      'error': '같은 상태에서 이미 거절된 행동입니다. '+prior['error']}
            if self.rejected_in_a_row >= REJECT_LIMIT:
                self.closed = True
                self.controller_stop = {'action': 'STOP', 'status': 'blocked',
                    'summary': '연속 거절 3회로 단계를 종료했습니다.', 'issues': [record['error']],
                    'controller_generated': True, 'reason': 'three_consecutive_rejections', 'failure': True}
                record['controller_stop'] = deepcopy(self.controller_stop)
            self.actions[role].append(record)
            self.last_result = deepcopy(record)
        if not record['valid']:
            self.stage_rejections.setdefault((state, canonical), {
                'action': deepcopy(record.get('value')), 'error': record.get('error', 'invalid action')})
        return record
