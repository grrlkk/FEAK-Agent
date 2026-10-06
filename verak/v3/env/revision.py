"""Sequential role environment with factual tool observations and hidden rewards."""
from copy import deepcopy
import json
import time

from ..common import sha_text
from ..corrupt.document import Document
from ..ko.structural_rendering import render_structural
from ..reward.total import rewards
from .actions import ActionExecutor
from .analysis import ParagraphAnalyzer, alias_structure
from .feedback import cohesion_facts
from .protocol import ActionError, ActionParseError, parse_action, validate_action

RUBRICS = {
    '설명': ['과제충실성', '설명명료성', '설명구체성', '설명적절성', '문장연결성', '글통일성', '어휘적절성', '어법적절성'],
    '논증': ['과제충실성', '주장명료성', '근거타당성', '주장적절성', '문장·문단연결성', '글통일성', '어휘·문장적절성', '어법정확성'],
    '정서': ['과제충실성', '주제·정서명료성', '주제·정서구체성', '주제·정서적절성', '문장·문단연결성', '글통일성', '어휘·문장적절성', '어법정확성']}


class RevisionEnv:
    def __init__(self, config, *, analysis=None, scorer=None, similarity=None, tokenizer=None, mode=None):
        self.config = config
        self.analysis = analysis or ParagraphAnalyzer(config)
        self.scorer, self.similarity, self.tokenizer = scorer, similarity, tokenizer
        self.mode = mode or config['env']['mode']
        if self.mode not in ('single', 'two_stage'):
            raise ValueError('Unknown environment mode')

    def reset(self, episode):
        self.episode = episode
        self.question, self.genre = episode['question'], episode.get('genre', '기타')
        if 'document' in episode:
            self.document = episode['document'].clone()
        else:
            profile = self.analysis.profile(episode['draft'])
            self.document = Document.from_profile(episode['draft'], profile)
        self.corrupted = self.document.clone()
        self.source = episode.get('source')
        self.records = episode.get('records')
        if self.records is not None and self.source is None:
            raise ValueError('Corruption rewards need a hidden source document')
        self.sentence_ids = {u.sid: f'S{i+1}' for i, u in enumerate(self.document.units)}
        self.paragraph_ids = {p.pid: f'P{i+1}' for i, p in enumerate(self.document.paragraphs)}
        self.executor = ActionExecutor(self.analysis, self.sentence_ids, self.paragraph_ids)
        self.analysis.refresh(self.document, set(self.paragraph_ids))
        self.role = 'global' if self.mode == 'two_stage' else 'single'
        self.roles = ('global', 'korean') if self.mode == 'two_stage' else ('single',)
        self.actions = {role: [] for role in self.roles}
        self.notices = {role: [] for role in self.roles}
        self.steps = {role: 0 for role in self.roles}
        self.checks = {role: 0 for role in self.roles}
        self.termination = {}
        self.starts = {self.role: self.document.clone()}
        self.last_checks = {}
        self.undo_stack, self.score_log, self.scores = [], [], {}
        initial_score = episode.get('corrupted_score')
        if initial_score:
            self.scores[sha_text(self.corrupted.text)] = deepcopy(initial_score)
        self.errors = 0
        self.done = False
        self.handoff = None
        self.stage1 = None
        self._reward = None
        observation = self.observe()
        if self.compact_tokens is not None and self.compact_tokens > self.config['view_token_budget']:
            raise ValueError('Initial compact view exceeds 3000 tokens; exclude, never truncate')
        return observation

    def public_structure(self, document=None):
        return alias_structure((document or self.document).structure(), self.sentence_ids, self.paragraph_ids)

    def budgets(self):
        setting = self.config['env'][self.role]
        result = {'steps_left': setting['max_steps']-self.steps[self.role]}
        if self.check_enabled:
            result['checks_left'] = setting['max_checks']-self.checks[self.role]
        return result

    @property
    def check_enabled(self):
        return self.config['env'].get('enable_check', False)

    def observe(self, *, facts=(), error=None, check=None, handoff=False):
        structure = self.public_structure()
        compact = render_structural(structure, compact=True)
        self.compact_tokens = len(self.tokenizer.encode(compact, add_special_tokens=False)) if self.tokenizer else None
        rubric = RUBRICS.get(self.genre, [f'항목{i}' for i in range(1, 9)])
        lines = [f'[역할] {self.role.upper()}', '[문항] ' + self.question,
                 f'[장르] {self.genre}', '[루브릭] ' + ' / '.join(rubric),
                 '[남은 예산] ' + json.dumps(self.budgets(), ensure_ascii=False)]
        if error:
            lines.append('[실행 오류] ' + error)
        if check:
            lines.append('[점수] ' + json.dumps(check, ensure_ascii=False, sort_keys=True))
        lines.append('[marker-change notices: 표지 변화 알림]\n' + ('\n'.join(f['message'] for f in facts) if facts else '없음'))
        if handoff:
            lines.append('[GLOBAL 인계: 행동 및 누적 marker-change notices]\n' + json.dumps({
                'actions': self.handoff['actions'], 'cohesion_changes': self.handoff['cohesion_changes']},
                ensure_ascii=False, sort_keys=True))
        lines.append('[글]')
        for paragraph in self.document.paragraphs:
            if paragraph.units:
                lines.append('[' + self.paragraph_ids[paragraph.pid] + ']')
                lines.extend(self.sentence_ids[u.sid] + ' | ' + u.text for u in paragraph.units)
        lines.extend(('[Korean document profile: 한국어 문서 프로필; 생략·DEP는 위치 힌트]', compact))
        return '\n'.join(lines)

    def score(self, document, purpose):
        key = sha_text(document.text)
        started = time.monotonic()
        cached = key in self.scores
        if not cached:
            if self.scorer is None:
                raise RuntimeError('Scorer is unavailable')
            value = self.scorer.score(self.question, document.text)
            self.scores[key] = value.to_dict() if hasattr(value, 'to_dict') else value
        result = self.scores[key]
        self.score_log.append({'purpose': purpose, 'role': self.role, 'text_hash': key,
            'episode_cache_hit': cached, 'scorer_cache_hit': result.get('cache_hit', False),
            'elapsed_s': time.monotonic()-started, 'result': deepcopy(result)})
        return result

    def check(self):
        current = self.score(self.document, 'CHECK')
        start = self.score(self.starts[self.role], 'CHECK_stage_start')
        previous = self.last_checks.get(self.role)
        settings = self.config['reward']['quality']
        result = {'expected_scores': current['expected'], 'Q': current['mean'],
            'stage_start_Q': start['mean'], 'delta_stage_start': current['mean']-start['mean'],
            'last_CHECK_Q': previous['mean'] if previous else None,
            'delta_last_CHECK': current['mean']-previous['mean'] if previous else None,
            'noise_floor': settings['noise_floor_by_genre'].get(self.genre, settings['noise_floor']),
            'rubric_names': RUBRICS.get(self.genre, [f'항목{i}' for i in range(1, 9)])}
        self.last_checks[self.role] = current
        return result

    def _finish(self, reason):
        self.termination[self.role] = reason
        if self.role == 'global':
            self.stage1 = self.document.clone()
            self.handoff = {'current_text': self.document.text,
                'actions': [{k: deepcopy(a[k]) for k in ('t', 'action', 'args', 'thought', 'valid', 'error')}
                            for a in self.actions['global']],
                'cohesion_changes': deepcopy(self.notices['global'])}
            self.role = 'korean'
            self.starts['korean'] = self.document.clone()
            self.undo_stack = []
            self.errors = 0
            return True
        self.done = True
        return False

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
            if action in ('EDIT', 'MOVE'):
                candidate = self.document.clone()
                changed, paragraphs = getattr(self.executor, action.lower())(candidate, args, role)
                self.analysis.refresh(candidate, paragraphs)
                self.document = candidate
                self.undo_stack.append(before)
            elif action == 'UNDO':
                if not self.undo_stack:
                    raise ActionError('undo_empty', '이 단계에서 취소할 EDIT/MOVE가 없습니다.')
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
        observation = self.observe(facts=facts, error=error, check=check_result, handoff=handoff)
        return observation, self.done, {'action': record, 'role': role, 'next_role': self.role,
            'handoff': handoff, 'stage_termination': reason,
            'stage_terminal_observation': '[단계 종료] ' + str(reason)}

    def final_text(self):
        return self.document.text

    def reward(self):
        if self.records is None:
            return None
        if not self.done:
            raise RuntimeError('Rewards are available only after episode termination')
        if self._reward is None:
            start = self.score(self.corrupted, 'terminal_reward_start')
            end = self.score(self.document, 'terminal_reward_final')
            middle = self.score(self.stage1, 'terminal_reward_stage1') if self.stage1 else None
            self._reward = rewards(self.source, self.corrupted, self.document, self.records,
                genre=self.genre, q_corrupted=start['mean'], q_final=end['mean'],
                config=self.config['reward'], mode=self.mode, stage1=self.stage1,
                q_stage1=middle['mean'] if middle else None,
                stage1_actions=self.actions.get('global', []), stage2_actions=self.actions.get('korean', []),
                actions=self.actions.get('single', []), similarity=self.similarity,
                tau=self.config['similarity']['tau'],
                preexisting_spell_spans=self.episode.get('preexisting_spell_spans', ()))
        return deepcopy(self._reward)
