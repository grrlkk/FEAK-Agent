"""Append-only conversations, one parse retry, deterministic stage hand-off."""
from collections import Counter
from pathlib import Path
import copy
import json
import time

from ..env.protocol import parse_action, validate_action, ActionParseError, ActionError
from .backends import append_jsonl

PROMPTS = Path(__file__).with_name('prompts')


def system_prompt(role, variant='default', *, allow_check=False):
    prompt = (PROMPTS / {'global': 'global_ko.txt', 'korean': 'korean_ko.txt',
                       'single': 'system_ko.txt'}[role]).read_text(encoding='utf-8')
    if allow_check or variant == 'check_once':
        prompt += '\n비교 실험 설정: 위 행동 목록에 CHECK를 추가한다. CHECK: {}. 필요할 때 채점 도구를 호출하며 남은 checks_left 예산을 따른다.\n'
    if variant == 'check_once':
        prompt += '\n이번 실행에서는 자신의 단계에서 STOP 전에 CHECK를 정확히 한 번 사용한다. 점수와 잡음 범위를 읽고 남은 수정 필요성을 판단한다. CHECK와 STOP에 쓸 step을 남겨 둔다.\n'
    elif variant != 'default':
        raise ValueError('Unknown prompt variant')
    return prompt


def fit_history(messages, backend, tokenizer, actions):
    def size(values):
        return len(tokenizer.apply_chat_template(values, tokenize=True, add_generation_prompt=True))
    if tokenizer is None or size(messages)+2048 <= backend.context_limit:
        return copy.deepcopy(messages), False
    if backend.name == 'teacher':
        raise ValueError('Teacher context limit reached; earlier turns are never rewritten')
    diary = [{'action': a['action'], 'args': a['args'], 'valid': a['valid']} for a in actions]
    # Eight latest conversation turns, original system and first observation.
    shortened = messages[:2] + [{'role': 'user', 'content': '[작업 일지]\n' +
                 json.dumps(diary, ensure_ascii=False, sort_keys=True)}] + messages[-8:]
    if size(shortened)+2048 > backend.context_limit:
        raise ValueError('Context still exceeds model limit after specified history compaction')
    return copy.deepcopy(shortened), True


def validity(raw, *, allow_check=True):
    try:
        value = parse_action(raw)
    except ActionParseError:
        return False, False
    try:
        validate_action(value, allow_check=allow_check)
    except (ActionError, TypeError):
        return True, False
    return True, True


def run_episode(env, episode, backend, *, event_path=None, prompt_variant='default'):
    if prompt_variant == 'check_once' and not env.check_enabled:
        raise ValueError('check_once requires env.enable_check=true')
    started = time.monotonic()
    episode_id = f"{backend.name}:{env.mode}:{episode['episode_id']}"
    histories, observations, calls, events = {}, {}, [], []
    error = None
    def event(value):
        record = {'episode_id': episode_id, **value}
        events.append(record)
        if event_path:
            append_jsonl(event_path, record)
    try:
        observation = env.reset(episode)
        histories[env.role] = [{'role': 'system', 'content': system_prompt(env.role, prompt_variant, allow_check=env.check_enabled)},
                               {'role': 'user', 'content': observation}]
        observations[env.role] = [observation]
        event({'event': 'reset', 'role': env.role, 'observation': observation})
        while not env.done:
            role = env.role
            history = histories[role]
            retried = False
            for parse_attempt in range(2):
                sent, compacted = fit_history(history, backend, env.tokenizer, env.actions[role])
                # Transport retries do not invent model turns or consume action steps.
                for transport_attempt in range(4):
                    try:
                        response = backend.generate(sent, episode_id=episode_id, role=role,
                            turn=f'{env.steps[role]+1}:{parse_attempt}')
                        break
                    except RuntimeError:
                        latest = backend.records[-1] if getattr(backend, 'records', None) else {}
                        retryable = latest.get('http_status') in (429, 500, 502, 503, 504, 520) or 'Timeout' in latest.get('error_type', '')
                        if not retryable or transport_attempt == 3:
                            raise
                        delay = (10, 40, 120)[transport_attempt]
                        event({'event': 'transport_retry', 'role': role, 'delay_s': delay})
                        time.sleep(delay)
                raw = response['raw']
                json_valid, protocol_valid = validity(raw, allow_check=env.check_enabled)
                call = {**response, 'json_valid': json_valid, 'valid_json_action': protocol_valid,
                        'history_compacted': compacted, 'parse_attempt': parse_attempt}
                calls.append(call)
                history.append({'role': 'assistant', 'content': raw})
                event({'event': 'model_turn', 'role': role, 'raw': raw,
                       'json_valid': json_valid, 'valid_json_action': protocol_valid,
                       'phase_call': response.get('phase_call'), 'replayed': response.get('replayed')})
                if json_valid or parse_attempt:
                    break
                retried = True
                try:
                    parse_action(raw)
                except ActionParseError as exc:
                    retry_observation = '[JSON 형식 오류] ' + str(exc) + '\n동일한 행동을 JSON 객체 하나로 다시 출력하세요. 재시도는 한 번입니다.'
                history.append({'role': 'user', 'content': retry_observation})
                observations[role].append(retry_observation)
            observation, done, info = env.step(raw)
            info['action']['parse_retry_used'] = retried
            event({'event': 'action', **info, 'observation': observation})
            if info['handoff']:
                history.append({'role': 'user', 'content': info['stage_terminal_observation']})
                observations[role].append(info['stage_terminal_observation'])
                histories[env.role] = [{'role': 'system', 'content': system_prompt(env.role, prompt_variant, allow_check=env.check_enabled)},
                                       {'role': 'user', 'content': observation}]
                observations[env.role] = [observation]
                event({'event': 'handoff', 'handoff': env.handoff})
            else:
                history.append({'role': 'user', 'content': observation})
                observations[role].append(observation)
        reward = env.reward()
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        reward = None
        event({'event': 'runtime_error', 'error': error})
    usage = Counter()
    for call in calls:
        values = call.get('cost', {})
        if backend.name == 'teacher':
            usage.update({key: values.get(key, 0) for key in ('input', 'output', 'reasoning', 'cache_read', 'cache_write', 'confirmed_usd')})
        else:
            raw_usage = call.get('usage', {})
            usage.update({'input': raw_usage.get('prompt_tokens', 0), 'output': raw_usage.get('completion_tokens', 0),
                          'cache_read': (raw_usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)})
    source_ids = {u.sid for u in env.source.units} if getattr(env, 'source', None) else set()
    result = {'episode_id': episode_id, 'corpus_episode_id': episode['episode_id'],
        'source_id': episode.get('source_id'), 'genre': episode.get('genre'), 'level': episode.get('level'),
        'mode': env.mode, 'backend': backend.name, 'model': backend.model,
        'prompt_variant': prompt_variant,
        'completed': error is None and getattr(env, 'done', False), 'runtime_error': error,
        'messages_by_role': histories, 'observations_by_role': observations, 'calls': calls,
        'actions_by_role': getattr(env, 'actions', {}), 'handoff': getattr(env, 'handoff', None),
        'final_text': env.final_text(), 'reward': reward,
        'initial_layout': env.corrupted.snapshot(), 'final_layout': env.document.snapshot(),
        'stage1_layout': env.stage1.snapshot() if getattr(env, 'stage1', None) else None,
        'termination': getattr(env, 'termination', {}), 'steps': getattr(env, 'steps', {}),
        'checks': getattr(env, 'checks', {}), 'score_calls': getattr(env, 'score_log', []),
        'source_id_mapping': {public: sid if sid in source_ids else None for sid, public in env.sentence_ids.items()
                              if any(u.sid == sid for u in env.document.units)},
        'usage': dict(usage), 'cost_usd': usage.get('confirmed_usd', 0),
        'elapsed_s': time.monotonic()-started}
    event({'event': 'episode_end', 'completed': result['completed'], 'error': error,
           'termination': result['termination'], 'reward': reward})
    return result
