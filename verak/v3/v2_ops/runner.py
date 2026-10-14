"""v2 trajectory loop using the unchanged v1 context compaction and hand-off contract."""
from collections import Counter
import time
from types import SimpleNamespace

from ..agent.backends import append_jsonl
from ..agent.runner import fit_history
from ..env.protocol import parse_action, ActionParseError
from .protocol import validity
from .prompts import system_prompt
from .config import require_v2


def run_episode(env, episode, backend, *, event_path=None, prompt_variant='default', prompt_factory=None):
    require_v2(env.config)
    if prompt_variant == 'check_once' and not env.check_enabled:
        raise ValueError('check_once requires env.enable_check=true')
    started = time.monotonic()
    episode_id = f"{backend.name}:{env.mode}:{episode['episode_id']}"
    histories, observations, calls, events = {}, {}, [], []
    error = None
    def role_prompt(role):
        return prompt_factory(role) if prompt_factory else system_prompt(role, prompt_variant, allow_check=env.check_enabled)
    def event(value):
        record = {'episode_id': episode_id, **value}
        events.append(record)
        if event_path:
            append_jsonl(event_path, record)
    try:
        observation = env.reset(episode)
        histories[env.role] = [{'role': 'system', 'content': role_prompt(env.role)},
                               {'role': 'user', 'content': observation}]
        observations[env.role] = [observation]
        event({'event': 'reset', 'role': env.role, 'observation': observation})
        while not env.done:
            role = env.role
            history = histories[role]
            retried = False
            for parse_attempt in range(2):
                context = SimpleNamespace(name=backend.name,
                    context_limit=env.config['policy']['context_limit'],
                    generation_reserve=env.config['policy'].get('generation_reserve', 1024))
                sent, compacted = fit_history(history, context, env.tokenizer, env.actions[role])
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
                histories[env.role] = [{'role': 'system', 'content': role_prompt(env.role)},
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
