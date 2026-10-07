"""The controller dispatches only the Orchestrator's explicit decisions."""
from copy import deepcopy
import json

from ..common import write_json
from ..env.protocol import parse_action
from .environment import context
from .rewards import editor_reward, final_rewards
from .schemas import action_schema

PILOT_STAGE = 'pilot_final_'


def run(env, api, tokenizer, config, event_path):
    event_path.parent.mkdir(parents=True, exist_ok=True)
    result = {'episode_id': env.episode['episode_id'], 'cohort': 'corrupted' if 'records' in env.episode else 'real',
              'completed': False, 'calls': [], 'sequences': [], 'runtime_error': None, 'reward': None}
    histories = {'orchestrator': []}
    with event_path.open('w', encoding='utf-8') as log:
        def emit(event):
            log.write(json.dumps(event, ensure_ascii=False) + '\n')
            log.flush()
        emit({'event': 'reset', 'question': env.question, 'question_id': env.question_id,
              'rows': env.public_rows(), 'initial_graph': env.initial_graph})

        def turn(role, index, limit, history):
            messages, stats = context(env, role, limit - index + 1, tokenizer, history)
            response = api.request(messages, stage=PILOT_STAGE + result['cohort'],
                item_id=f"{result['episode_id']}:{role}:{env.delegations if role != 'orchestrator' else 0}:{index}",
                effort='low', max_output=1024, schema=action_schema(role))
            call = {k: deepcopy(response.get(k)) for k in ('raw', 'phase_call', 'usage', 'cost', 'response_model')}
            call.update(role=role, delegation=env.delegations if role != 'orchestrator' else 0,
                        turn=index, messages=messages, **stats)
            call['total_tokens'] = len(tokenizer.apply_chat_template(
                messages + [{'role': 'assistant', 'content': response['raw']}], tokenize=True, add_generation_prompt=False))
            result['calls'].append(call)
            emit({'event': 'call', **call})
            if call['total_tokens'] > 8192:
                raise ValueError('Teacher input/action exceeds the 8192-token policy context')
            try:
                value = parse_action(response['raw'])
            except ValueError as exc:
                value = {'action': 'INVALID', 'args': {}, 'thought': str(exc)}
            action = env.step(value, role)
            action['turn'] = index
            history.append({'action': value, 'result': action['result']})
            emit({'event': 'action', **action, 'plan': env.plan, 'progress': env.progress,
                  'latest_report': env.latest_report})
            return action

        try:
            for ot in range(1, 13):
                action = turn('orchestrator', ot, 12, histories['orchestrator'])
                if action['valid'] and action['action'] == 'FINISH':
                    result['reward'] = final_rewards(env, result['sequences'], config)
                    result['completed'] = True
                    break
                if action['valid'] and action['action'] == 'DELEGATE':
                    args = action['args']
                    role = args['agent']
                    env.begin_editor(args)
                    before = env.document.clone()
                    scope_ids = {u.sid for p in before.paragraphs if p.pid in env.scope for u in p.units}
                    history, editor_actions = [], []
                    sequence = {'role': role, 'delegation': env.delegations, 'scope': args.get('scope', 'all'),
                                'scope_ids': sorted(scope_ids), 'task': args['task'], 'terminal': None, 'reward': None}
                    result['sequences'].append(sequence)
                    for et in range(1, 17):
                        editor_action = turn(role, et, 16, history)
                        editor_actions.append(editor_action)
                        if editor_action['valid'] and editor_action['action'] == 'REPORT':
                            sequence['terminal'] = 'REPORT'
                            break
                    if sequence['terminal'] is None:
                        env.latest_report = {'agent': role, 'status': 'blocked', 'summary': '편집 행동 예산 소진', 'origin': 'controller'}
                    sequence['rejected'] = sum(not a['valid'] for a in editor_actions)
                    sequence['reward'] = editor_reward(env, role, before, scope_ids, editor_actions, config)
                    env.last_result = {'editor_return': deepcopy(env.latest_report)}
                    histories['orchestrator'].append(deepcopy(env.last_result))
                    emit({'event': 'editor_return', **sequence, 'report': env.latest_report})
            if not result['completed']:
                result['termination'] = 'orchestrator_budget'
        except Exception as exc:
            result['runtime_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            emit({'event': 'runtime_error', **result['runtime_error']})
        result.update(actions=env.actions, final_text=env.document.text, final_layout=env.document.snapshot(),
                      final_rows=env.public_rows(), final_graph=env.graph(), plan=env.plan, progress=env.progress,
                      delegations=env.delegations, score_calls=env.score_calls, latest_report=env.latest_report)
        result['confirmed_episode_cost'] = sum((c.get('cost') or {}).get('confirmed_usd', 0) for c in result['calls'])
        emit({'event': 'end', 'completed': result['completed'], 'runtime_error': result['runtime_error']})
    return result
