"""Fix1 old/new workers: repeated deterministic smoke and separate rollout preview."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from .common import atomic_new, load_config, read_json, safe_id, write_json
from .dryrun_common import tokenizer, local_analysis_environment
from .dryrun_student import document_state, prompt_token_audit
from .environment_v43 import BoundaryParagraphs
from .environment_v46 import V46Environment, CONTEXT, OUTPUT, REJECT_LIMIT
from .scale2_projection import save_document
from .full_data import DRY, ref, verified
from .common import REPO
from . import maps_v45

from .fix1_taxonomy import ROOT
CONDITIONS = {
    'old': {'orchestrator': 'fix1-orchestrator', 'revision': 'fix1-old-revision', 'korean': 'fix1-old-korean'},
    'new': {'orchestrator': 'fix1-orchestrator', 'revision': 'fix1-new-revision', 'korean': 'fix1-new-korean'},
}
SETTINGS = [{'name': 'deterministic', 'run': i+1, 'temperature': 0., 'top_p': 1., 'seed': seed}
            for i, seed in enumerate((47, 73, 101))]
SETTINGS += [{'name': 'sampling_preview', 'run': 1, 'temperature': .7, 'top_p': .95, 'seed': 173}]
from . import orchestrator_v44 as orch

PORT = 8046


def blocked_reason(report):
    """A nonblank summary or issue is an explicit reason, without a quality judge."""
    return bool(isinstance(report.get('summary'), str) and report['summary'].strip()
                or any(isinstance(x, str) and x.strip() for x in report.get('issues', [])))


def completion(attempt):
    errors = []
    if attempt.get('status') != 'completed': errors.append('execution_exception')
    if attempt.get('orchestrator_ending') != 'FINISH':
        errors.append('orchestrator:'+str(attempt.get('orchestrator_ending')))
    if attempt.get('controller_events'): errors.append('controller_forced_stop')
    if attempt.get('context_limits'): errors.append('context_limit')
    if attempt.get('execution_failures'): errors.append('execution_failure')
    if any(s['status'] != 'completed' for s in attempt.get('score_tools', [])):
        errors.append('scorer_tool_error')
    if not any(s['role'] == 'korean' for s in attempt.get('stages', [])):
        errors.append('missing_korean_pass')
    for stage in attempt.get('stages', []):
        if stage.get('ending') != 'STOP':
            errors.append(stage['role']+':'+str(stage.get('ending')))
        elif stage.get('stop_status') == 'blocked':
            if not blocked_reason(stage.get('report', {})):
                errors.append(stage['role']+':blocked_without_reason')
        elif stage.get('stop_status') != 'done':
            errors.append(stage['role']+':invalid_STOP_status')
    return {'no_protocol_error': not errors,
            'all_stages_STOP_done': not errors and all(s.get('stop_status') == 'done' for s in attempt['stages']),
            'protocol_error_reasons': sorted(set(errors))}


class LocalBackend:
    def __init__(self, condition, setting):
        self.adapters = CONDITIONS[condition]
        self.setting = setting
        import httpx
        self.client = httpx.Client(timeout=600)

    def generate(self, messages, role):
        started = time.monotonic()
        response = self.client.post(f'http://127.0.0.1:{PORT}/v1/chat/completions', json={
            'model': self.adapters[role], 'messages': messages, 'temperature': self.setting['temperature'], 'top_p': self.setting['top_p'],
            'seed': self.setting['seed'], 'max_tokens': OUTPUT[role]})
        response.raise_for_status(); data = response.json()
        return {'raw': data['choices'][0]['message']['content'], 'usage': data.get('usage', {}),
                'finish_reason': data['choices'][0]['finish_reason'], 'elapsed_s': time.monotonic()-started}

    def close(self):
        self.client.close()


def paragraphs(env):
    return [{'id': p.pid, 'sentences': [{'id': u.sid, 'text': u.text} for u in p.units]} for p in env.document.paragraphs]


def sync_map(env):
    if env.shared_map is None:
        return
    ids = {u.sid for u in env.document.units}
    members = {u.sid: p.pid for p in env.document.paragraphs for u in p.units}
    env.shared_map['sentence_relations'] = [e for e in env.shared_map['sentence_relations']
        if e['source'] in ids and e['target'] in ids | {'Q'}]
    env.shared_map['off_topic'] = [sid for sid in env.shared_map['off_topic'] if sid in ids]
    for role in env.shared_map['paragraph_roles']:
        if members.get(role['key_sentence']) != role['paragraph']:
            role['key_sentence'] = None


def run_case(row, *, case_id, output, backend, tok, analysis, score,
             cooperation=False, capture_states=False):
    path = output/'attempts'/(safe_id(case_id)+'.json')
    if path.exists():
        return read_json(path)
    if cooperation:
        from verak.v5.cooperate_env import (CooperationEnvironment, snapshot,
                                            worker_report, orchestrator_observation, VERSION)
        env = CooperationEnvironment(row, analysis)
    else:
        env = V46Environment(row, analysis)
    result = {'case_id': case_id, 'source_id': row['source_id'], 'genre': row['genre'],
        'status': 'running', 'calls': [], 'stages': [], 'orchestrator_actions': [],
        'context_limits': [], 'execution_failures': [], 'controller_events': [], 'score_tools': [],
        'initial': document_state(env), 'quality_results': False, 'new_paid_calls': 0,
        'privileged_plan_at_inference': False, 'records_at_inference': False,
        'environment_version': 'v4.6', 'worker_prompts': 'v4.3', 'orchestrator_prompt': 'v4.4'}
    if cooperation:
        result['environment_version'] = VERSION
    if capture_states:
        result['initial_token_document'] = save_document(env.document)
    directory = output/'requests'/safe_id(case_id)
    sequence = 0
    def request(role, messages):
        nonlocal sequence
        sequence += 1
        prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prefix_path = directory/f'{sequence:04d}_{role}.prefix.json'
        atomic_new(prefix_path, {'messages': messages, 'prefix_input_ids': prefix, 'prefix_tokens': len(prefix),
            'role': role, 'context_limit': CONTEXT[role], 'output_limit': OUTPUT[role],
            'source_characters': len(row['text']), 'current_characters': len(env.document.text),
            'current_text_tokens': len(tok.encode(env.document.text, add_special_tokens=False))})
        if len(prefix)+OUTPUT[role] > CONTEXT[role]:
            event = {'role': role, 'reason': 'prefix_context_limit', 'prefix_tokens': len(prefix),
                     'context_limit': CONTEXT[role], 'dispatched': False, 'prefix': ref(prefix_path)}
            result['context_limits'].append(event)
            return None, event
        try:
            value = backend.generate(messages, role)
            total = len(tok.apply_chat_template(messages+[{'role': 'assistant', 'content': value['raw']}], tokenize=True))
            audit = prompt_token_audit(len(prefix), value.get('usage'))
            value.update(role=role, prefix=ref(prefix_path), prompt_tokens=len(prefix), full_tokens=total, template_audit=audit)
            atomic_new(directory/f'{sequence:04d}_{role}.raw.json', value)
            if audit['status'] != 'match':
                event = {'role': role, 'reason': 'chat_template_prompt_tokens_'+audit['status'],
                         'dispatched': True, 'prefix': ref(prefix_path)}
                result['execution_failures'].append(event)
                result['calls'].append({**value, 'stage': 'before_execution', 'action': {
                    'valid': False, 'action': 'NOT_EXECUTED_TEMPLATE', 'error': event['reason']}})
                return None, event
            if total > CONTEXT[role]:
                event = {'role': role, 'reason': 'target_context_limit', 'tokens': total, 'dispatched': True}
                result['context_limits'].append(event)
                result['calls'].append({**value, 'stage': 'before_execution', 'action': {
                    'valid': False, 'action': 'NOT_EXECUTED_CONTEXT', 'error': event['reason']}})
                return None, event
            return value, None
        except Exception as exc:
            atomic_new(directory/f'{sequence:04d}_{role}.failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            raise

    def stage(role, items, *, automatic=False):
        number = 1+sum(s['role'] == role for s in result['stages'])
        if role == 'revision':
            env.delegate(items, number)
        else:
            env.start_korean(items)
        entry = {'role': role, 'delegation': number, 'automatic': automatic, 'steps': 0,
                 'ending': None, 'stop_status': None, 'failure': False}
        if capture_states:
            entry['initial_token_document'] = save_document(env.document)
        if cooperation:
            entry.update(assigned_tasks=deepcopy(items), initial_snapshot=snapshot(env))
        start_actions = len(env.actions[role])
        result['stages'].append(entry)
        while env.remaining and not env.closed:
            try:
                response, error = request(role, env.public_messages(role))
            except Exception as exc:
                entry.update(ending='request_error', failure=True, error=type(exc).__name__+': '+str(exc))
                break
            if error:
                entry.update(ending=error['reason'], failure=True)
                break
            action = env.step(response['raw'], role)
            sync_map(env)
            result['calls'].append({**response, 'action': deepcopy(action), 'stage': len(result['stages'])})
            entry['steps'] = env.steps_used
            if env.controller_stop:
                entry.update(ending='controller_STOP_blocked', stop_status='blocked', failure=True,
                             report=deepcopy(env.controller_stop))
                result['controller_events'].append({'role': role, 'delegation': number, **env.controller_stop})
                break
            if action['valid'] and action['action'] == 'STOP':
                entry.update(ending='STOP', stop_status=action['value']['status'],
                    failure=action['value']['status'] == 'blocked' and not blocked_reason(action['value']),
                    report=deepcopy(action['value']))
                break
        if entry['ending'] is None:
            entry.update(ending='step_limit', failure=True)
        if capture_states:
            entry['final_token_document'] = save_document(env.document)
        if cooperation:
            entry.update(final_snapshot=snapshot(env), changed_sids=sorted({sid
                for a in env.actions[role][start_actions:] if a['valid']
                for sid in a.get('changed_sids', [])}))
            env.worker_reports.append(worker_report(entry))
        write_json(output/'partial'/(safe_id(case_id)+'.json'), result)
        return entry

    scores = None; history = []; reports = []
    phase = 'score'; counts = Counter(); rejected = 0; korean_current = False
    final_review_pending = False
    try:
        for turn in range(1, 21):
            messages = orch.messages(row['question'], paragraphs(env), phase=phase,
                scores=scores, essay_map=env.shared_map, history=history, reports=reports,
                last_result={k: v for k, v in result['orchestrator_actions'][-1].items() if k != 'value'}
                    if result['orchestrator_actions'] else None)
            if cooperation:
                messages = orchestrator_observation(messages, env, reports,
                                                     final_review_pending=final_review_pending)
            response, error = request('orchestrator', messages)
            if error:
                result['orchestrator_ending'] = error['reason']; break
            record = {'valid': False, 'action': 'INVALID', 'value': None}
            try:
                action, diagnostics = maps_v45.validate(response['raw'], paragraphs(env))
                name = action['action']; record.update(action=name, value=action)
                if phase == 'score' and name != 'SCORE' or phase == 'map' and name != 'MAP':
                    raise ValueError('Initial SCORE then MAP are required')
                if name in {'SCORE', 'MAP'} and counts[name] >= 2 or name == 'DELEGATE' and counts[name] >= 6:
                    raise ValueError('Orchestrator tool/delegation cap reached')
                counts[name] += 1
                if name == 'SCORE':
                    try:
                        scored = score(row['question'], env.document.text)
                        scores = {'rubrics': ['task', 'clarity', 'specificity', 'appropriateness', 'connection', 'unity', 'vocabulary', 'grammar'],
                                  'expected': scored['expected'], 'integers': scored['integers']}
                        result['score_tools'].append({'status': 'completed', 'result': scored})
                    except Exception as exc:
                        scores = {'status': 'unavailable', 'reason': type(exc).__name__}
                        result['score_tools'].append({'status': 'error', 'error': str(exc)})
                    if phase == 'score': phase = 'map'
                elif name == 'MAP':
                    env.shared_map, protection = maps_v45.protected_map(action['map'], paragraphs(env))
                    record.update(diagnostics, relevance_protection=protection.get('off_topic'))
                    phase = 'plan'
                elif name == 'DELEGATE':
                    # Only student-produced public items enter worker delegation.
                    report = stage(action['to'], action['items'])
                    korean_current = action['to'] == 'korean'
                    reports.append(worker_report(report) if cooperation else
                        {k: report.get(k) for k in ('role', 'delegation', 'ending', 'stop_status', 'steps')})
                    final_review_pending = False
                else:
                    ran_final_pass = False
                    if not korean_current:
                        report = stage('korean', [], automatic=True)
                        reports.append(worker_report(report) if cooperation else
                            {k: report.get(k) for k in ('role', 'delegation', 'ending', 'stop_status', 'steps')})
                        korean_current = True
                        ran_final_pass = True
                    if cooperation and ran_final_pass:
                        final_review_pending = True
                        record['finish_deferred_for_worker_report'] = True
                    else:
                        result['finish_report'] = action
                        result['orchestrator_ending'] = 'FINISH'
                        if cooperation:
                            result['finish_snapshot'] = snapshot(env)
                record['valid'] = True
                # MAP is already present in the current snapshot; history keeps
                # the fact of the call without repeating its complete payload.
                history.append({'action': 'MAP'} if name == 'MAP' else action)
                rejected = 0
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                record['error'] = type(exc).__name__+': '+str(exc); rejected += 1
                if isinstance(exc, maps_v45.EmptyMapError):
                    record.update(action='MAP', **exc.diagnostics)
            result['orchestrator_actions'].append(deepcopy(record))
            result['calls'].append({**response, 'action': record, 'stage': 'orchestrator'})
            write_json(output/'partial'/(safe_id(case_id)+'.json'), result)
            if rejected >= REJECT_LIMIT:
                event = {'role': 'orchestrator', 'action': 'STOP', 'status': 'blocked', 'controller_generated': True,
                         'reason': 'three_consecutive_rejections', 'failure': True}
                result['controller_events'].append(event)
                result['orchestrator_ending'] = 'controller_STOP_blocked'; break
            if result.get('orchestrator_ending') == 'FINISH': break
        else:
            result['orchestrator_ending'] = 'step_limit'
        # Mandatory form pass also survives an Orchestrator protocol failure.
        if not korean_current:
            stage('korean', [], automatic=True)
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    result.update(final=document_state(env), final_token_document=save_document(env.document), actions=env.actions, source_citations=env.search.export(env.document.units),
        delegation_counts={'model': counts['DELEGATE'], 'automatic': sum(s['automatic'] for s in result['stages'])},
        complete=result['status'] == 'completed' and result.get('orchestrator_ending') == 'FINISH'
                 and all(not s['failure'] for s in result['stages']) and not result['controller_events']
                 and all(s['status'] == 'completed' for s in result['score_tools']))
    result['completion'] = completion(result)
    if capture_states:
        result['lineage'] = deepcopy(env.lineage)
    result['complete'] = result['completion']['no_protocol_error']
    atomic_new(path, result)
    return result


def summarize(attempts, planned):
    roles = {}
    for role in ('orchestrator', 'revision', 'korean'):
        calls = [c for a in attempts for c in a['calls'] if c['role'] == role]
        stages = [s for a in attempts for s in a['stages'] if s['role'] == role]
        roles[role] = {'returned_actions': len(calls), 'valid_actions': sum(c['action']['valid'] for c in calls),
            'valid_action_rate': sum(c['action']['valid'] for c in calls)/len(calls) if calls else None,
            'invalid_errors': dict(Counter(c['action'].get('error') for c in calls if not c['action']['valid'])),
            'model_STOP_done': sum(s['ending'] == 'STOP' and s['stop_status'] == 'done' for s in stages),
            'model_STOP_blocked': sum(s['ending'] == 'STOP' and s['stop_status'] == 'blocked' for s in stages),
            'model_STOP_rate': sum(s['ending'] == 'STOP' for s in stages)/len(stages) if stages else None,
            'model_STOP_done_rate': sum(s['ending'] == 'STOP' and s['stop_status'] == 'done' for s in stages)/len(stages) if stages else None,
            'controller_STOP_blocked': sum(e['role'] == role for a in attempts for e in a['controller_events']),
            'stages': len(stages), 'stage_endings': dict(Counter(s['ending'] for s in stages)),
            'context_limits': [e for a in attempts for e in a['context_limits'] if e['role'] == role],
            'template_audits': dict(Counter(c['template_audit']['status'] for c in calls)),
            'failure_stages': sum(s['failure'] for s in stages)}
    completions = [completion(a) for a in attempts]
    map_records = [r for a in attempts for r in a['orchestrator_actions'] if r.get('action') == 'MAP']
    span_records = [dict(case_id=a['case_id'], role=c['role'], valid=c['action']['valid'],
                         **c['action']['span_match']) for a in attempts for c in a['calls']
                    if 'span_match' in c['action'] and c['action']['span_match']['mode'] == 'whitespace_normalized']
    return {'planned': planned, 'saved': len(attempts),
        'no_protocol_error': sum(c['no_protocol_error'] for c in completions),
        'all_stages_STOP_done': sum(c['all_stages_STOP_done'] for c in completions),
        'completion_by_case': [{'case_id': a['case_id'], **c} for a, c in zip(attempts, completions)],
        'protocol_error_reasons': dict(Counter(r for c in completions for r in c['protocol_error_reasons'])),
        'map_validation': {'MAP_calls': len(map_records), 'accepted': sum(r['valid'] for r in map_records),
            'raw_edges': sum(r.get('raw_edges', 0) for r in map_records),
            'kept_edges': sum(r.get('kept_edges', 0) for r in map_records),
            'dropped_edges_by_reason': dict(Counter(e['reason'] for r in map_records for e in r.get('dropped_edges', []))),
            'dropped_fields_by_reason': dict(Counter(e['reason'] for r in map_records for e in r.get('dropped_fields', []))),
            'defaulted_fields': sum(len(r.get('defaulted_fields', [])) for r in map_records),
            'salvaged_maps': sum(r['valid'] and bool(r.get('dropped_edges') or r.get('dropped_fields') or r.get('defaulted_fields')) for r in map_records)},
        'whitespace_fallback': {'attempted': len(span_records),
            'unique_match': sum(r['matched'] for r in span_records),
            'accepted_EDIT': sum(r['valid'] for r in span_records), 'records': span_records},
        'quality_results': False, 'roles': roles,
        'orchestrator_endings': dict(Counter(a.get('orchestrator_ending', 'error') for a in attempts)),
        'orchestrator_FINISH_rate': sum(a.get('orchestrator_ending') == 'FINISH' for a in attempts)/len(attempts) if attempts else None,
        'delegations_per_essay': [{'case_id': a['case_id'], **a['delegation_counts']} for a in attempts],
        'mean_model_delegations': sum(a['delegation_counts']['model'] for a in attempts)/len(attempts) if attempts else None,
        'blocked_reports': [dict(case_id=a['case_id'], **s) for a in attempts for s in a['stages'] if s['stop_status'] == 'blocked'],
        'controller_events': [dict(case_id=a['case_id'], **e) for a in attempts for e in a['controller_events']],
        'failed_stages': [dict(case_id=a['case_id'], **s) for a in attempts for s in a['stages'] if s['failure']],
        'execution_failures': [dict(case_id=a['case_id'], **e) for a in attempts for e in a.get('execution_failures', [])],
        'exceptions': [{'case_id': a['case_id'], 'error': a['error']} for a in attempts if 'error' in a],
        'scorer_tool_errors': [dict(case_id=a['case_id'], **s) for a in attempts for s in a['score_tools'] if s['status'] == 'error']}
