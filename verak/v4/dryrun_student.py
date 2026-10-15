"""Strict local-student Dv3 execution; no teacher API, trimming, or quality scorer."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from .dryrun_common import ROOT, ROLES, read_json, write_json, atomic_new, file_sha, reference, checked
from .common import safe_id, load_config, collection_lock
from .environment_v43 import V43Environment, BoundaryParagraphs, ENVIRONMENT_VERSION
from .policy_prompts_v43 import assert_messages, verify_frozen, VERSION
from .render_v43 import assert_rendered


def document_state(env):
    return {'text': env.document.text, 'layout': env.document.snapshot(),
            'lineage': deepcopy(env.lineage)}


def prompt_token_audit(expected, usage):
    observed = usage.get('prompt_tokens') if isinstance(usage, dict) else None
    if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
        return {'status': 'missing', 'HF_prefix_tokens': expected,
                'vLLM_prompt_tokens': observed, 'delta': None}
    return {'status': 'match' if observed == expected else 'mismatch',
            'HF_prefix_tokens': expected, 'vLLM_prompt_tokens': observed, 'delta': observed-expected}


def run_case(row, plan, backend, tok, analysis, *, output_root, case_id=None, environment=V43Environment):
    """A saved error is terminal. Every attempted prefix is durable before dispatch."""
    case_id = case_id or row['source_id']
    path = output_root / 'attempts' / (safe_id(case_id)+'.json')
    if path.exists():
        return read_json(path)
    call_root = output_root / 'requests' / safe_id(case_id)
    result = {'case_id': case_id, 'source_id': row['source_id'], 'genre': row['genre'],
        'prompt_version': VERSION, 'status': 'running', 'calls': [], 'delegations': [],
        'termination': {}, 'states': {}, 'paid_calls': 0, 'quality_scoring': False,
        'teacher_trimming': False, 'strict_single_action': True, 'at': time.time()}
    env = None
    def stage(role, number):
        entry = {'role': role, 'delegation': number, 'steps': 0, 'ending': None, 'stop_status': None}
        if role == 'revision':
            result['delegations'].append(entry)
        else:
            result['korean_stage'] = entry
        while env.remaining and not env.closed:
            turn = env.steps_used+1
            public = env.public_messages(role)
            assert_messages(public, role)
            payload = assert_rendered(public)
            prefix_ids = tok.apply_chat_template(public, tokenize=True, add_generation_prompt=True)
            prefix = {'case_id': case_id, 'role': role, 'delegation': number, 'turn': turn,
                'public_messages': public, 'prefix_input_ids': prefix_ids, 'prefix_tokens': len(prefix_ids),
                'current_text': env.document.text, 'current_text_characters': len(env.document.text),
                'current_text_tokens': len(tok.encode(env.document.text, add_special_tokens=False)),
                'source_characters': len(row['text']),
                'source_tokens': len(tok.encode(row['text'], add_special_tokens=False)),
                'notices_present': 'notices' in payload, 'journal_present': 'work_journal' in payload,
                'handoff_present': 'revision_handoff' in payload, 'remaining_steps': env.remaining,
                'state': document_state(env), 'attempted_at': time.time()}
            prefix_path = call_root / f'{role}_d{number}_t{turn}.prefix.json'
            if prefix_path.exists():
                raise RuntimeError('An unfinished saved student request exists; never retry silently')
            atomic_new(prefix_path, prefix)
            if len(prefix_ids) > 7168:
                entry['ending'] = 'context_limit'
                result.setdefault('context_failures', []).append({**reference(prefix_path),
                    'role': role, 'prefix_tokens': len(prefix_ids), 'dispatched': False})
                break
            try:
                response = backend.generate(public, episode_id=case_id, role=role,
                                            turn=f'd{number}:t{turn}')
            except Exception as exc:
                atomic_new(call_root / f'{role}_d{number}_t{turn}.failure.json', {
                    'error': type(exc).__name__+': '+str(exc), 'traceback': traceback.format_exc(),
                    'prefix': reference(prefix_path), 'retried': False})
                raise
            raw = response['raw']
            total = len(tok.apply_chat_template(public+[{'role': 'assistant', 'content': raw}], tokenize=True))
            template_audit = prompt_token_audit(len(prefix_ids), response.get('usage'))
            # No first_action import: student raw text always reaches strict parsing.
            if template_audit['status'] != 'match':
                record = {'valid': False, 'action': 'NOT_EXECUTED_TEMPLATE', 'value': None,
                          'error': 'template_prompt_tokens_'+template_audit['status'], 'text_changed': False}
                entry['ending'] = record['error']
            elif total > 8192:
                record = {'valid': False, 'action': 'NOT_EXECUTED_CONTEXT', 'value': None,
                          'error': 'target_context_limit', 'text_changed': False}
                entry['ending'] = 'target_context_limit'
            else:
                record = env.step(raw, role)
            call = {'role': role, 'delegation': number, 'turn': turn, 'raw': raw,
                'action': record, 'prompt_tokens': len(prefix_ids), 'full_tokens': total,
                'prefix': reference(prefix_path), 'finish_reason': response.get('finish_reason'),
                'usage': response.get('usage'), 'elapsed_s': response.get('elapsed_s'),
                'chat_template_audit': template_audit}
            atomic_new(call_root / f'{role}_d{number}_t{turn}.response.json', call)
            result['calls'].append(call);entry['steps'] = env.steps_used
            write_json(output_root / 'partial' / (safe_id(case_id)+'.json'), result)
            if total > 8192 or template_audit['status'] != 'match':
                break
            if record['valid'] and record['action'] == 'STOP':
                stop = record['value']
                entry.update(ending='STOP', stop_status=stop['status'],
                             summary=stop['summary'], issues=stop['issues'])
                break
        if entry['ending'] is None:
            entry['ending'] = 'step_limit'
        return entry
    try:
        if row.get('environment_version') != ENVIRONMENT_VERSION:
            raise ValueError('Student must receive the same frozen prepared v4.3 row')
        env = environment(row, analysis)
        result['states']['initial'] = document_state(env)
        revision = [i for i in plan['items'] if i['owner'] == 'revision']
        korean = [i for i in plan['items'] if i['owner'] == 'korean']
        for number, start in enumerate(range(0, len(revision), 2), 1):
            env.delegate(revision[start:start+2], number)
            stage('revision', number)
        result['states']['revision_final'] = document_state(env)
        result['termination']['revision'] = ('no_items' if not revision else 'STOP'
            if all(d['ending'] == 'STOP' for d in result['delegations']) else 'not_all_STOP')
        env.start_korean(korean)
        result['states']['korean_initial'] = document_state(env)
        first_korean = assert_rendered(env.public_messages('korean'))
        result['handoff_audit'] = {
            'document_identical': result['states']['revision_final'] == result['states']['korean_initial'],
            'handoff_present': 'revision_handoff' in first_korean,
            'journal_present': 'work_journal' in first_korean,
            'notices_present': 'notices' in first_korean,
            'whole_essay_scope': set(env.scope) == ({u.sid for u in env.document.units}
                                                 | {p.pid for p in env.document.paragraphs}),
            'revision_actions': len(env.actions['revision']),
            'handoff_actions': len(first_korean.get('revision_handoff', {}).get('action_log', []))}
        if not all(result['handoff_audit'][k] for k in
                   ('document_identical', 'handoff_present', 'journal_present', 'notices_present', 'whole_essay_scope')):
            raise ValueError('Frozen Revision→Korean handoff invariant failed')
        stage('korean', 1)
        result['termination']['korean'] = result['korean_stage']['ending']
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    if env is not None:
        result['states']['final'] = document_state(env)
        result.update(actions=env.actions, final_text=env.document.text,
                      final_paragraphs=[{'id': p.pid, 'sentences': [{'id': u.sid, 'text': u.text} for u in p.units]}
                                        for p in env.document.paragraphs])
        result.update(env.search.export(env.document.units))
    for entry in result['delegations']+([result['korean_stage']] if 'korean_stage' in result else []):
        if entry['ending'] is None:
            entry['ending'] = result['status']
    result['complete'] = (result['status'] == 'completed'
        and result['termination'].get('revision') in {'STOP', 'no_items'}
        and result['termination'].get('korean') == 'STOP')
    atomic_new(path, result)
    return result


def metrics(attempts, *, planned):
    calls = [c for a in attempts for c in a['calls']]
    roles = {}
    for role in ROLES:
        values = [c for c in calls if c['role'] == role]
        valid = [c for c in values if c['action']['valid']]
        invalid = [c for c in values if not c['action']['valid']]
        endings = [d for a in attempts for d in (a['delegations'] if role == 'revision'
                    else [a['korean_stage']] if 'korean_stage' in a else [])]
        stops = [c['action']['value'] for c in valid if c['action']['action'] == 'STOP']
        roles[role] = {'returned_actions': len(values), 'valid_actions': len(valid),
            'invalid_actions': len(invalid), 'valid_action_rate': len(valid)/len(values) if values else None,
            'invalid_action_types': dict(Counter(c['action']['action'] for c in invalid)),
            'invalid_errors': dict(Counter(c['action'].get('error', 'unknown') for c in invalid)),
            'valid_action_types': dict(Counter(c['action']['action'] for c in valid)),
            'endings': dict(Counter(e['ending'] for e in endings)),
            'STOP_status': dict(Counter(s['status'] for s in stops)), 'STOP_reports': stops}
    return {'planned': planned, 'saved': len(attempts), 'complete': sum(a['complete'] for a in attempts),
        'execution_status': dict(Counter(a['status'] for a in attempts)), 'roles': roles,
        'handoffs': [dict(source_id=a['source_id'], **a['handoff_audit']) for a in attempts if 'handoff_audit' in a],
        'prefix_context_failures': sum(len(a.get('context_failures', [])) for a in attempts),
        'chat_template_audit': {'status_counts': dict(Counter(c['chat_template_audit']['status'] for c in calls)),
            'failures': [dict(case_id=a['case_id'], role=c['role'], delegation=c['delegation'], turn=c['turn'],
                             **c['chat_template_audit']) for a in attempts for c in a['calls']
                         if c['chat_template_audit']['status'] != 'match'],
            'comparison': 'HF inference prefix token count vs vLLM usage.prompt_tokens; no token-ID equality claim'},
        'exceptions': [{'case_id': a['case_id'], 'error': a['error']} for a in attempts if 'error' in a],
        'paid_calls': 0, 'quality_results': False, 'research_performance_claim': False,
        'teacher_trimming': False, 'scorer_calls': 0}


def evaluate(*, root=ROOT, backend=None, tok=None):
    from .dryrun_common import tokenizer, PORT, local_analysis_environment
    from verak.v3.agent.backends import PolicyBackend
    with collection_lock(root / 'evaluation.lock'):
        local_analysis_environment()
        manifest = read_json(root / 'evaluation/manifest.json')
        if manifest.get('user_approved_cohorts') != ['smoke20', 'agent_dev10']:
            raise ValueError('Only the explicitly approved 20+10 diagnostic cohorts may run')
        tok = tok or tokenizer()
        own_backend = backend is None
        if own_backend:
            config = load_config()
            config['policy'].update(base_url=f'http://127.0.0.1:{PORT}/v1', model='kanana-dryrun',
                context_limit=8192, generation_reserve=1024,
                adapters={role: 'dryrun-'+role for role in ROLES})
            backend = PolicyBackend(config, root / 'evaluation')
        result = {}
        try:
            for name, cohort in manifest['cohorts'].items():
                output = root / 'evaluation' / name
                analysis = BoundaryParagraphs(load_config(), cache_dir=output / 'bareun')
                attempts = []
                for item in cohort['cases']:
                    row, plan = checked(item['row']), checked(item['plan'])
                    attempts.append(run_case(row, plan, backend, tok, analysis, output_root=output,
                                             case_id=item['case_id']))
                    write_json(root / 'status.json', {'stage': 'evaluating_'+name, 'saved': len(attempts),
                        'planned': len(cohort['cases']), 'at': time.time()})
                result[name] = {**metrics(attempts, planned=len(cohort['cases'])),
                                'cohort': {k: v for k, v in cohort.items() if k != 'cases'}}
                atomic_new(output / 'metrics.json', result[name])
        finally:
            if own_backend:
                backend.close()
        atomic_new(root / 'evaluation/complete.json', {'status': 'complete',
            'manifest': reference(root / 'evaluation/manifest.json'), 'cohorts': result,
            'technical_diagnostics_only': True})
        return result
