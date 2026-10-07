"""Validate saved experimental contracts without model or analyzer calls."""
from collections import Counter
import json

from ..common import read_json, file_sha, write_json
from .data import PHASE, prepare
from .environment import ALLOWED, prompts_for, named_sentence_ids
from .schemas import action_schema
from .graph import intersection, validate
from ..train.pilot import safe_id


def scope_sentence_ids(rows, scope):
    paragraphs = list(dict.fromkeys(r['paragraph'] for r in rows))
    if scope == 'all':
        selected = set(paragraphs)
    else:
        ends = scope.split('-')
        first, last = paragraphs.index(ends[0]), paragraphs.index(ends[-1])
        selected = set(paragraphs[first:last + 1])
    return {r['sid'] for r in rows if r['paragraph'] in selected}


def audit_v3_row(row, initial_graph, check):
    """Verify v3 controller contracts from saved actions, calls and return records."""
    eid = row['episode_id']
    actions, sequences = row.get('actions', []), row.get('sequences', [])
    by_delegation = {s['delegation']: s for s in sequences}
    check(len(by_delegation) == len(sequences), 'unique_editor_delegations', eid)
    check(not actions or row.get('version') == 3, 'episode_version', eid)
    flagged = set(initial_graph.get('off_topic_candidate', []))
    initial_nodes = {s['id']: s for s in initial_graph.get('sentences', [])}
    initial_prev = {sid: value['predecessor'] for sid, value in initial_nodes.items()}
    notices, deletion_counts, delegation = [], Counter(), 0
    valid_scores = 0
    for index, action in enumerate(actions):
        if not action.get('valid'):
            continue
        name, args = action['action'], action['args']
        if name == 'SCORE':
            valid_scores += 1
            check(index == 0, 'score_only_at_start', eid)
        if name == 'DELEGATE':
            delegation += 1
            sequence = by_delegation.get(delegation)
            check(sequence is not None, 'delegate_has_sequence', eid)
            if sequence is not None:
                check(sequence['task'] == args['task'] and sequence['role'] == args['agent'] and
                      sequence['scope'] == args.get('scope', 'all'), 'exact_delegation_task_scope', eid)
            try:
                scoped = scope_sentence_ids(action['before_rows'], args.get('scope', 'all'))
            except (KeyError, ValueError):
                check(False, 'valid_delegation_scope', eid)
                scoped = set()
            expected = list({(f['sid'], f['type']): f for f in notices if f['sid'] in scoped}.values())
            check(action['result'].get('disturbed_markers') == expected, 'factual_scoped_disturbed_markers', eid)
            if args['agent'] == 'cohesion':
                check(bool(expected), 'cohesion_requires_disturbed_markers', eid)
        if name == 'DELETE':
            sequence = by_delegation.get(action['delegation'])
            check(sequence is not None and args['target'] in named_sentence_ids(sequence['task']),
                  'delete_target_named_in_task', eid)
            exact = any(previous['valid'] and previous['role'] == 'composition' and previous['action'] == 'PREVIEW' and
                        previous['delegation'] == action['delegation'] and previous['before_rows'] == action['before_rows'] and
                        previous['args']['action'] == {'action': 'DELETE', 'args': args} for previous in actions[:index])
            check(exact, 'delete_exact_preview_in_same_state', eid)
            deletion_counts[action['delegation']] += 1
            check(deletion_counts[action['delegation']] <= 2, 'delete_limit_per_delegation', eid)
        if action['role'] == 'composition' and name in {'MOVE', 'INSERT', 'DELETE', 'UNDO'}:
            notices.extend(action['result']['marker_changes'])
            current = {s['sid'] for s in action['after_rows']}
            check(action['result'].get('off_topic_candidate') == sorted(current & flagged),
                  'post_edit_explicit_off_topic_only', eid)
        audit_results = [action['result']] if name == 'AUDIT' else [action['result']['audit_at_finish']] if name == 'FINISH' else []
        for result in audit_results:
            check(set(result) == {'off_register', 'predecessor_changes', 'off_topic_candidate', 'dangling'},
                  'audit_facts_only_no_unsupported', eid)
            current = {s['sid'] for s in action['before_rows']}
            check(result.get('off_topic_candidate') == sorted(current & flagged), 'audit_explicit_off_topic_only', eid)
            order = [s['sid'] for s in action['before_rows']]
            predecessors = {sid: order[n - 1] if n else None for n, sid in enumerate(order)}
            for changed in result.get('predecessor_changes', []):
                sid = changed['sid']
                node = initial_nodes.get(sid, {})
                check(sid in initial_prev and changed['before'] == initial_prev[sid] and
                      changed['after'] == predecessors.get(sid) and changed['before'] != changed['after'] and
                      bool(changed.get('conjunction') or changed.get('subject_omitted') or
                           node.get('conjunction') or node.get('subject_omitted')),
                      'audit_predecessor_fact_since_start', eid)
    if actions:
        check(row.get('delegations') == delegation, 'delegation_count_matches_actions', eid)
        check(row.get('score_calls') == valid_scores, 'score_count_matches_actions', eid)
    for sequence in sequences:
        key = sequence['role'], sequence['delegation']
        editor_actions = [a for a in actions if (a['role'], a['delegation']) == key]
        calls = [c for c in row.get('calls', []) if (c['role'], c['delegation']) == key]
        check(len(editor_actions) <= len(calls) <= len(editor_actions) + bool(row.get('runtime_error')) and
              [c['turn'] for c in calls] == list(range(1, len(calls) + 1)), 'editor_actions_have_ordered_call_evidence', eid)
        check(isinstance(sequence.get('final_notice_sent'), bool) and isinstance(sequence.get('auto_report'), bool),
              'explicit_editor_budget_fields', eid)
        if sequence.get('final_notice_sent'):
            check(len(calls) >= 15 or (len(calls) == 14 and bool(row.get('runtime_error'))),
                  'final_notice_reached_last_two_steps', eid)
        for call in calls:
            if call['turn'] >= 15:
                check(sequence.get('final_notice_sent') is True and '[최종 알림]' in call['messages'][-1]['content'] and
                      'done (budget)' in call['messages'][-1]['content'], 'final_notice_with_two_steps_remaining', eid)
        if sequence['terminal'] == 'AUTO_REPORT':
            check(sequence.get('auto_report') is True and sequence.get('final_notice_sent') is True and
                  len(editor_actions) == 16 and not any(a['valid'] and a['action'] == 'REPORT' for a in editor_actions),
                  'auto_report_only_at_exhausted_editor_budget', eid)
            report = sequence.get('report') or {}
            check(report.get('agent') == sequence['role'] and report.get('origin') == 'controller' and
                  report.get('status') == 'done (budget)', 'controller_budget_report_status', eid)
        elif sequence['terminal'] == 'REPORT':
            check(sequence.get('auto_report') is False and bool(editor_actions) and editor_actions[-1]['valid'] and
                  editor_actions[-1]['action'] == 'REPORT' and sequence.get('report') == editor_actions[-1]['result'],
                  'explicit_report_preserved', eid)
        else:
            check(sequence['terminal'] is None and sequence.get('auto_report') is False and bool(row.get('runtime_error')),
                  'unfinished_editor_is_recorded_failure', eid)
    if row.get('final_graph'):
        current = {s['id'] for s in row['final_graph']['sentences']}
        check(row['final_graph'].get('off_topic_candidate') == sorted(current & flagged), 'final_explicit_off_topic_only', eid)


def audit(config, rows, tokenizer):
    design, train, dev, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    version = config[PHASE].get('version', 2)
    failures, checks = [], Counter()
    def check(condition, label, item=None):
        checks[label] += 1
        if not condition:
            failures.append({'check': label, 'item': item})
    for i, baseline in design['baseline_files'].items():
        check(file_sha(baseline['path']) == baseline['sha256'], 'baseline_unchanged', i)
    for path in (root / 'graphs').glob('*.json'):
        value = read_json(path)
        if value['status'] == 'completed':
            check(len(value['runs']) == 2, 'two_graph_extractions', value['id'])
        if value['status'] != 'completed':
            continue
        intersected, counts = intersection(*[r['discourse'] for r in value['runs']])
        check(intersected == value['discourse'], 'exact_intersection', value['id'])
        validate(intersected, value['sentence_ids'].values(), value['paragraph_ids'].values())
        if version == 3:
            check('off_topic' in intersected and all('off_topic' in r['discourse'] for r in value['runs']),
                  'explicit_off_topic_in_both_extractions', value['id'])
            check(value['graph'].get('off_topic_candidate') == intersected.get('off_topic'),
                  'graph_uses_explicit_agreement', value['id'])
            if 'off_topic_rule' in value['graph']:
                check(value['graph']['off_topic_rule'] == 'explicit_agreement', 'explicit_rule_metadata_when_present', value['id'])
        requests = [read_json(root / 'api/requests' / f"{r['phase_call']:06}.json") for r in value['runs']]
        check(requests[0]['messages'] == requests[1]['messages'], 'identical_graph_inputs', value['id'])
        check(all(r['model'] == 'gpt-6-luna' and r['reasoning_effort'] == 'low' for r in requests), 'graph_model', value['id'])
    for row in rows:
        eid = row['episode_id']
        question = train[eid]['question'] if eid in train else examples[eid].question
        check(row.get('delegations', 0) <= (3 if version == 3 else 4), 'delegation_limit', eid)
        check(row.get('score_calls', 0) <= (1 if version == 3 else 2), 'score_limit', eid)
        check(sum(a['role'] == 'orchestrator' for a in row['actions']) <= 12, 'orchestrator_step_limit', eid)
        per_turn = Counter((a['role'], a['delegation']) for a in row['actions'] if a['role'] != 'orchestrator')
        check(all(n <= 16 for n in per_turn.values()), 'editor_step_limit', eid)
        for a in row['actions']:
            if a['valid']:
                check(a['action'] in ALLOWED[a['role']], 'role_permissions', eid)
            if a['role'] == 'orchestrator':
                check(a['before_hash'] == a['after_hash'], 'orchestrator_never_edits', eid)
            if a['action'] == 'PREVIEW':
                check(a['before_rows'] == a['after_rows'], 'preview_no_mutation', eid)
        for call in row['calls']:
            role = call['role']
            check(call['messages'][0]['content'] == prompts_for(version)[role], 'same_role_prompt', eid)
            check(call['messages'][-1]['content'].startswith('[문항] ' + question), 'question_at_top', eid)
            obs = call['messages'][-1]['content']
            check(all(s in obs for s in ('[계획]', '[진행]', '[최근 REPORT]', '[맡은 일]')), 'mandatory_observation_fields', eid)
            check(call['input_tokens'] <= 7168 and call['total_tokens'] <= 8192, 'policy_context_limit', eid)
            request = read_json(root / 'api/requests' / f"{call['phase_call']:06}.json")
            check(request['messages'] == call['messages'], 'exact_request_observation', eid)
            check(request['schema'] == action_schema(role), 'role_output_schema', eid)
            check(request['model'] == 'gpt-6-luna' and request['reasoning_effort'] == 'low' and
                  request['max_output_tokens'] == 1024, 'teacher_settings', eid)
        if version == 3 and row.get('actions'):
            graph_path = root / 'graphs' / (safe_id(eid) + '.json')
            check(graph_path.exists(), 'episode_has_saved_graph', eid)
            initial = read_json(graph_path)['graph'] if graph_path.exists() else {}
            audit_v3_row(row, initial, check)
        if row['completed']:
            check(bool(row['actions']) and row['actions'][-1]['action'] == 'FINISH' and row['actions'][-1]['valid'], 'finish_completion', eid)
            if row.get('reward'):
                r = row['reward']['combined']
                check(abs(r['R'] - sum(r['weighted_components'].values())) < 1e-9, 'reward_components', eid)
    for role in ALLOWED:
        path = root / 'export' / (role + '.jsonl')
        if not path.exists():
            continue
        with path.open() as source:
            for line in source:
                value = json.loads(line)
                prompt = tokenizer.apply_chat_template(value['input'], tokenize=True, add_generation_prompt=True)
                ids, labels = value['input_ids'], value['labels']
                check(ids[:len(prompt)] == prompt and all(x == -100 for x in labels[:len(prompt)]) and
                      labels[len(prompt):] == ids[len(prompt):], 'export_input_masked_target_only', value['episode_id'])
                check(len(ids) <= 8192, 'export_context_limit', value['episode_id'])
                if version == 3 and role != 'orchestrator':
                    row = next((r for r in rows if r['episode_id'] == value['episode_id']), None)
                    check(row is not None and all(s['terminal'] == 'REPORT' and s.get('auto_report') is False
                        for s in row['sequences'] if s['role'] == role), 'no_auto_reports_in_editor_export', value['episode_id'])
    expected_ids = set(design['corrupted_ids'] + design['real_ids'])
    saved_ids = {row['episode_id'] for row in rows}
    result = {'passed': not failures, 'checks': dict(checks), 'failures': failures, 'api_calls': 0,
              'coverage': {'intended_episodes': len(expected_ids), 'saved_episodes': len(rows),
                           'unattempted_ids': sorted(expected_ids - saved_ids),
                           'completed_episodes': sum(bool(r.get('completed')) for r in rows)},
              'coverage_note': 'Contract checks apply to saved successful actions and requests; incomplete and unattempted episodes are not claimed completed.'}
    write_json(root / 'validation.json', result)
    if failures:
        raise ValueError(f'{len(failures)} saved-contract checks failed; see validation.json')
    return result
