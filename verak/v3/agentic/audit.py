"""Validate saved experimental contracts without model or analyzer calls."""
from collections import Counter
import json

from ..common import read_json, file_sha, write_json
from .data import PHASE, prepare
from .environment import ALLOWED, PROMPTS
from .schemas import action_schema
from .graph import intersection, validate


def audit(config, rows, tokenizer):
    design, train, dev, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    failures, checks = [], Counter()
    def check(condition, label, item=None):
        checks[label] += 1
        if not condition:
            failures.append({'check': label, 'item': item})
    for i, baseline in design['baseline_files'].items():
        check(file_sha(baseline['path']) == baseline['sha256'], 'baseline_unchanged', i)
    for path in (root / 'graphs').glob('*.json'):
        value = read_json(path)
        check(len(value['runs']) == 2, 'two_graph_extractions', value['id'])
        if value['status'] != 'completed':
            continue
        intersected, counts = intersection(*[r['discourse'] for r in value['runs']])
        check(intersected == value['discourse'], 'exact_intersection', value['id'])
        validate(intersected, value['sentence_ids'].values(), value['paragraph_ids'].values())
        requests = [read_json(root / 'api/requests' / f"{r['phase_call']:06}.json") for r in value['runs']]
        check(requests[0]['messages'] == requests[1]['messages'], 'identical_graph_inputs', value['id'])
        check(all(r['model'] == 'gpt-6-luna' and r['reasoning_effort'] == 'low' for r in requests), 'graph_model', value['id'])
    for row in rows:
        eid = row['episode_id']
        question = train[eid]['question'] if eid in train else examples[eid].question
        check(row.get('delegations', 0) <= 4, 'delegation_limit', eid)
        check(row.get('score_calls', 0) <= 2, 'score_limit', eid)
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
            check(call['messages'][0]['content'] == PROMPTS[role], 'same_role_prompt', eid)
            check(call['messages'][-1]['content'].startswith('[문항] ' + question), 'question_at_top', eid)
            obs = call['messages'][-1]['content']
            check(all(s in obs for s in ('[계획]', '[진행]', '[최근 REPORT]', '[맡은 일]')), 'mandatory_observation_fields', eid)
            check(call['input_tokens'] <= 7168 and call['total_tokens'] <= 8192, 'policy_context_limit', eid)
            request = read_json(root / 'api/requests' / f"{call['phase_call']:06}.json")
            check(request['messages'] == call['messages'], 'exact_request_observation', eid)
            check(request['schema'] == action_schema(role), 'role_output_schema', eid)
            check(request['model'] == 'gpt-6-luna' and request['reasoning_effort'] == 'low' and
                  request['max_output_tokens'] == 1024, 'teacher_settings', eid)
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
    result = {'passed': not failures, 'checks': dict(checks), 'failures': failures, 'api_calls': 0}
    write_json(root / 'validation.json', result)
    if failures:
        raise ValueError(f'{len(failures)} saved-contract checks failed; see validation.json')
    return result
