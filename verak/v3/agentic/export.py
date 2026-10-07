"""Exact teacher inputs and target-only labels, split by agent; never trains."""
from collections import Counter
import json

from ..common import write_json
from ..phase2 import write_jsonl
from ..train.formatting import encode_labels
from ..view_data import percentiles

ROLES = ('orchestrator', 'composition', 'cohesion')


def select(episodes, corpus):
    best, excluded = {}, Counter()
    for row in episodes:
        if not row.get('completed') or not row.get('reward') or row['episode_id'] not in corpus:
            continue
        for role in ROLES:
            actions = [a for a in row['actions'] if a['role'] == role]
            if not actions:
                excluded[role + ':no_turns'] += 1
                continue
            reward = row['reward']['combined']['R'] if role == 'orchestrator' else row['reward'][role]['R']
            terminal = 'FINISH' if role == 'orchestrator' else 'REPORT'
            sequences = [s for s in row['sequences'] if s['role'] == role]
            if reward < .80:
                excluded[role + ':reward_below_0.80'] += 1
                continue
            if (actions[-1]['action'] != terminal or not actions[-1]['valid'] or
                    any(s['terminal'] != 'REPORT' for s in sequences)):
                excluded[role + ':missing_terminal'] += 1
                continue
            if sum(not a['valid'] for a in actions) > 1:
                excluded[role + ':multiple_rejections'] += 1
                continue
            try:
                for call in row['calls']:
                    if call['role'] == role:
                        value = json.loads(call['raw'])
                        if not isinstance(value, dict):
                            raise ValueError('Action target must be a JSON object')
            except (ValueError, TypeError):
                excluded[role + ':non_json_target'] += 1
                continue
            if (role == 'composition' and not any(r['level'] == 'GLOBAL' for r in corpus[row['episode_id']]['records'])
                    and any(a['action'] in {'MOVE', 'INSERT', 'DELETE'} for a in actions)):
                excluded[role + ':no_global_structural_attempt'] += 1
                continue
            key = row['episode_id'], role
            if key not in best or reward > best[key][0]:
                best[key] = reward, row
    return best, dict(excluded)


def export(episodes, corpus, tokenizer, output):
    selected, excluded = select(episodes, corpus)
    stats = {'excluded': excluded, 'roles': {}}
    for role in ROLES:
        rows = []
        kept = [(eid, reward, row) for (eid, r), (reward, row) in selected.items() if r == role]
        for eid, reward, row in kept:
            for call in row['calls']:
                if call['role'] != role:
                    continue
                # Preserve exact bytes of the actual action JSON; refuse malformed
                # targets instead of training on fabricated repaired responses.
                json.loads(call['raw'])
                messages = call['messages'] + [{'role': 'assistant', 'content': call['raw']}]
                encoded = encode_labels(tokenizer, messages, last_assistant_only=True)
                if len(encoded['input_ids']) > 8192:
                    raise ValueError('Export exceeds inference context')
                rows.append({'episode_id': eid, 'role': role, 'turn': call['turn'], 'delegation': call['delegation'],
                             'selection_reward': reward, 'input': call['messages'], 'target': call['raw'], **encoded})
        write_jsonl(output / (role + '.jsonl'), rows)
        stats['roles'][role] = {'essays': len(kept), 'samples': len(rows),
            'tokens': percentiles([len(r['input_ids']) for r in rows]) if rows else None,
            'target_tokens': percentiles([sum(v != -100 for v in r['labels']) for r in rows]) if rows else None}
    write_json(output / 'stats.json', stats)
    return stats
