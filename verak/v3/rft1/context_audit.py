"""Read-only tokenizer measurements of the two known long real essays."""
import json

from ..agent.runner import split_handoff
from ..common import file_sha, read_json, write_json
from ..train.sft_eval import eval_config, saved_rows as sft_rows
from .config import PHASE
from .evaluate import saved_rows


def layout_text(layout):
    return ''.join(gap + ''.join(u['leading'] + u['text'] for u in p['units'])
                   for gap, p in zip(layout['gaps'], layout['paragraphs'])) + layout['tail']


def mandatory_context(messages, actions):
    full = [i for i, m in enumerate(messages) if m['role'] == 'user' and '[전체 갱신]' in m['content']]
    latest = full[-1] if full else 1
    handoff = next((split_handoff(m['content'])[0] for m in messages
                    if m['role'] == 'user' and '[GLOBAL 인계:' in m['content']), None)
    selected = [messages[0]]
    labels = ['system']
    if handoff:
        selected.append({'role': 'user', 'content': handoff})
        labels.append('GLOBAL_handoff')
    selected.append({'role': 'user', 'content': split_handoff(messages[latest]['content'])[1]})
    labels.append('latest_full_observation')
    diary = [{'action': a['action'], 'args': a['args'], 'valid': a['valid']} for a in actions]
    selected.append({'role': 'user', 'content': '[작업 일지]\n' +
                     json.dumps(diary, ensure_ascii=False, sort_keys=True, separators=(',', ':'))})
    labels.append('work_journal')
    suffix = [messages[-1]] if len(messages) - 1 > latest and messages[-1]['role'] == 'user' else []
    return selected, labels, suffix


def audit(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    text_tokens = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    context_tokens = lambda m: len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True))
    result = {'tokenizer': str(config['paths']['policy_base']), 'context': 8192, 'reserve': 1024,
              'maximum_prompt_tokens': 7168, 'essays': {}, 'v1_changed': False}
    old_config = eval_config()
    rows = [(condition, row, path) for condition in ('base', 'epoch_1', 'epoch_2', 'luna_low')
            for row, path, _ in sft_rows(old_config, condition) if row['source_id'] in {'valid:7863', 'valid:7872'}]
    rows.extend(('rft1', row, path) for row, path, _ in saved_rows(config)
                if row['source_id'] in {'valid:7863', 'valid:7872'})
    for condition, row, path in rows:
        text = layout_text(row['initial_layout'])
        essay = result['essays'].setdefault(row['source_id'], {'characters': len(text),
            'essay_tokens': text_tokens(text), 'paragraphs': len(row['initial_layout']['paragraphs']),
            'sentences': sum(len(p['units']) for p in row['initial_layout']['paragraphs']), 'conditions': {}})
        histories = {}
        for role, messages in row['messages_by_role'].items():
            mandatory, labels, suffix = mandatory_context(messages, row['actions_by_role'][role])
            histories[role] = {'final_saved_history_tokens': context_tokens(messages),
                'initial_prompt_tokens': context_tokens(messages[:2]),
                'mandatory_compaction_tokens': context_tokens(mandatory),
                'mandatory_plus_last_observation_tokens': context_tokens(mandatory + suffix),
                'mandatory_components_content_tokens': {label: text_tokens(m['content']) for label, m in zip(labels, mandatory)},
                'last_observation_content_tokens': text_tokens(suffix[0]['content']) if suffix else 0}
        essay['conditions'][condition] = {'path': str(path), 'sha256': file_sha(path),
            'completed': row['completed'], 'runtime_error': row['runtime_error'], 'histories': histories}
    result['proposal'] = ('Keep 8192 context/1024 generation, the same essay IDs, tool rules and reward. '
        'For a separately versioned future renderer, losslessly materialize the current full text/profile from '
        'the latest snapshot plus all later updates, and retain handoff and complete journal once. '
        'Serialize one current snapshot instead of an old full view plus a nearly full repeated paragraph. '
        'Validate ID/text/fact equivalence and measure the entire prompt before adoption. '
        'Do not truncate, split the evaluation essay, increase context, or hide failed episodes. '
        'This proposal is not applied in SFT/RFT1 evaluation.')
    write_json(config['paths'][PHASE + '_output'] / 'context_audit.json', result)
    return result
