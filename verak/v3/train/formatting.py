"""Inference-identical policy contexts and assistant-only labels; no trainer."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import numpy as np

from ..agent.runner import fit_history, system_prompt
from ..common import write_json, file_sha, sha_text


def select_roles(episodes, corpus_rows):
    rows=[row for row in episodes if row.get('completed') and row.get('reward')]
    has_global={row['corpus_episode_id']:any(r['level']=='GLOBAL' for r in
        corpus_rows[row['corpus_episode_id']]['records']) for row in rows}
    global_values=[r['reward']['global']['R'] for r in rows if has_global[r['corpus_episode_id']]]
    korean_values=[r['reward']['korean']['R'] for r in rows]
    threshold={'global':float(np.quantile(global_values,.6)) if global_values else None,
               'korean':float(np.quantile(korean_values,.6)) if korean_values else None}
    kept={'global':[],'korean':[]}
    reasons={}
    for row in rows:
        id=row['corpus_episode_id']
        if has_global[id] and row['reward']['global']['R']>=threshold['global']:
            kept['global'].append(id)
            reasons[id]='GLOBAL_record_R_ge_p60'
        elif (not has_global[id] and row['termination']['global']=='STOP' and row['steps']['global']<=2
              and row['reward']['global']['R_over']==0):
            kept['global'].append(id)
            reasons[id]='no_GLOBAL_STOP_within_2_steps_R_over_0'
        if row['reward']['korean']['R']>=threshold['korean']:
            kept['korean'].append(id)
    return {'percentile':60,'percentile_method':'linear (n-1)*q; ties included',
        'thresholds':threshold,'threshold_population':{'global':len(global_values),'korean':len(korean_values)},
        'kept_ids':kept,'global_reasons':reasons,'global_reason_counts':dict(Counter(reasons.values())),
        'extra_minimum_reward_or_STOP_gate':False}


def encode_labels(tokenizer, messages, *, last_assistant_only=False):
    """Mask system/user/tool/header tokens, using template token boundaries.

    A prefix check catches token merging or a non-prefix-stable template instead of
    silently shifting labels. Assistant content and its end-of-turn token have loss.
    """
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
    labels=[-100]*len(ids)
    indices=[i for i,m in enumerate(messages) if m['role']=='assistant']
    if last_assistant_only:indices=indices[-1:]
    spans=[]
    for i in indices:
        prompt=tokenizer.apply_chat_template(messages[:i],tokenize=True,add_generation_prompt=True)
        completed=tokenizer.apply_chat_template(messages[:i+1],tokenize=True,add_generation_prompt=False)
        if ids[:len(completed)]!=completed or completed[:len(prompt)]!=prompt:
            raise ValueError('Chat template is not prefix-stable; cannot mask safely')
        labels[len(prompt):len(completed)]=ids[len(prompt):len(completed)]
        spans.append([len(prompt),len(completed)])
    return {'input_ids':ids,'labels':labels,'assistant_spans':spans}


def formatted_turns(row, role, tokenizer, context_limit):
    backend=SimpleNamespace(name='policy',context_limit=context_limit)
    for i,call in enumerate(c for c in row['calls'] if c['role']==role):
        t=int(call['turn'].split(':')[0])
        prior_actions=row['actions_by_role'][role][:t-1]
        messages,compacted=fit_history(call['messages'],backend,tokenizer,prior_actions)
        if messages[0]['content']!=system_prompt(role):
            raise ValueError('Prompt mismatch between pilot and current policy inference')
        target={'role':'assistant','content':call['raw']}
        full=messages+[target]
        encoded=encode_labels(tokenizer,full,last_assistant_only=True)
        prompt_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if encoded['input_ids'][:len(prompt_ids)]!=prompt_ids:
            raise ValueError('Training prefix differs from policy generation prefix')
        if len(encoded['input_ids'])>context_limit:
            raise ValueError('Actual teacher action does not fit the policy context')
        yield {'episode_id':row['corpus_episode_id'],'role':role,'turn':call['turn'],
            'sample_index':i,'messages':full,**encoded,'history_compacted':compacted,
            'prompt_tokens':len(prompt_ids),'total_tokens':len(encoded['input_ids']),
            'loss_tokens':sum(v!=-100 for v in encoded['labels']),
            'loss_policy':'current_assistant_only; historical assistants are context, each target trained once',
            'work_journal_present':any(m['role']=='user' and m['content'].startswith('[작업 일지]\n') for m in messages)}


def lengths(values):
    if not values:return {'n':0,'mean':None,'median':None,'p90':None,'max':None,'above_8192':0}
    return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p90':float(np.quantile(values,.9)),'max':max(values),'min':min(values),
            'above_8192':sum(v>8192 for v in values)}


def export(config,episodes,selection,tokenizer):
    root=config['paths']['phase7_pilot_output']/'sft_pilot'
    root.mkdir(exist_ok=True)
    lookup={r['corpus_episode_id']:r for r in episodes}
    context_limit=config['policy']['context_limit']
    result={'policy_base_revision':config['policy']['base_revision'],
        'tokenizer_sha256':file_sha(config['paths']['policy_base']/'tokenizer.json'),
        'chat_template_sha256':sha_text(tokenizer.chat_template),'context_limit':context_limit,
        'training_recipe_max_length':8192,'truncated':0,'training_run':False,'roles':{}}
    for role,ids in selection['kept_ids'].items():
        trajectory_path=root/(role+'.trajectories.jsonl.gz')
        turn_path=root/(role+'.turns.jsonl.gz')
        full_lengths,turn_lengths,max_turn_lengths=[],[],[]
        compacted_turns,loss_tokens=0,0
        errors=[]
        with gzip.open(trajectory_path,'wt',encoding='utf-8') as trajectories,gzip.open(turn_path,'wt',encoding='utf-8') as turns:
            for id in ids:
                row=lookup[id]
                history=deepcopy(row['messages_by_role'][role])
                while history and history[-1]['role']!='assistant':history.pop()
                full=encode_labels(tokenizer,history)
                full_lengths.append(len(full['input_ids']))
                samples=[]
                try:
                    samples=list(formatted_turns(row,role,tokenizer,context_limit))
                except ValueError as exc:
                    errors.append({'episode_id':id,'error':str(exc)})
                for sample in samples:
                    turns.write(json.dumps(sample,ensure_ascii=False)+'\n')
                    turn_lengths.append(sample['total_tokens'])
                    compacted_turns+=sample['history_compacted']
                    loss_tokens+=sample['loss_tokens']
                if samples:max_turn_lengths.append(max(s['total_tokens'] for s in samples))
                # Full transcript is an audit representation. Turn samples are inference-exact SFT inputs.
                trajectories.write(json.dumps({'episode_id':id,'role':role,'messages':history,**full,
                    'audit_only_full_history':True,'inference_turn_samples':len(samples),
                    'total_tokens':len(full['input_ids']),'max_inference_turn_tokens':max((s['total_tokens'] for s in samples),default=None)},ensure_ascii=False)+'\n')
        result['roles'][role]={'kept_trajectories':len(ids),'full_transcript':lengths(full_lengths),
            'inference_turns':lengths(turn_lengths),'max_inference_turn_per_trajectory':lengths(max_turn_lengths),
            'compacted_turns':compacted_turns,'assistant_loss_tokens':loss_tokens,'format_errors':errors,
            'selected_json_parse_error_turns':sum(not c['json_valid'] for id in ids for c in lookup[id]['calls'] if c['role']==role),
            'selected_protocol_invalid_turns':sum(not c['valid_json_action'] for id in ids for c in lookup[id]['calls'] if c['role']==role),
            'selected_invalid_actions':sum(not a['valid'] for id in ids for a in lookup[id]['actions_by_role'][role]),
            'trajectories_path':str(trajectory_path),'turns_path':str(turn_path),
            'turns_sha256':file_sha(turn_path),'trajectories_sha256':file_sha(trajectory_path)}
    write_json(root/'formatting.json',result)
    return result
