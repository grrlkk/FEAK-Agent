"""Validate the approved source-diverse teacher additions at future export.

No rollout, environment, reward, model, or training implementation is changed.
Active-corpus sources require evidence of a new structural position; historical
unused-source additions retain their original provenance contract.
"""
from collections import Counter
import json

from ..common import file_sha,read_json,sha_text


def position_payload(candidate):
    record=candidate['records'][0]
    params=record['params']
    if record['op']=='G_PARA_SWAP':
        position={'paragraph_indices':sorted(params['paragraph_indices'])}
    else:
        position={'sentence_id':record['sids'][0],**{k:params[k] for k in
            ('from_paragraph','from_position','to_paragraph','to_position')}}
    return {'source_id':candidate['source_id'],'source_hash':candidate['source_hash'],
            'operator':record['op'],'position':position}


def validate_source(candidate,active_sources,*,prior_path,holdout_path,source_scores):
    active=candidate['source_id'] in active_sources
    proof=candidate.get('source_provenance')
    if proof is None:
        if active:
            raise ValueError('Active-source extra teacher needs new-position provenance')
        return 'historical_unused_source'
    if proof.get('schema_version')!=1 or proof.get('pool')!='phase3b_agent_train':
        raise ValueError('Unrecognized extra source provenance')
    for key in ('source_id','source_hash','source_question_hash'):
        if proof.get(key)!=candidate.get(key):
            raise ValueError('Extra source identity differs from its proof')
    if proof.get('active_corpus_source') is not active:
        raise ValueError('Extra source active-corpus membership changed')
    if candidate['source_hash']!=sha_text(candidate['source_text']):
        raise ValueError('Extra source text changed')
    source=source_scores.get(candidate['source_id'])
    if (not source or not source['eligible'] or source['essay_hash']!=candidate['source_hash']
            or source['question_hash']!=candidate['source_question_hash']):
        raise ValueError('Extra teacher source is not in the frozen eligible train pool')
    if (proof.get('prior_position_index_path')!=str(prior_path)
            or proof.get('prior_position_index_sha256')!=file_sha(prior_path)
            or proof.get('sft_holdout_manifest_sha256')!=file_sha(holdout_path)):
        raise ValueError('Extra source prior-position/holdout evidence changed')
    payload=position_payload(candidate)
    signature=sha_text(json.dumps(payload,sort_keys=True,ensure_ascii=False))
    if proof.get('position_payload')!=payload or proof.get('position_signature')!=signature:
        raise ValueError('Extra structural position differs from its proof')
    if signature in read_json(prior_path)['signatures']:
        raise ValueError('Extra teacher repeats a prior structural position')
    return signature


def cap_source_practices(entries,cap=4):
    counts=Counter()
    kept,dropped=[],[]
    for entry in sorted(entries,key=lambda e:(-e['R'],e['episode_id'])):
        if len(entry['operators'])!=1:
            raise ValueError('An extra practice must contain exactly one operator')
        operator=next(iter(entry['operators']))
        key=(entry['source_id'],operator)
        if counts[key]>=cap:
            dropped.append(entry['episode_id'])
        else:
            kept.append(entry)
            counts[key]+=1
    return sorted(kept,key=lambda e:e['episode_id']),dropped
