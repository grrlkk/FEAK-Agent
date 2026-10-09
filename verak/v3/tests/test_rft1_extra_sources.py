from copy import deepcopy
import json

import pytest

from verak.v3.common import write_json,file_sha,sha_text
from verak.v3.rft1.extra_sources import position_payload,validate_source,cap_source_practices


def fixture(tmp_path):
    prior,held=tmp_path/'prior.json',tmp_path/'held.json'
    write_json(prior,{'signatures':{}})
    write_json(held,{'validation':['valid:2']})
    candidate={'source_id':'valid:1','source_hash':sha_text('원문'),'source_text':'원문','source_question_hash':'q',
        'records':[{'op':'G_PARA_SWAP','params':{'paragraph_indices':[2,1]}}]}
    payload=position_payload(candidate)
    candidate['source_provenance']={'schema_version':1,'pool':'phase3b_agent_train',
        'active_corpus_source':True,'source_id':'valid:1','source_hash':candidate['source_hash'],
        'source_question_hash':'q','position_signature':sha_text(json.dumps(payload,sort_keys=True,ensure_ascii=False)),
        'position_payload':payload,'prior_position_index_path':str(prior),'prior_position_index_sha256':file_sha(prior),
        'sft_holdout_manifest_sha256':file_sha(held)}
    kwargs={'prior_path':prior,'holdout_path':held,'source_scores':{'valid:1':{'eligible':True,
        'essay_hash':candidate['source_hash'],'question_hash':'q'}}}
    return candidate,kwargs


def test_active_source_requires_verified_new_position(tmp_path):
    row,kwargs=fixture(tmp_path)
    signature=validate_source(row,{'valid:1'},**kwargs)
    assert signature==row['source_provenance']['position_signature']
    unproved=deepcopy(row)
    del unproved['source_provenance']
    with pytest.raises(ValueError,match='needs new-position'):
        validate_source(unproved,{'valid:1'},**kwargs)
    write_json(kwargs['prior_path'],{'signatures':{signature:{}}})
    row['source_provenance']['prior_position_index_sha256']=file_sha(kwargs['prior_path'])
    with pytest.raises(ValueError,match='repeats a prior'):
        validate_source(row,{'valid:1'},**kwargs)


@pytest.mark.parametrize('change',('source_text','source_question_hash','position','holdout'))
def test_rejects_changed_source_position_or_holdout(tmp_path,change):
    row,kwargs=fixture(tmp_path)
    if change=='position':
        row['records'][0]['params']['paragraph_indices']=[1,3]
    elif change=='holdout':
        write_json(kwargs['holdout_path'],{'validation':['valid:1']})
    else:
        row[change]='different'
    with pytest.raises(ValueError):
        validate_source(row,{'valid:1'},**kwargs)


def test_cap_selects_four_highest_rewards_per_source_and_operator():
    rows=[{'episode_id':str(i),'source_id':'source','R':i/10,
        'operators':{'G_PARA_SWAP':{}}} for i in range(6)]
    rows += [{'episode_id':'move','source_id':'source','R':.9,'operators':{'G_SENT_MOVE':{}}}]
    before=deepcopy(rows)
    kept,dropped=cap_source_practices(rows)
    assert set(dropped)=={'0','1'} and len(kept)==5 and rows==before
    assert all(x['episode_id'] not in {'0','1'} for x in kept)
