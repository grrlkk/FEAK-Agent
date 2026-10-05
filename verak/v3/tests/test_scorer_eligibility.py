"""Unparseable scores cannot be invented or silently dropped from a corpus."""

import pytest

from verak.v3.common import file_sha, read_json, write_json
from verak.v3.phase2 import read_jsonl, write_jsonl
from verak.v3.corrupt.final_corpus import distribution, exclude_unscorable, score_corpus
from verak.v3.tests.test_final_corpus import examples, saved_scores


def selection(tmp_path, per_level=5):
    out = tmp_path/'full'; out.mkdir()
    config = {'paths':{'phase3b_full_output':out, 'metadata':tmp_path/'data'}}
    manifest, all_rows = {'splits':{}}, []
    for split in ('agent_train','agent_dev'):
        rows = examples([((1,0,0),per_level),((0,1,0),per_level),((0,0,1),per_level)])
        for row in rows:
            row['episode_id'] = split+':'+row['episode_id']
            row['split'] = split
        path = out/f'balanced_{split}.jsonl'
        kept_path = out/f'kept_{split}.jsonl'
        write_jsonl(path, rows); write_jsonl(kept_path, rows)
        manifest['splits'][split] = {'balanced_path':str(path),'balanced_sha256':file_sha(path),
            'kept_path':str(kept_path),'kept_sha256':file_sha(kept_path),
            'before':distribution(rows),'after':distribution(rows),'pattern_allocation':[]}
        all_rows.extend(rows)
    failed = all_rows[3*per_level-1]['episode_id']
    for row in all_rows:
        if row['episode_id'] != failed:
            write_json(out/'scores'/(row['episode_id']+'.json'), {'fingerprint':'frozen'})
    write_json(out/'balance_manifest.json',manifest)
    write_json(out/'scoring_status.json',{'scored':len(all_rows)-1,'total':len(all_rows),
        'errors':[{'episode_id':failed,'error':'ScoreParseError'}]})
    write_json(out/'score_parse_diagnostics.json',[{'episode_id':failed,'status':'error',
        'error_type':'ScoreParseError','fingerprint':'frozen','first_lines':[{'text':'7 7 7'}]}])
    return config, failed


def test_preserves_full_kept_and_initial_selection_while_filtering_only_failed_instance(tmp_path):
    config, failed = selection(tmp_path)
    out = config['paths']['phase3b_full_output']
    paths = [out/'balance_manifest.json'] + list(out.glob('*agent_*.jsonl'))
    hashes = {p:file_sha(p) for p in paths}
    result = exclude_unscorable(config)
    assert result == {'excluded_ids':[failed],'final_counts':{'agent_train':14,'agent_dev':15}}
    assert file_sha(out/'balance_manifest_before_scorer_exclusions.json') == hashes[out/'balance_manifest.json']
    assert all(file_sha(p)==h for p,h in hashes.items() if p.name!='balance_manifest.json')
    manifest = read_json(out/'balance_manifest.json')
    assert not manifest['selection_uses_q_corrupted']
    assert manifest['selection_requires_parseable_scores']
    selected = [r for p in out.glob('balanced_scorable_*.jsonl') for r in read_jsonl(p)]
    assert len(selected)==29 and failed not in {r['episode_id'] for r in selected}


@pytest.mark.parametrize('kind',['valid_line','other_error','valid_saved_score'])
def test_failure_evidence_cannot_be_used_to_filter_valid_or_unrelated_scores(tmp_path,kind):
    config, failed = selection(tmp_path)
    out = config['paths']['phase3b_full_output']
    before = file_sha(out/'balance_manifest.json')
    diagnostic = read_json(out/'score_parse_diagnostics.json')
    if kind=='valid_line': diagnostic[0]['first_lines'][0]['text']='7 7 7 7 7 7 7 7'
    if kind=='other_error': diagnostic[0]['error_type']='RuntimeError'
    if kind=='valid_saved_score': write_json(out/'scores'/(failed+'.json'),{'fingerprint':'frozen'})
    write_json(out/'score_parse_diagnostics.json',diagnostic)
    with pytest.raises(ValueError): exclude_unscorable(config)
    assert file_sha(out/'balance_manifest.json')==before
    assert not (out/'balance_manifest_before_scorer_exclusions.json').exists()


def test_refuses_exclusion_that_would_break_balance(tmp_path):
    config, _ = selection(tmp_path,per_level=3)
    out = config['paths']['phase3b_full_output']
    before = file_sha(out/'balance_manifest.json')
    with pytest.raises(ValueError,match='violate'): exclude_unscorable(config)
    assert file_sha(out/'balance_manifest.json')==before
    assert not list(out.glob('balanced_scorable_*.jsonl'))


def test_all_saved_scores_resume_refreshes_status_without_inference(saved_scores):
    config, rows = saved_scores
    out = config['paths']['phase3b_full_output']
    config['scorer'].update(mode='expected',average_k=1)
    config['paths']['phase3_output'] = out/'unused_source_cache'
    manifest = {'splits':{}}
    for row in rows:
        path = out/(row['split']+'.jsonl')
        write_jsonl(path,[row])
        manifest['splits'][row['split']] = {'balanced_path':str(path),'balanced_sha256':file_sha(path)}
    write_json(out/'balance_manifest.json',manifest)
    write_json(out/'scoring_status.json',{'errors':[{'episode_id':'previous-unscorable'}]})
    assert score_corpus(config)['scored']==2
    status = read_json(out/'scoring_status.json')
    assert status['scored']==status['total']==status['reused_saved_scores']==2
    assert status['newly_scored']==0 and status['errors']==[]
