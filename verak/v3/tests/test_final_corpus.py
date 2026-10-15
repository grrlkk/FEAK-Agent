"""Balancing keeps whole essays, prioritizes TEXT-heavy removals, and never copies."""

from copy import deepcopy
from collections import Counter
import json
import pytest

from verak.v3.corrupt.final_corpus import balanced_subset, distribution, judgment_metrics


def examples(counts):
    result=[]
    for pattern,n in counts:
        for i in range(n):
            sid=f'{pattern}:{i}'
            result.append({'episode_id':sid,'source_id':sid,'question':'synthetic question',
                'corrupted_text':f'synthetic example {sid}', 'split':'agent_train', 'genre':'논증', 'level':'L1',
                'records':[{'level':level,'op':{'WORD':'L_CONN','SENTENCE':'L_CONJ','TEXT':'L_REGISTER'}[level]}
                           for level,count in zip(('WORD','SENTENCE','TEXT'),pattern) for _ in range(count)]})
    return result


def test_already_balanced_keeps_every_essay():
    rows=examples([((1,0,0),33),((0,1,0),33),((0,0,1),33),((0,0,0),9)])
    selected,meta=balanced_subset(rows)
    assert len(selected)==len(rows)
    assert meta['balance_removed_ids']==[]


def test_text_heavy_removed_first_with_largest_retained_set():
    rows=examples([((1,0,0),40),((0,1,0),40),((0,0,1),80)])
    before=deepcopy(rows)
    selected,meta=balanced_subset(rows)
    assert len(selected)==129
    assert meta['non_text_heavy_removed']==0
    assert distribution(selected)['record_levels']=={'TEXT':49,'SENTENCE':40,'WORD':40}
    assert rows==before
    assert all(.28<=v<=.38 for v in distribution(selected)['local_shares'].values())


def test_non_text_removals_only_when_necessary():
    rows=examples([((1,0,0),30),((0,1,0),45),((0,0,1),100)])
    selected,meta=balanced_subset(rows)
    assert meta['non_text_heavy_removed']==5
    assert len(selected)==107


def test_deterministic_selection_whole_rows_no_duplication():
    rows=examples([((1,0,0),30),((0,1,0),30),((0,0,1),90),((1,1,1),10)])
    chosen,meta=balanced_subset(rows,seed=41)
    again,other=balanced_subset(list(reversed(rows)),seed=41)
    assert chosen==again and meta==other
    lookup={r['episode_id']:r for r in rows}
    assert len(chosen)==len({r['episode_id'] for r in chosen})
    assert all(r==lookup[r['episode_id']] for r in chosen)


def test_cannot_invent_missing_level():
    with pytest.raises(ValueError,match='feasible'):
        balanced_subset(examples([((1,0,0),10),((0,0,1),10)]))


def test_existing_duplicate_texts_not_used_for_balance():
    rows=examples([((1,0,0),10),((0,1,0),10),((0,0,1),10)])
    dup=deepcopy(rows[0]); dup['episode_id']='duplicate-id'
    rows.append(dup)
    chosen,meta=balanced_subset(rows)
    assert len(chosen)==30
    assert len(meta['duplicate_ids_removed'])==1
    assert len({(r['question'],r['corrupted_text']) for r in chosen})==30


def test_local_damage_shares_count_records_not_essay_labels():
    rows=examples([((1,1,1),2),((0,0,1),1)])
    d=distribution(rows)
    assert d['essays']==3 and d['local_records']==7
    assert d['local_shares']['TEXT']==pytest.approx(3/7)


def test_partial_audit_distinguishes_api_failures_from_qc_rejection(tmp_path, monkeypatch):
    import verak.v3.corrupt.final_corpus as corpus
    from verak.v3.common import read_json, write_json
    from verak.v3.phase2 import write_jsonl
    from verak.v3.tests.test_instance_pilot import judgment_row
    rows = [dict(judgment_row(sid), genre='논증') for sid in ('pilot', 'accepted', 'failed', 'pending')]
    pilot = {'calls':100, 'cost_nanos':1_000_000_000, 'ids':['pilot']}
    original = tmp_path/'pilot'; original.mkdir()
    output = tmp_path/'full'; output.mkdir()
    config = {'paths':{'phase3b_output':original, 'phase3b_full_output':output}}
    monkeypatch.setattr(corpus, 'load_population', lambda _: (rows, rows[:1], pilot))
    write_jsonl(original/'calls.jsonl', [{'usage':{'input_tokens':100, 'output_tokens':10}}])
    successful = {'sample_id':'accepted', 'status':'completed', 'model':'gpt-6.1-sol',
        'response_model':'gpt-6.1-sol', 'phase_call':101, 'prompt_sha256':'synthetic',
        'usage':{'input_tokens':100, 'output_tokens':10}, 'raw':json.dumps(rows[1]['llm_judgment'])}
    failed = {'sample_id':'failed', 'status':'error', 'error_type':'InternalServerError', 'http_status':503}
    write_jsonl(output/'calls.jsonl', [successful, failed])
    write_json(output/'preserved_hashes.json', {})
    ledger = corpus.CostLedger(output/'cost_ledger.json', pilot, max_api_calls=3033)
    ledger.reserve('accepted',100_000_000); ledger.reserve('failed',100_000_000)
    result = corpus.audit_judgments(config)
    assert (result['judged'], result['failed_requests'], result['unattempted']) == (2,1,1)
    assert result['final'] is False and result['scored'] is False
    assert result['unknown_cost_upper_usd'] == pytest.approx(.1)
    metrics = read_json(output/'partial/metrics.json')
    assert metrics['essay_yield']['overall'] == {'judged':2, 'kept':2, 'yield':1.}
    unresolved = read_json(output/'partial/unresolved.json')
    assert unresolved['unattempted_ids'] == ['pending']
    assert unresolved['failed'][0]['episode_id'] == 'failed'
    assert not (output/'balance_manifest.json').exists()
    write_json(output/'judging_status.json', {'complete':False, 'cost_usd':1.1, 'errors':[]})
    with pytest.raises(ValueError, match='must finish'):
        corpus.balance_corpus(config)
    assert not (output/'balance_manifest.json').exists()


@pytest.fixture
def saved_scores(tmp_path):
    from verak.v3.common import pair_key, sha_text, write_json
    config = {'paths':{'phase3b_full_output':tmp_path/'full', 'metadata':tmp_path/'data'},
              'scorer':{'gpu':1}}
    rows = []
    for split in ('agent_train', 'agent_dev'):
        question, text = f'synthetic {split}', f'synthetic essay {split}'
        row = {'episode_id':split, 'split':split, 'question':question, 'corrupted_text':text,
               'question_hash':sha_text(question), 'corrupted_hash':sha_text(text), 'genre':'논증',
               'q_corrupted':None}
        result = {'episode_id':split, 'question_hash':row['question_hash'],
                  'corrupted_hash':row['corrupted_hash'], 'fingerprint':'same-synthetic-model',
                  'average_k':1, 'score':{'expected':[6.5]*8, 'mean':6.5, 'genre':'논증',
                  'cache_key':pair_key(question,text), 'input_tokens':100}}
        write_json(tmp_path/'full/scores'/f'{split}.json', result)
        rows.append(row)
    return config, rows


def test_scored_corpus_publish_is_exact_and_resumable(saved_scores):
    from verak.v3.common import file_sha
    from verak.v3.phase2 import read_jsonl
    from verak.v3.corrupt.final_corpus import publish_scored
    config, rows = saved_scores
    result = publish_scored(config, rows, workers=1)
    assert result['scored'] == 2 and all(r['q_corrupted'] is None for r in rows)
    final = config['paths']['metadata']/'corrupt'
    assert read_jsonl(final/'agent_train.jsonl')[0]['q_corrupted'] == 6.5
    original_hash = file_sha(final/'agent_train.jsonl')
    (final/'agent_dev.jsonl').unlink()  # Simulate interruption between the two output writes.
    assert publish_scored(config, rows, workers=1) == result
    assert file_sha(final/'agent_train.jsonl') == original_hash


def test_incomplete_or_invalid_scores_cannot_publish_either_split(saved_scores):
    from verak.v3.common import read_json, write_json
    from verak.v3.corrupt.final_corpus import publish_scored
    config, rows = saved_scores
    path = config['paths']['phase3b_full_output']/'scores/agent_dev.json'
    saved = read_json(path); path.unlink()
    with pytest.raises(FileNotFoundError): publish_scored(config, rows, workers=1)
    assert not (config['paths']['metadata']/'corrupt').exists()
    saved['score']['mean'] = 7.5  # The mean must agree with the eight expectations.
    write_json(path,saved)
    with pytest.raises(ValueError, match='provenance'): publish_scored(config, rows, workers=1)
    assert not (config['paths']['metadata']/'corrupt').exists()


def test_mixed_scorer_fingerprints_cannot_publish(saved_scores):
    from verak.v3.common import read_json, write_json
    from verak.v3.corrupt.final_corpus import publish_scored
    config, rows = saved_scores
    path = config['paths']['phase3b_full_output']/'scores/agent_dev.json'
    saved = read_json(path); saved['fingerprint'] = 'another-model'
    write_json(path,saved)
    with pytest.raises(ValueError, match='fingerprint'): publish_scored(config, rows, workers=1)
    assert not (config['paths']['metadata']/'corrupt').exists()
