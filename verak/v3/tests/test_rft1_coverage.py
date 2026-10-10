from verak.v3.rft1.coverage import coverage


def test_four_samples_required_and_all_same_operator_records_in_one_sample():
    records = [{'record_id': str(i), 'op': 'G_SENT_MOVE', 'level': 'GLOBAL'} for i in (1, 2)]
    corpus = {'a': {'records': records, 'level': 'L3'}, 'b': {'records': records, 'level': 'L3'}}
    rows = []
    for eid, sample, values in [('a', 1, [1, 0]), ('a', 2, [0, 1]), ('a', 3, [0, 0]),
                                 ('a', 4, [0, 0]), ('b', 1, [1, 1])]:
        rows.append({'corpus_episode_id': eid, 'rft1': {'sample': sample},
                     'reward': {'global': {'per_record': [dict(r, main=v) for r, v in zip(records, values)]}}})
    result = coverage(corpus, rows)
    assert result['essays_with_four_saved_samples'] == 1
    assert result['partial_essays_excluded'] == {'b': [1]}
    main = result['operators']['G_SENT_MOVE']['role_main']
    assert main['share'] == 0 and main['record_share'] == 1
    assert main['unresolved_essays'] == 0


def test_missing_reward_is_unknown_and_completed_global_survives_later_error():
    record = {'record_id': 'x', 'op': 'G_OFFTOPIC', 'level': 'GLOBAL'}
    corpus = {'a': {'records': [record], 'level': 'L3'}}
    rows = [{'corpus_episode_id': 'a', 'rft1': {'sample': i}, 'reward': None} for i in range(1, 5)]
    result = coverage(corpus, rows)['operators']['G_OFFTOPIC']
    assert result['role_main']['share'] == 0
    assert result['role_main']['upper_share_if_unknown_success'] == 1
    rows[1]['global_only_reward'] = {'per_record': [dict(record, main=1)]}
    result = coverage(corpus, rows)['operators']['G_OFFTOPIC']
    assert result['role_main']['share'] == 1
    assert result['combined_full_record']['unresolved_essays'] == 1
