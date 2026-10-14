from copy import deepcopy
import pytest

from verak.v5.clean_conn import clean_task, excluded, render


def task():
    text = '공부했으려고 힘들다. 집으로 간다.'
    layout = {'gaps': [''], 'tail': '', 'paragraphs': [{'pid': 'P1', 'units': [
        {'sid': 'S1', 'text': '공부했으려고 힘들다.', 'leading': ''},
        {'sid': 'S2', 'text': '집으로 간다.', 'leading': ' '}]}]}
    rec = {'record_id': 'e:R1', 'op': 'L_CONN', 'level': 'WORD', 'sids': ['S1'],
           'original_text': {'S1': '공부했지만 힘들다.'},
           'corrupted_text': {'S1': '공부했으려고 힘들다.'},
           'params': {'span': [4, 6], 'old': '지만', 'new': '으려고'},
           'recovery_target': {'channel': 'EC', 'site': [4, 6]},
           'inverse': {'old': 'snapshot'}, 'before_sha256': 'old', 'after_sha256': 'old2'}
    other = deepcopy(rec)
    other.update(record_id='e:R2', op='L_REGISTER', sids=['S2'])
    return {'episode_id': 'e', 'corrupted_layout': layout, 'corrupted_text': text,
            'records': [rec, other], 'q_corrupted': 8, 'compact_view': 'stale'}


def test_restore_sentence_keep_other_damage_and_invalidate_old_derived_values():
    original = task(); snapshot = deepcopy(original)
    clean, changes = clean_task(original, {'e:R1': {'label': 'ill_formed_morphotactics'}})
    assert original == snapshot
    assert clean['corrupted_text'] == '공부했지만 힘들다. 집으로 간다.'
    assert render(clean['corrupted_layout']) == clean['corrupted_text']
    assert [r['record_id'] for r in clean['records']] == ['e:R2']
    assert clean['records'][0]['op'] == 'L_REGISTER'
    assert 'q_corrupted' not in clean and 'compact_view' not in clean
    assert 'inverse' not in clean['records'][0]
    assert clean['cleanup']['reward_ready'] is False
    assert changes['dropped'] == ['e:R1']


def test_overlapping_damage_must_not_be_silently_restored():
    row = task(); row['records'][1]['sids'] = ['S1']
    with pytest.raises(ValueError, match='Overlapping'):
        clean_task(row, {'e:R1': {'label': 'ill_formed_morphotactics'}})


def test_relabel_uses_conjunction_class_and_full_form_span_without_changing_text():
    row = task(); r = row['records'][0]
    r['original_text']['S1'] = '하지만 좋다.'
    r['corrupted_text']['S1'] = '하니까 좋다.'
    r['params'].update(old='지만', new='니까', span=[1,3])
    row['corrupted_layout']['paragraphs'][0]['units'][0]['text'] = '하니까 좋다.'
    row['corrupted_text'] = render(row['corrupted_layout'])
    value, _ = clean_task(row, {'e:R1': {'label': 'grammatical_form_relation_changed',
        'conjunction_misclassified_as_EC': True}})
    r = value['records'][0]
    assert r['op'] == 'L_CONJ' and r['level'] == 'SENTENCE'
    assert r['recovery_target']['channel'] == 'initial_conjunction'
    assert r['recovery_target']['coarse_class'] == 'ADVERSATIVE'
    assert r['recovery_target']['site'] == [0, 3]
    assert r['params']['old'] == '하지만' and r['params']['new'] == '하니까'
    assert value['corrupted_text'] == row['corrupted_text']


def test_uncertain_is_retained_and_exclusion_covers_all_roles_but_not_other_practices():
    row = task()
    value, changes = clean_task(row, {'e:R1': {'label': 'uncertain_predicate_selection'}})
    assert value == row and changes['uncertain'] == ['e:R1']
    for role in ('korean', 'revision', 'orchestrator', 'oneshot'):
        sample = {'role': role, 'provenance': {'episode_id': 'e'}}
        assert excluded(sample, {'e'}, set())
        sample['provenance']['episode_id'] = 'other-practice'
        assert excluded(sample, {'e'}, set()) is None
