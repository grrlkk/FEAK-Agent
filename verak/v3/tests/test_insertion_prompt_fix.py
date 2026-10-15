import pytest
from verak.v3.common import file_sha, read_json, write_json

from verak.v3.agent.runner import system_prompt as v1_prompt
from verak.v3.v2_ops.prompts import system_prompt as v2_prompt
from verak.v3.insertion_boost.prompt_fix import RULE, accepted_insertions, insertion_retention, system_prompt


def test_replacement_changes_only_global_content_rules():
    assert system_prompt('korean') == v2_prompt('korean') == v1_prompt('korean')
    text = system_prompt('global')
    assert text.count(RULE) == 1
    assert '내용이 부족하면 본문에 만들어 넣지 말고' not in text
    assert '글 어디에도 없는 이유·사실·사례' not in text
    assert '모든 STOP summary에는 무엇을 바꿨는지와' in text
    # Frozen source prompts remain untouched and continue to express v1 behavior.
    assert '내용이 부족하면 본문에 만들어 넣지 말고' in v2_prompt('global')
    assert '"action":"EDIT|MOVE|UNDO|STOP"' in v1_prompt('global')


def action(name='INSERT', *, valid=True, created=True, changed=True):
    return {'t': 1, 'action': name, 'valid': valid,
        'created_sids': ['new:1'] if created else [], 'before_hash': 'old',
        'after_hash': 'new' if changed else 'old',
        'args': {'text': '소개 문장이다.', 'target': 'before:S2', 'new_text': '소개 문장이다.'}}


@pytest.mark.parametrize('invalid', [action(valid=False), action(created=False), action(changed=False),
                                   action('EDIT'), action('STOP'), action('SPLIT')])
def test_gate_counts_only_executed_explicit_insert(invalid):
    assert accepted_insertions({'actions_by_role': {'global': [invalid]}}) == []


def test_gate_counts_actual_insert_and_reports_legacy_edit_separately():
    raw = {'actions_by_role': {'global': [action(), action('EDIT')]}}
    assert len(accepted_insertions(raw)) == 1
    assert len(accepted_insertions(raw, explicit=False)) == 2


def test_undone_insertion_is_executed_but_not_retained():
    raw = {'actions_by_role': {'global': [action(), {'t': 2, 'action': 'UNDO', 'valid': True,
        'removed_sids': ['new:1'], 'thought': '중복이므로 취소한다.'}]},
        'stage1_layout': {'paragraphs': []}, 'final_layout': {'paragraphs': []}}
    assert len(accepted_insertions(raw)) == 1
    row = insertion_retention(raw)[0]
    assert row['removed_by_UNDO']
    assert row['UNDO_reasons'] == ['중복이므로 취소한다.']
    assert row['retained_GLOBAL'] == row['retained_final'] == {}


def test_retention_reads_actual_global_and_korean_endpoint_texts():
    raw = {'actions_by_role': {'global': [action()]},
        'stage1_layout': {'paragraphs': [{'units': [{'sid': 'new:1', 'text': '소개 문장이다.'}]}]},
        'final_layout': {'paragraphs': [{'units': [{'sid': 'new:1', 'text': '소개 문장입니다.'}]}]}}
    row = insertion_retention(raw)[0]
    assert row['retained_GLOBAL'] == {'new:1': '소개 문장이다.'}
    assert row['retained_final'] == {'new:1': '소개 문장입니다.'}
    assert not row['removed_by_UNDO']


def test_catalog_preserves_old_files_and_uses_new_design_for_missing_slot(tmp_path):
    from verak.v3.insertion_boost.collection import entries
    config = {'paths': {'v2_ops_output': tmp_path}}
    old = tmp_path / 'teacher_design.json'
    write_json(old, {'orders': {'1': ['e'], '2': ['e']}})
    write_json(tmp_path / 'attempt_1/episodes/e.json', {'v2': {'design_sha256': file_sha(old)}})
    variant = tmp_path / 'prompt_fix/test_v1/design.json'
    write_json(variant, {'original_design_sha256': file_sha(old), 'tasks': [{'attempt': 2, 'episode_id': 'e'}]})
    new = tmp_path / 'prompt_fix/test_v1/attempt_2/episodes/e.json'
    write_json(new, {'v2': {'design_sha256': file_sha(variant)}})
    rows = list(entries(config))
    assert [r['variant'] for r in rows] == ['original', 'test_v1']
    assert rows[1]['path'] == new
    assert rows[1]['scored_path'] == tmp_path / 'scored/attempt_2/e.json'
    write_json(tmp_path / 'attempt_2/episodes/e.json', {'v2': {'design_sha256': file_sha(old)}})
    with pytest.raises(ValueError, match='overwrite or duplicate'):
        list(entries(config))


def test_clean_user_gate_stop_requires_predeclared_denominator_and_count():
    from verak.v3.insertion_boost.collection import finished
    row = {'saved_attempts': 119, 'planned_attempts': 238, 'stop_reason': 'user_prompt_gate_failed',
           'prompt_test_denominator': 20, 'prompt_test_insertions': 5, 'errors': []}
    assert finished(row)
    assert not finished({**row, 'prompt_test_insertions': 6})
    assert not finished({**row, 'prompt_test_denominator': 19})


def test_old_prompt_teacher_cannot_be_accidentally_restarted(tmp_path):
    from verak.v3.insertion_boost.teacher import run
    write_json(tmp_path / 'prompt_fix/pause/snapshot.json', {'status': 'paused'})
    with pytest.raises(RuntimeError, match='paused by the user'):
        run({'paths': {'v2_ops_output': tmp_path}}, max_api_calls=5000, paid_approved=True)


@pytest.mark.parametrize('insertion_count,expected', [(5, 'stop_remaining_generation'), (6, 'resume')])
def test_twenty_essay_gate_has_fixed_denominator(tmp_path, insertion_count, expected):
    from verak.v3.insertion_boost.prompt_fix import summarize_test, finish_test_report
    root = tmp_path / 'prompt_fix/test_v1'
    config = {'paths': {'v2_ops_output': tmp_path}}
    tasks = []
    for i in range(20):
        candidate = tmp_path / f'candidate_{i}.json'
        write_json(candidate, {'records': [{'original_text': {'s': f'original {i}'},
            'recovery_target': {'paragraph': 'p1', 'position': 0}}]})
        tasks.append({'attempt': 1, 'episode_id': f'e{i}', 'source_id': f's{i}', 'candidate': {'path': str(candidate)}})
        raw = {'actions_by_role': {'global': [action()] if i < insertion_count else []},
               'completed': True, 'termination': {'global': 'STOP', 'korean': 'STOP'}, 'steps': {'global': 2}}
        write_json(root / f'attempt_1/episodes/e{i}.json', raw)
    write_json(root / 'design.json', {'tasks': tasks, 'original_total_slots': 238})
    write_json(root / 'status.json', {'saved': 20, 'errors': [], 'budget_before': {'confirmed_usd': 3},
        'budget': {'confirmed_usd': 3.1, 'reserved_usd': .003, 'pending': 0}})
    result = summarize_test(config)
    assert result['decision'] == expected
    assert result['explicit_INSERT_essays'] == insertion_count
    assert result['denominator'] == result['distinct_sources'] == 20
    assert result['recovery']['unknown'] == 20
    assert len(result['examples']) == 5
    write_json(tmp_path / 'prompt_fix/pause/snapshot.json', {'saved_attempts': 99, 'pending': [{'reserved': .003}]})
    write_json(tmp_path / 'prompt_fix/diagnosis82.json', {'rows': [None]*82})
    detail = finish_test_report(config)
    assert detail['extra_old_attempts_preserved_outside_diagnosis'] == 17
    assert detail['unknown_recovery_count'] == 20
    assert detail['mean_R_over'] is None
    assert len(detail['examples']) == 5
    assert 'Unknown cases are not assigned zero' in (root / 'report.md').read_text()
