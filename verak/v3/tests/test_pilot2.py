"""Short observations, content-scope corpus, and order-aware reward contracts."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from verak.v3.tests.test_environment import setup_env, action, StubParagraphAnalysis
from verak.v3.tests.test_reward import example
from verak.v3.tests.test_phase7 import TinyTemplate
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v3.corrupt.missing_conjunction import candidates
from verak.v3.corrupt.operators import apply, restore_record, exact_restoration_satisfies
from verak.v3.reward.recovery import main_recovery
from verak.v3.reward.overedit import overedit, order_distance
from verak.v3.agent.runner import fit_history, system_prompt
from verak.v3.train.pilot2_data import without_deletions
from verak.v3.train.formatting import select_roles, formatted_turns


def test_drop_roundtrip_and_class_not_exact_word(example):
    doc, bank, _, _ = example
    proposal = next(candidates(doc))
    ann = next(a for a in doc.structure().annotations if a.sid == proposal.sids[0])
    # Cached fixture sentence minus a surface prefix; no live service in unit tests.
    text = ann.text[:proposal.span[0]] + ann.text[proposal.span[1]:]
    tokens = [replace(t, start=t.start-ann.start-proposal.span[1],
                      end=t.end-ann.start-proposal.span[1]) for t in ann.tokens
              if t.start-ann.start >= proposal.span[1]]
    original_tokens = bank.tokens
    bank.tokens = lambda value: deepcopy(tokens) if value == text else original_tokens(value)
    corrupted, record = apply(doc, proposal, bank)
    assert record['level'] == 'SENTENCE' and record['verification'] == 'bareun_reanalysis'
    assert main_recovery(doc, corrupted, record) == 0
    assert main_recovery(doc, doc, record) == 1
    restored = restore_record(corrupted, record, bank)
    assert restored.snapshot() == doc.snapshot() and exact_restoration_satisfies(restored, record)
    different = deepcopy(ann)
    different.initial_conj = {'eligible': True, 'form': '다른 표면형',
                              'coarse_class': record['recovery_target']['coarse_class']}
    assert main_recovery(doc, corrupted, record, annotations={ann.sid: different}) == 1
    different.initial_conj['coarse_class'] = 'WRONG'
    assert main_recovery(doc, corrupted, record, annotations={ann.sid: different}) == 0


def test_drop_does_not_modify_marker_or_multi_unit(example):
    doc, _, _, _ = example
    from verak.src.schemas import Token
    for u in doc.units:
        u.tokens += [Token('다', 'EF', 0, 1), Token('다', 'EF', 1, 2)]
    assert list(candidates(doc)) == []


def test_order_move_and_role_attribution(example):
    doc, _, _, _ = example
    moved = doc.clone()
    moved.paragraphs[0].units.reverse()
    result = overedit(doc, doc, moved, [])
    assert result['morpheme'] == 0 and result['order'] > 0
    assert result['value'] == .5*result['order']
    logs = [{'action': 'MOVE', 'valid': True}]
    assert overedit(doc, doc, moved, [], actions=logs)['order'] == result['order']
    assert overedit(doc, moved, moved, [], actions=[{'action': 'EDIT', 'valid': True}])['order'] == 0
    assert overedit(doc, moved, doc, [], actions=logs)['order'] == 0
    assert overedit(doc, doc, doc, [], actions=logs+[{'action': 'UNDO'}])['order'] == 0


def test_cross_paragraph_order_and_record_exclusion(example):
    doc, _, _, _ = example
    moved = doc.clone()
    u = moved.paragraphs[0].units.pop()
    moved.paragraphs[1].units.insert(0, u)
    assert order_distance(doc, doc, moved, [])['value'] > 0
    assert order_distance(doc, doc, moved, [{'sids': [u.sid]}])['value'] == 0
    assert overedit(doc, doc, doc, [])['value'] == 0


def test_morpheme_and_order_are_equal_weight(example):
    doc, bank, fixture, _ = example
    from verak.v3.corrupt.operators import Proposal
    modified, _ = apply(doc, Proposal(**fixture['proposals']['L_REGISTER']), bank)
    result = overedit(doc, doc, modified, [])
    assert result['order'] == 0 and result['morpheme'] > 0
    assert result['value'] == .5*result['morpheme']


def test_observations_local_neighbors_periodic_full_and_undo(setup_env):
    env, episode, analysis = setup_env
    paragraphs = []
    for p in range(1, 5):
        units = [Unit(f'S{3*(p-1)+i}', f'{3*(p-1)+i}번을 쓴다.',
                      analysis.pieces(f'{i}번을 쓴다.')[0][1], ' ' if i>1 else '') for i in range(1, 4)]
        paragraphs.append(Paragraph(f'P{p}', units))
    episode['document'] = Document(paragraphs, ['', '\n', '\n', '\n'])
    env.mode = 'single'
    assert '[전체 갱신]' in env.reset(episode)
    obs, _, _ = env.step(action('EDIT', target='S5:번', new_text='번째'))
    assert '[부분 갱신' in obs and '[전체 갱신]' not in obs
    assert all(f'S{i} |' in obs for i in (3, 4, 5, 6, 7))
    assert all(f'\nS{i} |' not in obs for i in (1, 2, 8, 9, 10, 11, 12))
    obs, _, _ = env.step(action('UNDO'))
    assert '[부분 갱신' in obs and 'S5 | 5번을 쓴다.' in obs
    env.step(action('EDIT', target='S5:번', new_text='번째'))
    env.step(action('UNDO'))
    obs, _, _ = env.step(action('EDIT', target='S5:번', new_text='번째'))
    assert '[전체 갱신]' in obs and all(f'S{i} |' in obs for i in range(1, 13))


def test_context_uses_latest_full_handoff_and_shared_teacher_policy():
    tok = TinyTemplate()
    history = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content':
        'stale\n[GLOBAL 인계: 행동 및 누적 marker-change notices]\n{"notice":"keep"}\n[전체 갱신]\nstale state'}]
    for i in range(8):
        history.extend([{'role': 'assistant', 'content': 'x'*300}, {'role': 'user', 'content': 'y'*300}])
    history += [{'role': 'user', 'content': '[전체 갱신]\nCURRENT'},
                {'role': 'assistant', 'content': 'do something'}, {'role': 'user', 'content': 'latest notice'}]
    teacher = SimpleNamespace(name='teacher', context_limit=2100)
    policy = SimpleNamespace(name='policy', context_limit=2100)
    context, changed = fit_history(history, teacher, tok, [])
    assert changed and context == fit_history(history, policy, tok, [])[0]
    assert context[0] == history[0]
    assert '[GLOBAL 인계:' in context[1]['content']
    assert context[2]['content'] == '[전체 갱신]\nCURRENT'
    assert context[3]['content'].startswith('[작업 일지]')
    assert 'stale state' not in str(context) and context[-1] == history[-1]
    assert len(tok.apply_chat_template(context, add_generation_prompt=True))+1024 <= 2100


def test_mandatory_context_overflow_fails_without_truncating_profile():
    history = [{'role': 'system', 'content': 'x'*1000}, {'role': 'user', 'content': '[전체 갱신]\n'+'y'*1000}]
    with pytest.raises(ValueError, match='Mandatory'):
        fit_history(history, SimpleNamespace(name='teacher', context_limit=2500), TinyTemplate(), [])


def test_filter_preserves_old_rows_and_removes_entire_mixed_essay():
    rows = [{'episode_id': 'a', 'records': [{'op': 'L_CONN'}]},
            {'episode_id': 'b', 'records': [{'op': 'G_DELETE_SUPPORT'}, {'op': 'L_CONJ'}]}]
    old = deepcopy(rows)
    assert without_deletions(rows) == [rows[0]] and rows == old


@pytest.mark.parametrize('a', [action('MOVE', target='S1', position='after:S2'),
    action('EDIT', target='before:S1', new_text='문장.'), action('EDIT', target='S1', new_text='')])
def test_no_global_selection_rejects_structural_action_even_if_undone(a):
    import json
    id = 'toy'
    row = {'corpus_episode_id': id, 'completed': True, 'reward': {
        'global': {'R': 0., 'R_over': 0.}, 'korean': {'R': 0.}},
        'termination': {'global': 'STOP'}, 'steps': {'global': 2},
        'actions_by_role': {'global': [{**json.loads(a), 'valid': True}]}}
    result = select_roles([row], {id: {'records': [{'level': 'SENTENCE'}]}})
    assert result['kept_ids']['global'] == []
    row['actions_by_role']['global'][0]['valid'] = False
    assert select_roles([row], {id: {'records': [{'level': 'SENTENCE'}]}})['kept_ids']['global'] == []


@pytest.mark.parametrize('role', ['global', 'korean', 'single'])
def test_prompt_allows_conjunction_and_explains_periodic_profile(role):
    prompt = system_prompt(role)
    assert '접속어 추가는 내용 보존 원칙상 허용' in prompt
    assert '매 5번째 행동' in prompt and 'STOP summary' in prompt


def test_qc_and_teacher_share_twenty_dollar_budget(tmp_path):
    from verak.v3.common import load_config
    from verak.v3.eval.api import Phase6API
    from feak_tc.runtime.openai import CallBudgetExceeded
    config = load_config()
    config['paths']['phase7_pilot2_output'] = tmp_path
    api = Phase6API(config, 100, phase='phase7_pilot2')
    api.reserve('conj_drop_qc', 'a', 'a', 1.)
    with api.db() as db:
        db.execute("UPDATE calls SET status='completed',reserved=0,confirmed=19.9")
    with pytest.raises(CallBudgetExceeded):
        api.reserve('teacher_pilot2', 'b', 'b', .11)
    with pytest.raises(ValueError):
        Phase6API(config, 7001, phase='phase7_pilot2')


def test_diagnostic_ignores_unchanged_surface_tag_drift_and_modal_gess():
    from verak.v3.eval.ending_diagnostic import changed_tense_markers
    assert changed_tense_markers('하고 있다.', '하고 있다.', [],
        [('PROGRESSIVE', '고 있', 1, 4)])[0] is False
    assert changed_tense_markers('한다.', '하겠다.', [],
        [('PROSPECTIVE', '겠', 1, 2)])[0] is False
    changed, before, after = changed_tense_markers('살아왔다.', '살아가고 있다.',
        [('PAST', '았', 2, 3)], [('PROGRESSIVE', '고 있', 3, 6)])
    assert changed and before and after


def test_latest_full_replaces_first_for_sft_as_well_as_teacher():
    tok = TinyTemplate()
    history = [{'role': 'system', 'content': system_prompt('global')},
               {'role': 'user', 'content': '[전체 갱신]\nstale'}]
    for i in range(8):
        history += [{'role': 'assistant', 'content': 'a'*500}, {'role': 'user', 'content': 'b'*500}]
    history += [{'role': 'user', 'content': '[전체 갱신]\ncurrent'}]
    backend = SimpleNamespace(name='teacher', context_limit=8192)
    sent, truncated = fit_history(history, backend, tok, [])
    assert truncated
    row = {'corpus_episode_id': 'synthetic', 'actions_by_role': {'global': []}, 'calls': [
        {'role': 'global', 'turn': '1:0', 'messages': sent, 'raw': 'STOP', 'history_compacted': truncated}]}
    sample = next(formatted_turns(row, 'global', tok, 8192))
    assert sample['messages'][:-1] == sent
    assert sample['work_journal_present'] and sample['history_compacted']
    assert all(v == -100 for v in sample['labels'][:sample['prompt_tokens']])
