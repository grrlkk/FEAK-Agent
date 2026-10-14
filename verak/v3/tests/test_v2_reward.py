"""v2 recovery and insertion penalties, using immutable Bareun fixture tokens."""
from copy import deepcopy
from dataclasses import replace

from verak.src.schemas import Token
from verak.v3.common import load_config
from verak.v3.corrupt.operators import Proposal, apply
from verak.v3.tests.test_reward import example
from verak.v3.v2_ops.operators import connective, delete_link, ending_prefix, fusion_candidates
from verak.v3.v2_ops.reward import fuse_recovery, insertion_origins, link_recovery, rewards_v2


def prepend(unit, form):
    shift = len(form) + 1
    unit.text = form + ' ' + unit.text
    unit.tokens = [Token(form, 'MAG', 0, len(form))] + [replace(t, start=t.start+shift, end=t.end+shift) for t in unit.tokens]


def fuse_record(paragraph, position, ids, classes):
    return {'op': 'L_FUSE', 'level': 'GLOBAL', 'record_id': 'r1', 'sids': ids,
            'coupled_changes': [], 'recovery_target': {'paragraph': paragraph, 'position': position,
                'source_sids': ids, 'source_classes': classes}}


def test_fusion_three_sentences_requires_every_boundary_and_preserves_content(example):
    source, _, _, _ = example
    record = fuse_record('P1', 0, ['S1', 'S2', 'S3'], ['NONE', 'NONE'])
    restored = source.clone()
    restored.locate('S2')[2].sid = 'S1a'
    restored.locate('S3')[2].sid = 'S1aa'
    assert fuse_recovery(source, restored, record)[0] == 1
    prepend(restored.locate('S1a')[2], '또한')
    assert fuse_recovery(source, restored, record)[0] == 1
    altered = restored.clone()
    prepend(altered.locate('S1aa')[2], '그러나')
    assert fuse_recovery(source, altered, record)[0] == 0
    restored.paragraphs[0].units.pop()
    assert fuse_recovery(source, restored, record)[0] == 0


def test_fusion_restores_the_source_conjunction_class(example):
    source, bank, fixture, _ = example
    record = fuse_record('P2', 0, ['S4', 'S5'], ['RESULT'])
    assert fuse_recovery(source, source, record)[0] == 1
    changed, _ = apply(source, Proposal(**fixture['proposals']['L_CONJ']), bank)
    assert fuse_recovery(source, changed, record)[0] == 0
    changed = source.clone()
    unit = changed.locate('S5')[2]
    unit.tokens = unit.tokens[:-2]
    # A missing content morpheme cannot be repaired by merely restoring punctuation.
    unit.tokens = [t for t in unit.tokens if t.form != '물']
    assert fuse_recovery(source, changed, record)[0] == 0


def test_deletion_match_is_inserted_at_source_position_and_extra_insertions_cost(example):
    source, _, _, _ = example
    corrupted, record = delete_link(source, 'S4', 'topic')
    record['record_id'] = 'r1'
    middle = source.clone()
    middle.locate('S4')[2].sid = 'N1'
    actions = [{'action': 'INSERT', 'args': {'position': 'before:S5', 'text': source.locate('S4')[2].text},
                'valid': True, 'created_sids': ['N1'], 'changed_sids': ['N1', 'S5']}]
    config = load_config()
    config['method_version'] = 'v2'
    kwargs = dict(config=config, genre='논증', q_corrupted=5, q_stage1=5, q_final=5,
                  stage1=middle, stage1_actions=actions, stage2_actions=[], judge=lambda r,u: 1.)
    reward = rewards_v2(source, corrupted, middle, [record], **kwargs)
    assert reward['global']['R_rec'] == 1 and reward['combined']['R_over'] == 0
    assert all(c['recovered'] == 1 for c in reward['combined']['per_record'][0]['coupled_changes'])
    extra = deepcopy(source.locate('S9')[2])
    extra.sid = 'N2'
    middle.paragraphs[-1].units.append(extra)
    actions.append({'action': 'EDIT', 'args': {'target': 'after:S9', 'new_text': extra.text},
                    'valid': True, 'created_sids': ['N2'], 'changed_sids': ['N2', 'S9']})
    reward = rewards_v2(source, corrupted, middle, [record], **kwargs)
    assert reward['global']['R_rec'] == 1 and reward['global']['R_over'] > 0
    assert reward['combined']['R_over'] > 0 and reward['korean']['R_over'] == 0
    wrong_place = source.clone()
    wrong_place.locate('S4')[2].sid = 'N1'
    item = wrong_place.paragraphs[1].units.pop(0)
    wrong_place.paragraphs[2].units.append(item)
    called = []
    assert link_recovery(wrong_place, record, {'N1'}, lambda r,u: called.append(u) or 1.)[0] == 0
    assert not called
    assert link_recovery(source, record, set(), lambda r,u: 1.)[0] == 0


def test_fusion_exclusions_and_generation_mapping_do_not_change_v1_sets(example):
    from verak.v3.ko.coarse import COARSE_EC
    source = example[0].clone()
    before = deepcopy(COARSE_EC)
    prepend(source.locate('S2')[2], '예를 들어')
    prepend(source.locate('S3')[2], '즉')
    _, counts = fusion_candidates(source)
    assert counts['excluded_EXAMPLE'] == 2
    assert counts['excluded_RESTATEMENT'] == 2
    assert counts['excluded_EXAMPLE_or_RESTATEMENT'] == 3
    assert connective('좋', 'RESULT') == '좋아서'
    assert connective('하', 'RESULT') == '해서'
    assert connective('하', 'ADVERSATIVE') == '하지만'
    assert connective('하', 'ADDITION') == '하고'
    assert connective('먹', 'NONE', 1) == '먹으며'
    assert ending_prefix(example[0].locate('S1')[2]).endswith('중요하')
    assert COARSE_EC == before and '고' not in COARSE_EC
    origins = insertion_origins([
        {'action': 'INSERT', 'created_sids': ['N1']},
        {'action': 'SPLIT', 'created_sids': ['N1a'], 'changed_sids': ['N1', 'N1a']},
        {'action': 'SPLIT', 'created_sids': ['S1a'], 'changed_sids': ['S1', 'S1a']}])
    assert origins == {'N1': 0, 'N1a': 0}
