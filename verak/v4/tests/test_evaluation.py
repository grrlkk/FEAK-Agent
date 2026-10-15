import copy
import pytest

from verak.v4.evaluation import link_recovery_candidates


def snapshot(*groups):
    return {'paragraphs': [{'pid': pid, 'units': [{'sid': sid, 'text': sid} for sid in ids]}
                           for pid, ids in groups]}


def record(pid='P2', position=0, sid='S2'):
    return {'op': 'G_DEL_LINK', 'sids': [sid], 'recovery_target': {'paragraph': pid, 'position': position}}


def test_empty_paragraph_accepts_immediate_boundaries_and_keeps_inputs_immutable():
    source = snapshot(('P1', ['S1']), ('P2', ['S2']), ('P3', ['S3']))
    damaged = snapshot(('P1', ['S1']), ('P2', []), ('P3', ['S3']))
    final = snapshot(('P1', ['S1', 'N1']), ('P2', []), ('P3', ['N2', 'S3']))
    saved = copy.deepcopy((source, damaged, final))
    result = link_recovery_candidates(source, damaged, final, record(), {'N1', 'N2'}, version='v4')
    assert [r['sid'] for r in result] == ['N1', 'N2']
    assert all(r['source_paragraph_emptied'] for r in result)
    assert (source, damaged, final) == saved


def test_ordinary_rule_does_not_allow_adjacent_paragraph_or_distant_source_position():
    source = snapshot(('P1', ['S0']), ('P2', ['S1', 'S2', 'S3']), ('P3', ['S4']))
    damaged = snapshot(('P1', ['S0']), ('P2', ['S1', 'S3']), ('P3', ['S4']))
    final = snapshot(('P1', ['S0', 'N0']), ('P2', ['S1', 'N1', 'S3', 'N3']), ('P3', ['N2', 'S4']))
    result = link_recovery_candidates(source, damaged, final, record(position=1), {'N0','N1','N2','N3'}, version='v4')
    assert [r['sid'] for r in result] == ['N1']


def test_empty_neighbor_is_not_skipped_and_nonboundary_insertions_are_excluded():
    source = snapshot(('P1', ['S1']), ('P2', ['S2']), ('P3', ['S3']), ('P4', ['S4']))
    damaged = snapshot(('P1', ['S1']), ('P2', []), ('P3', ['S3']), ('P4', ['S4']))
    final = snapshot(('P1', ['N0','S1']), ('P2', []), ('P3', []), ('P4', ['N1','S4']))
    assert link_recovery_candidates(source, damaged, final, record(), {'N0','N1'}, version='v4') == []


def test_source_empty_wrapper_may_be_missing_and_occupied_insertion_is_excluded():
    source = snapshot(('P1', ['S1']), ('P2', ['S2']), ('P3', ['S3']))
    damaged = snapshot(('P1', ['S1']), ('P3', ['S3']))
    final = snapshot(('P1', ['S1']), ('P3', ['N1','S3']))
    assert len(link_recovery_candidates(source, damaged, final, record(), {'N1'}, version='v4')) == 1
    assert link_recovery_candidates(source, damaged, final, record(), {'N1'}, occupied={'N1'}, version='v4') == []


@pytest.mark.parametrize('version', ['v1','v2','v3'])
def test_previous_versions_cannot_enable_new_boundary_rule(version):
    with pytest.raises(ValueError, match='v4 only'):
        link_recovery_candidates({}, {}, {}, {}, set(), version=version)
