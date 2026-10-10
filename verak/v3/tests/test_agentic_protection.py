"""Conservative two-hop relevance protection from either raw extraction."""
from copy import deepcopy

import pytest

from verak.v3.agentic import graph
from verak.v3.tests.test_agentic import act, env


def edge(source, target, label):
    return {'source': source, 'target': target, 'label': label}


def extraction(*edges, off_topic=()):
    return {'sentence_edges': list(edges), 'paragraph_edges': [], 'off_topic': list(off_topic)}


def test_union_protection_two_hops_does_not_change_intersected_edges():
    ids = [f'S{i}' for i in range(1, 9)]
    left = extraction(edge('S1', 'Q', 'addresses'), edge('S2', 'S1', 'supports'),
                      edge('S4', 'S3', 'contrasts'), edge('S6', 'S5', 'contrasts'), off_topic=ids)
    right = extraction(edge('S5', 'Q', 'addresses'), edge('S3', 'S2', 'example_of'),
                       edge('S7', 'S6', 'supports'), edge('S8', 'S7', 'supports'), off_topic=ids)
    before = deepcopy((left, right))
    kept, counts = graph.intersection(left, right)
    metadata = graph.relevance_protection(left, right, ids)
    assert metadata == {
        'seeds': ['S1', 'S5'], 'protected_ids': ['S1', 'S2', 'S3', 'S5', 'S6', 'S7'],
        'depths': {'S1': 0, 'S2': 1, 'S3': 2, 'S5': 0, 'S6': 1, 'S7': 2},
        'overridden_off_topic': ['S1', 'S2', 'S3', 'S5', 'S6', 'S7']}
    assert kept == {'sentence_edges': [], 'paragraph_edges': [], 'off_topic': ids}
    assert counts['sentence_edges'] == {'left': 4, 'right': 4, 'intersection': 0, 'union': 8}
    assert (left, right) == before


def test_contrasts_are_directed_and_shortest_depths_handle_cycles():
    left = extraction(edge('S1', 'Q', 'addresses'), edge('S1', 'S2', 'contrasts'),
                      edge('S3', 'S1', 'contrasts'), edge('S4', 'S3', 'supports'))
    right = extraction(edge('S1', 'S4', 'contrasts'), edge('S4', 'S1', 'example_of'),
                       edge('S5', 'S4', 'supports'))
    metadata = graph.relevance_protection(left, right, ['S1', 'S2', 'S3', 'S4', 'S5'])
    assert metadata['depths'] == {'S1': 0, 'S3': 1, 'S4': 1, 'S5': 2}
    assert 'S2' not in metadata['protected_ids']


def test_only_valid_sentence_relations_grant_protection():
    left = extraction(edge('S1', 'Q', 'addresses'), edge('P1', 'S1', 'supports'),
                      edge('S2', 'Q', 'supports'), edge('S3', 'S1', 'addresses'),
                      edge('S4', 'S1', 'elaborates'), edge('S5', 'S1', 'continues'),
                      edge('S6', 'S6', 'supports'), edge('foreign', 'Q', 'addresses'),
                      edge(['S2'], 'S1', 'supports'), {'source': 'S2'}, 'invalid',
                      {**edge('S2', 'S1', 'supports'), 'extra': True})
    left['paragraph_edges'] = [edge('S2', 'S1', 'supports'), edge('P1', 'P2', 'continues')]
    metadata = graph.relevance_protection(left, extraction(), [f'S{i}' for i in range(1, 7)])
    assert metadata == {'seeds': ['S1'], 'protected_ids': ['S1'],
                        'depths': {'S1': 0}, 'overridden_off_topic': []}


def test_derived_metadata_requires_deliberate_validation():
    left = extraction(edge('S1', 'Q', 'addresses'), off_topic=['S1', 'S2'])
    right = extraction(edge('S2', 'S1', 'supports'), off_topic=['S1', 'S2'])
    kept, _ = graph.intersection(left, right)
    kept['relevance_protection'] = graph.relevance_protection(left, right, ['S1', 'S2'])
    with pytest.raises(ValueError, match='Invalid graph fields'):
        graph.validate(kept, ['S1', 'S2'], [])
    assert graph.validate(kept, ['S1', 'S2'], [], allow_protection=True) == kept
    for field, bad in [('depths', {'S1': 0, 'S2': 3}), ('depths', {'S1': False, 'S2': 1}),
                       ('seeds', []), ('protected_ids', ['S1', 'S2', 'S2']),
                       ('overridden_off_topic', ['S1'])]:
        invalid = deepcopy(kept)
        invalid['relevance_protection'][field] = bad
        with pytest.raises(ValueError):
            graph.validate(invalid, ['S1', 'S2'], [], allow_protection=True)


def protected_env(env):
    env.version = 3
    left = extraction(edge('S1', 'Q', 'addresses'), off_topic=['S1', 'S2', 'S3'])
    right = extraction(edge('S2', 'S1', 'supports'), off_topic=['S1', 'S2', 'S3'])
    env.discourse, _ = graph.intersection(left, right)
    env.discourse['relevance_protection'] = graph.relevance_protection(left, right, ['S1', 'S2', 'S3'])
    env.initial_graph = env.graph()
    return env


def test_protection_overrides_both_labels_without_asserting_relevance(env):
    env = protected_env(env)
    current = env.graph()
    assert current['off_topic_candidate'] == ['S3']
    assert current['relevance_protection']['overridden_off_topic'] == ['S1', 'S2']
    assert current['sentence_edges'] == []
    assert current['unsupported'] == []
    query = act(env, 'composition', 'QUERY', target='off_topic')['result']
    assert query['off_topic_candidate'] == ['S3']
    assert '보호는 관련성 판정이 아님' in query['note']
    audit = act(env, 'orchestrator', 'AUDIT')['result']
    assert audit['off_topic_candidate'] == ['S3']
    assert set(audit) == {'off_register', 'predecessor_changes', 'off_topic_candidate', 'dangling'}


def test_saved_state_derivation_matches_live_state_and_preserves_other_fields(env):
    env = protected_env(env)
    expected = env.graph()
    raw_discourse = deepcopy(env.discourse)
    del raw_discourse['relevance_protection']
    env.discourse = raw_discourse
    saved_state = env.graph()
    assert saved_state['off_topic_candidate'] == ['S1', 'S2', 'S3']
    unaffected = {key: deepcopy(value) for key, value in saved_state.items()
                  if key not in {'off_topic_candidate', 'off_topic_rule', 'relevance_protection'}}
    protected_discourse = {**raw_discourse, 'relevance_protection': expected['relevance_protection']}
    before = deepcopy(protected_discourse)
    assert graph.apply_protection(saved_state, protected_discourse) is saved_state
    assert saved_state == expected
    assert protected_discourse == before
    assert all(saved_state[key] == value for key, value in unaffected.items())
    graph.apply_protection(saved_state, raw_discourse)
    assert saved_state == env.graph()


def test_protection_does_not_bypass_or_add_deletion_gates_or_cover_insertions(env):
    env = protected_env(env)
    env.begin_editor({'task': 'S2 삭제', 'scope': 'all'})
    assert not act(env, 'composition', 'DELETE', target='S2')['valid']
    assert act(env, 'composition', 'PREVIEW',
               action={'action': 'DELETE', 'args': {'target': 'S2'}})['valid']
    assert act(env, 'composition', 'DELETE', target='S2')['valid']
    assert 'S2' not in env.graph()['relevance_protection']['protected_ids']
    assert act(env, 'composition', 'INSERT', target='after:S1', new_text='설명이다.')['valid']
    current = env.graph()
    assert 'N1' not in current['relevance_protection']['protected_ids']
    assert 'N1' not in current['off_topic_candidate']
    assert current['off_topic_candidate'] == ['S3']


def test_protection_does_not_change_raw_extraction_contract(env):
    messages, schema = graph.request(env.document, '의견은?', version=3)
    assert set(schema['properties']) == {'sentence_edges', 'paragraph_edges', 'off_topic'}
    assert messages[0]['content'] == graph.PROMPT_V3
    # Explicit disagreement never becomes a candidate, with or without a seed.
    left = extraction(off_topic=['S1', 'S3'])
    right = extraction(off_topic=['S2', 'S3'])
    metadata = graph.relevance_protection(left, right, ['S1', 'S2', 'S3'])
    assert metadata == {'seeds': [], 'protected_ids': [], 'depths': {}, 'overridden_off_topic': []}
    assert graph.intersection(left, right)[0]['off_topic'] == ['S3']
