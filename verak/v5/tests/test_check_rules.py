from types import SimpleNamespace

from verak.v5.check_rules import spacing_issues, compatible_categories, leading_subject
from verak.v5.check_a4 import RuleEnvironment, ancestral_sid
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row


def test_spacing_refinements_require_identical_nonspace_text_and_source_offsets():
    text = '학생 은 간다.'
    raw = {'sentences': [{'text': {'content': text}, 'refined': '학생은 간다.'}]}
    assert spacing_issues(text, raw) == [{'start': 2, 'end': 3, 'original': ' ', 'suggestion': ''}]
    raw['sentences'][0]['refined'] = '교사는 간다.'
    assert not spacing_issues(text, raw)
    raw['sentences'][0]['refined'] = '학생은 간다.'
    raw['sentences'][0]['text']['begin_offset'] = 4
    assert not spacing_issues(text, raw)


def test_map_relation_direction_is_preserved_and_missing_edge_is_not_error():
    graph = {'sentence_relations': [{'source': 'S1', 'target': 'S2', 'type': 'cause_effect'}]}
    assert compatible_categories(graph, 'S2', 'S1')[0] == {'RESULT'}
    assert not compatible_categories(graph, 'S1', 'S2')[0]
    graph['sentence_relations'][0]['type'] = 'contrast'
    assert compatible_categories(graph, 'S1', 'S2')[0] == {'ADVERSATIVE'}
    assert not compatible_categories(graph, 'S3', 'S2')[0]


def test_repeated_subject_requires_initial_explicit_literal_np():
    ann = SimpleNamespace(start=20, subject_evidence=[{'surface': '학생은', 'span': [20, 23]}])
    assert leading_subject(ann) == '학생은'
    ann.subject_evidence[0]['span'] = [21, 24]
    assert leading_subject(ann) is None
    ann.subject_evidence[0].update(span=[20, 23], entity_id='masked1')
    assert leading_subject(ann) is None


def test_rule_handoff_replaces_only_korean_items_preserving_budget(tmp_path, monkeypatch):
    from verak.v5 import check_a4
    item = {'location': ['S1'], 'action': 'EDIT', 'instruction': '규칙 후보', 'needs_search': 'no'}
    monkeypatch.setattr(check_a4, 'candidates', lambda *args: ([{'item': item}], []))
    env = RuleEnvironment(row('첫 문장이다. 둘째 문장이다.'), FakeBoundaryAnalysis(), spacing=None, audit_dir=tmp_path)
    original = [{'location': ['S2'], 'action': 'EDIT', 'instruction': '기존 계획', 'needs_search': 'no'}]
    env.start_korean(original)
    assert env.public_items == [dict(item, item_id='KT1')]
    assert env.remaining == 14
    saved = check_a4.read_json(tmp_path/'pass01.json')
    assert saved['discarded_orchestrator_items'] == original
    assert saved['private_records_opened'] is False


def test_lineage_ancestry_cannot_loop():
    assert ancestral_sid('S1b', {'S1b': {'parent_sid': 'S1a'}, 'S1a': {'parent_sid': 'S1'}, 'S1': {'parent_sid': 'S1'}}) == 'S1'
