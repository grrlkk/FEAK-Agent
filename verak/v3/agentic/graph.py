"""Question-rooted, intersected discourse and factual Bareun marker state."""
from collections import Counter
from copy import deepcopy
import json

from ..observation.graph import object_schema, array_schema, enum, public_input, build

PROMPT = '''아래 글의 문장 사이 관계를 JSON으로만 답하라.
관계: supports(근거), example_of(예시), contrasts(대조), addresses(문항에 답함, 대상 Q).
문단 사이: continues(이어짐), shifts_topic(주제 전환).
각 문장은 최대 하나의 supports 또는 example_of만 가진다. 없으면 비워 둔다.
글에 쓰인 것만 표시하고, 없는 관계를 추측하지 마라. 문장 ID는 주어진 그대로 쓴다.'''
PROMPT_V3 = PROMPT + '''
off_topic에는 문항과 무관한 문장 ID만 넣어라. 문항에 직접 답하거나 관련 주장에 근거·예시·설명을 주면 관련 문장이다.
관계가 없거나 Q로 이어지는 경로가 없다는 이유만으로 무관하다고 표시하지 마라. 불확실하면 off_topic에 넣지 마라.'''


def request(document, question, *, version=2):
    paragraphs, smap, pmap = public_input(document)
    def edges(sources, targets, labels):
        return array_schema(object_schema({'source': enum(sources), 'target': enum(targets), 'label': enum(labels)}))
    schema = object_schema({
        'sentence_edges': edges(smap.values(), [*smap.values(), 'Q'], ['supports', 'example_of', 'contrasts', 'addresses']),
        'paragraph_edges': edges(pmap.values(), pmap.values(), ['continues', 'shifts_topic'])})
    if version == 3:
        schema['properties']['off_topic'] = array_schema(enum(smap.values()))
        schema['required'].append('off_topic')
    return [{'role': 'system', 'content': PROMPT_V3 if version == 3 else PROMPT}, {'role': 'user', 'content': json.dumps(
        {'Q': question, 'paragraphs': paragraphs}, ensure_ascii=False)}], schema


def validate(value, sids, pids, *, check_degree=True, allow_protection=False):
    fields = set(value)
    if allow_protection and 'relevance_protection' in fields:
        fields = fields - {'relevance_protection'}
        if 'off_topic' not in fields:
            raise ValueError('Relevance protection requires explicit off_topic IDs')
    if fields not in ({'sentence_edges', 'paragraph_edges'}, {'sentence_edges', 'paragraph_edges', 'off_topic'}):
        raise ValueError('Invalid graph fields')
    if 'off_topic' in value:
        marked = value['off_topic']
        if (not isinstance(marked, list) or any(not isinstance(sid, str) for sid in marked)
                or len(set(marked)) != len(marked) or not set(marked) <= set(sids)):
            raise ValueError('Invalid explicit off_topic IDs')
    if allow_protection and 'relevance_protection' in value:
        _validate_protection(value['relevance_protection'], sids, value['off_topic'])
    for key, ids, labels in [('sentence_edges', set(sids), {'supports', 'example_of', 'contrasts', 'addresses'}),
                              ('paragraph_edges', set(pids), {'continues', 'shifts_topic'})]:
        seen, outgoing = set(), Counter()
        for e in value[key]:
            triple = (e['source'], e['target'], e['label'])
            valid_target = e['target'] == 'Q' if e['label'] == 'addresses' else e['target'] in ids
            if (set(e) != {'source', 'target', 'label'} or e['source'] not in ids or
                    not valid_target or e['label'] not in labels or e['source'] == e['target'] or triple in seen):
                raise ValueError('Invalid graph edge')
            seen.add(triple)
            if e['label'] in {'supports', 'example_of'}:
                outgoing[e['source']] += 1
        if check_degree and any(n > 1 for n in outgoing.values()):
            raise ValueError('Multiple outgoing support/example edges')
    return deepcopy(value)


def _validate_protection(value, sids, off_topic):
    if not isinstance(value, dict) or set(value) != {
            'seeds', 'protected_ids', 'depths', 'overridden_off_topic'}:
        raise ValueError('Invalid relevance protection fields')
    ids = set(sids)
    for key in ('seeds', 'protected_ids', 'overridden_off_topic'):
        values = value[key]
        if (not isinstance(values, list) or any(not isinstance(sid, str) for sid in values)
                or len(set(values)) != len(values) or not set(values) <= ids):
            raise ValueError('Invalid relevance protection IDs')
    seeds, protected = set(value['seeds']), set(value['protected_ids'])
    depths = value['depths']
    if (not seeds <= protected or bool(seeds) != bool(protected)
            or not isinstance(depths, dict) or set(depths) != protected
            or any(type(depth) is not int or not 0 <= depth <= 2 for depth in depths.values())
            or {sid for sid, depth in depths.items() if depth == 0} != seeds
            or set(value['overridden_off_topic']) != protected & set(off_topic)):
        raise ValueError('Invalid relevance protection depths or overrides')


def relevance_protection(left, right, sids):
    """Derive a conservative flag exclusion; this does not assert relevance.

    Sentence edges point from a supporting/example/contrast child to its target.
    Walk the reverse direction from either extraction's Q-addressing seeds for
    at most two hops. Union edges are used only here, never as graph assertions.
    Raw extraction validation is separate; malformed/foreign edges cannot grant
    protection even if this helper is used while diagnosing an invalid response.
    """
    ids, edges = set(sids), set()
    for extraction in (left, right):
        for edge in extraction.get('sentence_edges', []):
            if not isinstance(edge, dict) or set(edge) != {'source', 'target', 'label'}:
                continue
            source, target, label = edge['source'], edge['target'], edge['label']
            if not all(isinstance(part, str) for part in (source, target, label)):
                continue
            if source not in ids or source == target:
                continue
            if label == 'addresses' and target == 'Q':
                edges.add((source, target, label))
            elif label in {'supports', 'example_of', 'contrasts'} and target in ids:
                edges.add((source, target, label))
    seeds = {source for source, target, label in edges if label == 'addresses'}
    depths = {sid: 0 for sid in seeds}
    for depth in (1, 2):
        parents = {sid for sid, found_depth in depths.items() if found_depth == depth - 1}
        children = {source for source, target, label in edges
                    if target in parents and label in {'supports', 'example_of', 'contrasts'}}
        for sid in children - depths.keys():
            depths[sid] = depth
    protected = set(depths)
    explicit = set(left.get('off_topic', [])) & set(right.get('off_topic', []))
    return {'seeds': sorted(seeds), 'protected_ids': sorted(protected),
            'depths': {sid: depths[sid] for sid in sorted(depths)},
            'overridden_off_topic': sorted(protected & explicit)}


def intersection(left, right):
    def triples(edges):
        return {(e['source'], e['target'], e['label']) for e in edges}
    kept, counts = {}, {}
    for key in ('sentence_edges', 'paragraph_edges'):
        a, b = triples(left[key]), triples(right[key])
        kept[key] = [dict(source=s, target=t, label=l) for s, t, l in sorted(a & b)]
        counts[key] = {'left': len(a), 'right': len(b), 'intersection': len(a & b), 'union': len(a | b)}
    if 'off_topic' in left or 'off_topic' in right:
        if 'off_topic' not in left or 'off_topic' not in right:
            raise ValueError('Both extractions must explicitly include off_topic')
        a, b = set(left['off_topic']), set(right['off_topic'])
        kept['off_topic'] = sorted(a & b)
        counts['off_topic'] = {'left': len(a), 'right': len(b), 'intersection': len(a & b), 'union': len(a | b)}
    return kept, counts


def apply_protection(graph_state, discourse):
    """Mutate and return an existing state's explicit flags/protection view only.

    Sentence/paragraph assertions and Bareun facts stay unchanged, so previously
    extracted states can receive this deterministic rule without another call.
    Protection stays frozen to original IDs; only surviving IDs enter the view.
    """
    if 'off_topic' not in discourse:
        raise ValueError('Relevance protection requires explicit off_topic IDs')
    ids = {sentence['id'] for sentence in graph_state['sentences']}
    candidates = ids & set(discourse['off_topic'])
    graph_state['off_topic_candidate'] = sorted(candidates)
    graph_state['off_topic_rule'] = 'explicit_agreement'
    graph_state.pop('relevance_protection', None)
    if 'relevance_protection' in discourse:
        protection = discourse['relevance_protection']
        protected = ids & set(protection['protected_ids'])
        graph_state['off_topic_candidate'] = sorted(candidates - protected)
        graph_state['off_topic_rule'] = 'explicit_agreement_with_relevance_protection'
        graph_state['relevance_protection'] = {
            'seeds': sorted(ids & set(protection['seeds'])), 'protected_ids': sorted(protected),
            'depths': {sid: protection['depths'][sid] for sid in sorted(protected)},
            'overridden_off_topic': sorted(candidates & protected)}
    return graph_state


def state(structure, discourse, pids):
    # Reuse Bareun marker construction; Q edges are handled separately.
    plain = {**discourse, 'roles': [{'id': a.sid, 'role': 'other'} for a in structure.annotations],
             'sentence_edges': [e for e in discourse['sentence_edges'] if e['target'] != 'Q']}
    result = build(structure, plain, paragraph_ids=pids)
    ids = {a.sid for a in structure.annotations}
    result['sentence_edges'] += [deepcopy(e) for e in discourse['sentence_edges']
                                 if e['target'] == 'Q' and e['source'] in ids]
    reachable = {'Q'}
    while True:
        grown = reachable | {e['source'] for e in result['sentence_edges'] if e['target'] in reachable}
        if grown == reachable:
            break
        reachable = grown
    if 'off_topic' in discourse:
        apply_protection(result, discourse)
    else:
        result['off_topic_candidate'] = sorted(ids - reachable)
    # No claim-role classifier is introduced: these are Q-addressing nodes with no
    # incoming support/example edge, not semantic claims of missing evidence.
    supported = {e['target'] for e in result['sentence_edges'] if e['label'] in {'supports', 'example_of'}}
    addressed = {e['source'] for e in result['sentence_edges'] if e['label'] == 'addresses'}
    result['unsupported'] = sorted(addressed - supported)
    return result


def query(graph, target):
    if target in {'unsupported', 'off_topic'}:
        key = 'off_topic_candidate' if target == 'off_topic' else target
        if target == 'off_topic' and graph.get('off_topic_rule') == 'explicit_agreement_with_relevance_protection':
            note = '두 독립 추출 모두 무관하다고 명시한 후보에서 Q 응답 문장 및 합집합 관계 2단계 보호 문장을 제외함. 보호는 관련성 판정이 아님'
        else:
            note = ('두 독립 추출 모두 문항과 무관하다고 명시한 후보' if target == 'off_topic' and
                    graph.get('off_topic_rule') == 'explicit_agreement' else '추출 관계의 유무만 표시하며 의미 오류 판정이 아님')
        return {key: graph[key], 'note': note}
    nodes = [m for m in graph['sentences'] if m['id'] == target or m['paragraph'] == target]
    if not nodes and target not in graph['paragraphs']:
        raise ValueError('Unknown graph ID')
    ids = {n['id'] for n in nodes} | {target}
    return {'nodes': nodes, 'edges': [e for k in ('sentence_edges', 'paragraph_edges') for e in graph[k]
                                      if e['source'] in ids or e['target'] in ids],
            'dangling': [e for e in graph['dangling'] if e['source'] in ids]}


def audit(graph, initial, *, version=2):
    old = {m['id']: m for m in initial['sentences']}
    changes = [{'sid': m['id'], 'before': old[m['id']]['predecessor'], 'after': m['predecessor'],
                'conjunction': m['conjunction'], 'subject_omitted': m['subject_omitted']}
               for m in graph['sentences'] if m['id'] in old and
               m['predecessor'] != old[m['id']]['predecessor'] and
               (m['conjunction'] or m['subject_omitted'] or old[m['id']]['conjunction'] or old[m['id']]['subject_omitted'])]
    result = {'off_register': [m['id'] for m in graph['sentences']
                             if graph['dominant_style'] not in {'unknown', 'mixed'} and
                             m['register'] not in {'unknown', 'mixed', graph['dominant_style']}],
            'predecessor_changes': changes, 'off_topic_candidate': graph['off_topic_candidate'],
            'unsupported': graph['unsupported'], 'dangling': graph['dangling']}
    if version == 3:
        del result['unsupported']
    return result
