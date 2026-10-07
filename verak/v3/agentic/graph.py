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


def validate(value, sids, pids, *, check_degree=True):
    if set(value) not in ({'sentence_edges', 'paragraph_edges'}, {'sentence_edges', 'paragraph_edges', 'off_topic'}):
        raise ValueError('Invalid graph fields')
    if 'off_topic' in value:
        marked = value['off_topic']
        if not isinstance(marked, list) or len(set(marked)) != len(marked) or not set(marked) <= set(sids):
            raise ValueError('Invalid explicit off_topic IDs')
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
    result['off_topic_candidate'] = (sorted(ids & set(discourse['off_topic'])) if 'off_topic' in discourse
                                      else sorted(ids - reachable))
    if 'off_topic' in discourse:
        result['off_topic_rule'] = 'explicit_agreement'
    # No claim-role classifier is introduced: these are Q-addressing nodes with no
    # incoming support/example edge, not semantic claims of missing evidence.
    supported = {e['target'] for e in result['sentence_edges'] if e['label'] in {'supports', 'example_of'}}
    addressed = {e['source'] for e in result['sentence_edges'] if e['label'] == 'addresses'}
    result['unsupported'] = sorted(addressed - supported)
    return result


def query(graph, target):
    if target in {'unsupported', 'off_topic'}:
        key = 'off_topic_candidate' if target == 'off_topic' else target
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
