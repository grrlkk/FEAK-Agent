"""Public-input-only A4 candidates. Rules suggest locations, never certify errors."""
from copy import deepcopy
from difflib import SequenceMatcher

from .common import read_json, atomic_new, sha_text
from .check_judge import ROOT

# Direction is taken from the frozen genre-neutral MAP contract. No reverse
# EXAMPLE / CAUSE label is invented when the closed conjunction set lacks it.
COMPATIBILITY = {
    ('contrast', 'either'): {'ADVERSATIVE'},
    ('example', 'current_to_previous'): {'EXAMPLE'},
    ('cause_effect', 'previous_to_current'): {'RESULT'},
    ('sequence', 'previous_to_current'): {'ADDITION'},
    ('support', 'current_to_previous'): {'ADDITION', 'EXAMPLE', 'RESTATEMENT'},
    ('support', 'previous_to_current'): {'RESULT', 'RESTATEMENT'},
}


def spacing_issues(text, response):
    """Accept only character-preserving refinements with verified source offsets."""
    issues = []
    for sentence in response.get('sentences', []):
        source = sentence.get('text', {})
        original, refined = source.get('content', ''), sentence.get('refined', '')
        offset = source.get('begin_offset', 0)
        if not original or not refined or text[offset:offset+len(original)] != original:
            continue
        if ''.join(original.split()) != ''.join(refined.split()):
            continue
        for tag, i, j, k, l in SequenceMatcher(None, original, refined, autojunk=False).get_opcodes():
            if tag != 'equal':
                issues.append({'start': offset+i, 'end': offset+j,
                    'original': original[i:j], 'suggestion': refined[k:l]})
    return issues


class LocalSpacing:
    """Local AnalyzeSyntax spacing refinement, independent of the frozen profile.

    This deployment has no CorrectError service. Its AnalyzeSyntax supports
    auto_spacing/auto_jointing; the base document analyzer remains unchanged.
    """
    def __init__(self, config, *, cache=ROOT/'spacing'):
        if config['bareun']['host'] not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('A4 spacing must remain on the local Bareun server')
        from verak.src.analyzer import BareunBackend
        cfg = config['bareun']
        self.backend = BareunBackend(cfg['host'], cfg.get('port', 5656), cfg.get('api_key_env', 'BAREUN_API_KEY'))
        self.cache = cache; self.calls = 0

    def __call__(self, text):
        path = self.cache/(sha_text('AnalyzeSyntax-spacing-v1\n'+text)+'.json')
        if path.exists():
            result = read_json(path)
            if result['text'] != text:
                raise ValueError('Spacing cache collision')
            return result
        from bareunpy._lang_service_client import pb, lpb
        from google.protobuf.json_format import MessageToDict
        request = pb.AnalyzeSyntaxRequest()
        request.document.content = text
        request.document.language = 'ko_KR'
        request.encoding_type = lpb.EncodingType.UTF32
        request.auto_split_sentence = True
        request.auto_spacing = True
        request.auto_jointing = True
        self.calls += 1
        result = {'text': text, 'local_host': '127.0.0.1:5656', 'available': False, 'issues': []}
        try:
            client = self.backend.tagger.client
            response = client.stub.AnalyzeSyntax(request, metadata=client.metadata, timeout=30)
            raw = MessageToDict(response, preserving_proto_field_name=True)
            result.update(available=True, issues=spacing_issues(text, raw), response=raw)
        except Exception as exc:
            # SDK wrappers can embed credentials in error text; never persist it.
            result['error_type'] = type(exc).__name__
            if callable(getattr(exc, 'code', None)):
                result['error_code'] = str(exc.code())
        atomic_new(path, result)
        return result


def compatible_categories(graph, sid, predecessor):
    allowed, edges = set(), []
    for edge in (graph or {}).get('sentence_relations', []):
        direction = ('current_to_previous' if (edge['source'], edge['target']) == (sid, predecessor)
            else 'previous_to_current' if (edge['source'], edge['target']) == (predecessor, sid) else None)
        if direction:
            values = COMPATIBILITY.get((edge['type'], direction), COMPATIBILITY.get((edge['type'], 'either'), set()))
            if values:
                allowed |= values; edges.append(deepcopy(edge))
    return allowed, edges


def leading_subject(annotation):
    """Literal repeated initial marked NP; no inferred coreference or gold text."""
    for evidence in annotation.subject_evidence:
        surface = evidence.get('surface', '')
        if evidence.get('entity_id') or not surface or '#@' in surface:
            continue
        start = evidence.get('span', [None])[0]
        if start is None:
            continue
        # Annotation offsets are document offsets; only an initial explicit NP.
        if start == annotation.start:
            return surface
    return None


def candidates(document, graph, spacing):
    structure = document.structure(); annotations = {a.sid: a for a in structure.annotations}
    units = {u.sid: u for u in document.units}
    result, unavailable = [], []
    def add(op, ann, evidence, instruction):
        scope = [ann.sid]+([ann.predecessor_id] if ann.predecessor_id in units else [])
        result.append({'candidate_id': f'C{len(result)+1}', 'operator': op, 'sid': ann.sid,
            'scope': scope, 'evidence': evidence, 'error_verdict': False,
            'item': {'location': scope, 'action': 'EDIT', 'needs_search': 'no',
                'instruction': '다음 관찰은 오류 확정이 아니다. 현재 문맥에서 필요한 부분만 고친다. '+instruction}})
    for ann in structure.annotations:
        if ann.style not in {'unknown', 'mixed', None} and structure.dominant_style not in {'unknown', 'mixed', None} and ann.style != structure.dominant_style:
            add('L_REGISTER', ann, {'style': ann.style, 'majority': structure.dominant_style},
                f'{ann.sid}의 종결체는 {ann.style}, 글의 다수 종결체는 {structure.dominant_style}이다. 문체 일관성을 확인한다.')
        if ann.initial_conj and ann.initial_conj['eligible']:
            allowed, edges = compatible_categories(graph, ann.sid, ann.predecessor_id)
            actual = ann.initial_conj['coarse_class']
            if allowed and actual not in allowed:
                add('L_CONJ', ann, {'conjunction': ann.initial_conj, 'compatible_categories': sorted(allowed), 'MAP_edges': edges},
                    f'{ann.sid}의 문두 접속사 {ann.initial_conj["form"]}({actual})와 앞 문장 사이 MAP 관계 {edges}를 확인한다.')
        before = annotations.get(ann.predecessor_id)
        subject = leading_subject(ann)
        if before and subject and subject == leading_subject(before):
            add('L_SUBJ_INSERT', ann, {'repeated_explicit_subject': subject, 'predecessor': before.sid},
                f'{before.sid}와 {ann.sid}의 문두에 같은 명시적 주어 {subject!r}가 연속한다. 불필요한 반복인지 확인한다.')
        evidence = spacing(units[ann.sid].text)
        if not evidence['available']:
            unavailable.append({'sid': ann.sid, 'error_type': evidence.get('error_type'), 'error_code': evidence.get('error_code')})
        elif evidence['issues']:
            add('L_SPACING', ann, {'suggestions': evidence['issues'], 'provider': 'local Bareun AnalyzeSyntax spacing refinement'},
                f'{ann.sid}의 로컬 바른 띄어쓰기 제안 {evidence["issues"]}을 검토한다.')
    return result, unavailable


if __name__ == '__main__':
    from .common import load_config, local_analysis_environment
    local_analysis_environment()
    checker = LocalSpacing(load_config(), cache=ROOT/'spacing_probe')
    value = checker('학생 은 학교에갈 수있다.')
    print({k: v for k, v in value.items() if k != 'response'})
    if not value['available']:
        raise SystemExit(1)
