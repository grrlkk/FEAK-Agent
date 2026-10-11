"""Inference-only v4.5 span matching; all frozen prompts and v4.4 guards remain."""
from copy import deepcopy
import re

from verak.v3.env.protocol import ActionError
from .environment_v43 import BoundaryExecutor
from .environment_v44 import V44Environment, OUTPUT, REJECT_LIMIT

VERSION = 'v4.5_inference_20261012'
CONTEXT = {'revision': 8192, 'korean': 16384, 'orchestrator': 16384}
SPAN_POLICY = ('Exact unique substring first (including overlapping occurrences); otherwise '
               'collapse each Unicode whitespace run to one ASCII space in both strings. '
               'Retain boundary whitespace; require one normalized occurrence. Map its '
               'offsets to the original string; change only that span. No fuzzy matching, '
               'case/punctuation normalization, or removal of word spaces.')


def occurrences(text, old):
    if not old:
        return []
    return [m.start() for m in re.finditer('(?='+re.escape(old)+')', text)]


def whitespace_view(text):
    """Return normalized characters and their lossless original offset ranges."""
    tokens = list(re.finditer(r'\s+|\S', text))
    return (''.join(' ' if m.group().isspace() else m.group() for m in tokens),
            [m.span() for m in tokens])


def match_span(text, old):
    exact = occurrences(text, old)
    if len(exact) == 1:
        start = exact[0]
        return (start, start+len(old)), {'mode': 'exact', 'exact_matches': 1}
    normalized, offsets = whitespace_view(text)
    target, _ = whitespace_view(old)
    matches = occurrences(normalized, target) if target.strip() else []
    diagnostic = {'mode': 'whitespace_normalized', 'exact_matches': len(exact),
                  'normalized_matches': len(matches), 'matched': len(matches) == 1}
    if len(matches) != 1:
        return None, diagnostic
    index = matches[0]
    span = (offsets[index][0], offsets[index+len(target)-1][1])
    diagnostic['original_span'] = list(span)
    return span, diagnostic


class WhitespaceExecutor(BoundaryExecutor):
    last_span_match = None

    def edit(self, document, args, role):
        target = args['target']
        if target.startswith(('before:', 'after:')) or ':' not in target:
            return super().edit(document, args, role)
        public, _, old = target.partition(':')
        sid = self.sid(public, document)
        unit = document.locate(sid)[2]
        span, self.last_span_match = match_span(unit.text, old)
        if span is None:
            raise ActionError('substring', '부분 문자열이 해당 문장에 정확히 한 번 있어야 합니다.')
        # Only the executor receives the actual span. The model's raw action and
        # journal target remain intact. The original mask/boundary/undo checks run.
        start, end = span
        return super().edit(document, {**args, 'target': public+':'+unit.text[start:end]}, role)


class V45Environment(V44Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = WhitespaceExecutor(self.analysis, self.executor.sentences, self.executor.paragraphs)

    def step(self, raw, role):
        self.executor.last_span_match = None
        record = super().step(raw, role)
        if self.executor.last_span_match is not None:
            record['span_match'] = deepcopy(self.executor.last_span_match)
        return record
