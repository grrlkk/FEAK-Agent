"""Phase 7 addition: remove a closed-set prefix without changing the analyzer."""
import re

from .document import protected
from .operators import Proposal


def candidates(document):
    for ann in document.structure().annotations:
        conjunction = ann.initial_conj
        if ann.multi_unit or not conjunction or not conjunction['eligible']:
            continue
        pattern = r'\s*'.join(map(re.escape, conjunction['form'].split()))
        # Remove the prefix's optional comma and following spaces as one patch;
        # retain quote/parenthesis characters and the rest of the sentence exactly.
        match = re.match(r'[\s\"\'“‘(\[]*(' + pattern + r'(?=$|\s|[,，:;])[,，]?\s*)', ann.text)
        if not match or protected(ann.text, *match.span(1)) or not ann.text[match.end(1):].strip():
            continue
        yield Proposal('L_CONJ_DROP', [ann.sid], {
            'coarse_before': conjunction['coarse_class'], 'coarse_after': None,
            'conjunction': conjunction['form'], 'old': match[1], 'new': ''}, match.span(1), '')
