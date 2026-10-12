"""v4.6: permit ONLY a missing final terminal mark; v4.5 otherwise unchanged."""
from .environment_v43 import terminal_positions
from .environment_v45 import V45Environment, WhitespaceExecutor, match_span, CONTEXT, OUTPUT, REJECT_LIMIT

VERSION = 'v4.6_terminal_only_20261012'


def terminal_only(before, after):
    return (bool(before.strip()) and not terminal_positions(before)
            and len(after) == len(before) + 1 and after[:-1] == before
            and after[-1] in '.?!')


class TerminalExecutor(WhitespaceExecutor):
    def edit(self, document, args, role):
        target = args['target']
        allow = False
        if self.editor_role == 'korean' and ':' in target and not target.startswith(('before:', 'after:')):
            public, _, old = target.partition(':')
            unit = document.locate(self.sid(public, document))[2]
            span, _ = match_span(unit.text, old)
            if span is not None:
                start, end = span
                after = unit.text[:start] + args['new_text'] + unit.text[end:]
                allow = terminal_only(unit.text, after)
        # The accepted exception produces exactly one sentence. Reuse every
        # existing mask/alias/analysis/undo check, relaxing only Korean's added-
        # terminal guard for this exact operation, then restore the role.
        original_role = self.editor_role
        try:
            if allow:
                self.editor_role = 'revision'
            return super().edit(document, args, role)
        finally:
            self.editor_role = original_role


class V46Environment(V45Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = TerminalExecutor(self.analysis, self.executor.sentences, self.executor.paragraphs)
