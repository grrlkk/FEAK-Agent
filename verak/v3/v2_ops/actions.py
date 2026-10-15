"""Mechanical INSERT/SPLIT execution with stable IDs and existing Bareun feedback."""
from ..corrupt.document import Unit
from ..env.actions import ActionExecutor, check_markers
from ..env.protocol import ActionError


class V2ActionExecutor(ActionExecutor):
    @staticmethod
    def _global(role):
        if role != 'global':
            raise ActionError('role_forbidden', 'INSERT와 SPLIT은 글 수정 에이전트(GLOBAL)만 사용할 수 있습니다.')

    def _sentence(self, text):
        pieces = self.analysis.pieces(text)
        if len(pieces) != 1:
            raise ActionError('sentence_count', '각 text는 정확히 한 문장이어야 합니다.')
        return pieces[0]

    def insert(self, document, args, role):
        self._global(role)
        side, separator, anchor = args['position'].partition(':')
        if not separator or side not in {'before', 'after'}:
            raise ActionError('position', 'position은 before:ID 또는 after:ID여야 합니다.')
        text, tokens, _ = self._sentence(args['text'])
        if anchor.startswith('P'):
            pid = self.pid(anchor, document)
            paragraph = next(p for p in document.paragraphs if p.pid == pid)
            index = 0 if side == 'before' else len(paragraph.units)
            affected = set()
        else:
            sid = self.sid(anchor, document)
            pi, si, _ = document.locate(sid)
            paragraph = document.paragraphs[pi]
            index = si + (side == 'after')
            affected = {sid}
        added = Unit(self.fresh(), text, tokens, ' ')
        if index == 0:
            added.leading = paragraph.units[0].leading if paragraph.units else ''
            if paragraph.units:
                paragraph.units[0].leading = paragraph.units[0].leading or ' '
        paragraph.units.insert(index, added)
        return affected | {added.sid}, {paragraph.pid}

    def split(self, document, args, role):
        self._global(role)
        sid = self.sid(args['sentence_id'], document)
        pi, si, old = document.locate(sid)
        first, second = self._sentence(args['text_1']), self._sentence(args['text_2'])
        check_markers(old.text, first[0] + ' ' + second[0])
        added = self.fresh(parent=sid)
        document.paragraphs[pi].units[si:si+1] = [Unit(sid, first[0], first[1], old.leading),
                                                Unit(added, second[0], second[1], ' ')]
        return {sid, added}, {document.paragraphs[pi].pid}
