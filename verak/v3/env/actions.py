"""Transactional surface edits and moves over stable document units."""
from difflib import SequenceMatcher
from collections import Counter
import re

from ..corrupt.document import MARKER, Unit, protected
from .protocol import ActionError


def check_markers(before, after):
    if Counter(MARKER.findall(before)) != Counter(MARKER.findall(after)):
        raise ActionError('anonymization', '기존 익명화 표지는 그대로 유지해야 합니다.')
    for op, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if op != 'equal' and protected(before, a, b):
            raise ActionError('anonymization', '익명화 표지 내부는 편집할 수 없습니다.')


class ActionExecutor:
    def __init__(self, analysis, sentence_aliases, paragraph_aliases):
        self.analysis = analysis
        self.sentences = sentence_aliases
        self.paragraphs = paragraph_aliases
        self.serial = 0
        self.split_serial = {}

    def sid(self, public, document):
        internal = next((key for key, value in self.sentences.items() if value == public), None)
        if internal is None or all(u.sid != internal for u in document.units):
            raise ActionError('unknown_id', f'현재 문장 ID가 없습니다: {public}')
        return internal

    def pid(self, public, document):
        internal = next((key for key, value in self.paragraphs.items() if value == public), None)
        if internal is None or all(p.pid != internal for p in document.paragraphs):
            raise ActionError('unknown_id', f'현재 문단 ID가 없습니다: {public}')
        return internal

    def fresh(self, parent=None):
        if parent is None:
            while True:
                self.serial += 1
                sid = f'N{self.serial}'
                if sid not in self.sentences and sid not in self.sentences.values():
                    break
            self.sentences[sid] = sid
        else:
            self.split_serial[parent] = self.split_serial.get(parent, 0) + 1
            i = self.split_serial[parent]
            suffix = ''
            while i:
                i, r = divmod(i-1, 26)
                suffix = chr(97+r) + suffix
            sid = parent + suffix
            while sid in self.sentences:
                return self.fresh(parent)
            self.sentences[sid] = self.sentences[parent] + suffix
        return sid

    def edit(self, document, args, role):
        target, new_text = args['target'], args['new_text']
        insertion = target.startswith(('before:', 'after:'))
        if role == 'global' and not (insertion or ':' not in target and new_text == ''):
            raise ActionError('role_forbidden', 'GLOBAL은 문장 전체 삽입·삭제만 EDIT할 수 있습니다.')
        if role == 'korean' and (insertion or ':' not in target and not new_text.strip()):
            raise ActionError('role_forbidden', 'KOREAN은 기존 문장 내부만 EDIT할 수 있습니다.')
        if insertion:
            position, anchor = target.split(':', 1)
            sid = self.sid(anchor, document)
            pi, si, old = document.locate(sid)
            pieces = self.analysis.pieces(new_text)
            added = [Unit(self.fresh(), text, tokens, leading or ' ') for text, tokens, leading in pieces]
            at = si + (position == 'after')
            if at == 0:
                added[0].leading = old.leading
                old.leading = old.leading or ' '
            document.paragraphs[pi].units[at:at] = added
            return {u.sid for u in added} | {sid}, {document.paragraphs[pi].pid}
        public, separator, substring = target.partition(':')
        sid = self.sid(public, document)
        pi, si, old = document.locate(sid)
        paragraph = document.paragraphs[pi]
        if separator:
            if not substring or old.text.count(substring) != 1:
                raise ActionError('substring', '부분 문자열이 해당 문장에 정확히 한 번 있어야 합니다.')
            start = old.text.index(substring)
            if protected(old.text, start, start+len(substring)):
                raise ActionError('anonymization', '익명화 표지 내부는 편집할 수 없습니다.')
            replacement = old.text[:start] + new_text + old.text[start+len(substring):]
        else:
            replacement = new_text
        if not replacement.strip():
            if role == 'korean':
                raise ActionError('role_forbidden', 'KOREAN은 문장을 삭제할 수 없습니다.')
            paragraph.units.pop(si)
            return {sid}, {paragraph.pid}
        check_markers(old.text, replacement)
        pieces = self.analysis.pieces(replacement)
        if role == 'korean' and len(pieces) != 1:
            raise ActionError('role_forbidden', 'KOREAN은 문장 수를 늘릴 수 없습니다.')
        rewritten = [Unit(sid if i == 0 else self.fresh(sid), text, tokens,
                          old.leading if i == 0 else leading or ' ')
                     for i, (text, tokens, leading) in enumerate(pieces)]
        paragraph.units[si:si+1] = rewritten
        return {u.sid for u in rewritten}, {paragraph.pid}

    def move(self, document, args, role):
        if role == 'korean':
            raise ActionError('role_forbidden', 'KOREAN은 MOVE를 사용할 수 없습니다.')
        target, position = args['target'], args['position']
        side, colon, anchor = position.partition(':')
        if not colon or side not in ('before', 'after'):
            raise ActionError('position', 'position은 before:ID 또는 after:ID여야 합니다.')
        if target.startswith('P'):
            pid = self.pid(target, document)
            if not anchor.startswith('P'):
                raise ActionError('position', '문단 MOVE의 기준은 문단 ID여야 합니다.')
            destination = self.pid(anchor, document)
            if pid == destination:
                raise ActionError('move_self', '문단을 자기 자신으로 옮길 수 없습니다.')
            paragraph = next(p for p in document.paragraphs if p.pid == pid)
            document.paragraphs.remove(paragraph)
            index = next(i for i, p in enumerate(document.paragraphs) if p.pid == destination)
            document.paragraphs.insert(index + (side == 'after'), paragraph)
            return {u.sid for u in document.units}, {pid, destination}
        ends = target.split('-')
        if len(ends) > 2:
            raise ActionError('target', '문장 ID 또는 연속 범위가 필요합니다.')
        first = self.sid(ends[0], document)
        last = self.sid(ends[-1], document)
        pi, si, _ = document.locate(first)
        pj, sj, _ = document.locate(last)
        if pi != pj or si > sj:
            raise ActionError('range', '범위는 같은 문단에서 현재 순서상 연속이어야 합니다.')
        source = document.paragraphs[pi]
        moved = source.units[si:sj+1]
        if anchor.startswith('P'):
            pid = self.pid(anchor, document)
            dest = next(p for p in document.paragraphs if p.pid == pid)
            at_sid = None
        else:
            at_sid = self.sid(anchor, document)
            if at_sid in {u.sid for u in moved}:
                raise ActionError('move_self', '이동 대상 안을 목적지로 지정할 수 없습니다.')
            qi, _, _ = document.locate(at_sid)
            dest = document.paragraphs[qi]
        del source.units[si:sj+1]
        if at_sid is None:
            index = 0 if side == 'before' else len(dest.units)
        else:
            index = next(i for i, u in enumerate(dest.units) if u.sid == at_sid) + (side == 'after')
        for u in moved:
            u.leading = u.leading or ' '
        dest.units[index:index] = moved
        return {u.sid for u in source.units + dest.units + moved}, {source.pid, dest.pid}
