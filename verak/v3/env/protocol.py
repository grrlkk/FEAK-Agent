"""Strict action parsing, separate from executable-action validation."""
import json


class ActionParseError(ValueError):
    pass


class ActionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def parse_action(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ActionParseError('JSON 키 중복: ' + key)
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ActionParseError('유한 JSON 값 필요')))
    except (ValueError, TypeError) as error:
        raise ActionParseError('JSON 객체 하나가 필요합니다: ' + str(error)) from None
    if not isinstance(value, dict):
        raise ActionParseError('JSON 최상위 값은 객체여야 합니다.')
    return value


def validate_action(value, *, allow_check=True):
    if set(value) != {'thought', 'action', 'args'}:
        raise ActionError('action_schema', 'thought, action, args만 필요합니다.')
    if not isinstance(value['thought'], str) or len(value['thought']) > 80:
        raise ActionError('action_schema', 'thought는 80자 이하 문자열이어야 합니다.')
    action, args = value['action'], value['args']
    if action == 'CHECK' and not allow_check:
        raise ActionError('check_disabled', '현재 행동 공간은 EDIT/MOVE/UNDO/STOP이며 역할별 제한을 따릅니다.')
    allowed = ('EDIT', 'MOVE', 'CHECK', 'UNDO', 'STOP') if allow_check else ('EDIT', 'MOVE', 'UNDO', 'STOP')
    if action not in allowed or not isinstance(args, dict):
        raise ActionError('unknown_action', '/'.join(allowed)+'과 args 객체가 필요합니다.')
    keys = {'EDIT': {'target', 'new_text'}, 'MOVE': {'target', 'position'},
            'CHECK': set(), 'UNDO': set(), 'STOP': {'summary'}}[action]
    if set(args) != keys or any(not isinstance(v, str) for v in args.values()):
        raise ActionError('action_schema', f'{action} args: {sorted(keys)} 문자열 필드가 필요합니다.')
    return value
