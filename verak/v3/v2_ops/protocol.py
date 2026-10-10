"""Two opt-in GLOBAL actions; the original v1 validator is unchanged."""
from ..env.protocol import ActionError, ActionParseError, parse_action, validate_action as validate_v1


def validate_action(value, *, allow_check=False):
    if value.get('action') not in ('INSERT', 'SPLIT'):
        return validate_v1(value, allow_check=allow_check)
    if set(value) != {'thought', 'action', 'args'}:
        raise ActionError('action_schema', 'thought, action, args만 필요합니다.')
    if not isinstance(value['thought'], str) or len(value['thought']) > 80:
        raise ActionError('action_schema', 'thought는 80자 이하 문자열이어야 합니다.')
    fields = {'INSERT': {'position', 'text'}, 'SPLIT': {'sentence_id', 'text_1', 'text_2'}}[value['action']]
    args = value['args']
    if not isinstance(args, dict) or set(args) != fields or any(not isinstance(x, str) for x in args.values()):
        raise ActionError('action_schema', f'{value["action"]} args: {sorted(fields)} 문자열 필드가 필요합니다.')
    return value


def validity(raw, *, allow_check=False):
    try:
        value = parse_action(raw)
    except ActionParseError:
        return False, False
    try:
        validate_action(value, allow_check=allow_check)
    except (ActionError, TypeError):
        return True, False
    return True, True
