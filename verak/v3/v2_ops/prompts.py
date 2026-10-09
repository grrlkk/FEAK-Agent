"""The v1 KOREAN prompt is byte-identical; v2 adds two GLOBAL tool lines."""
from ..agent.runner import system_prompt as v1_prompt


def system_prompt(role, variant='default', *, allow_check=False):
    prompt = v1_prompt(role, variant, allow_check=allow_check)
    if role != 'global':
        return prompt
    prompt = prompt.replace('GLOBAL 에이전트다.', '글 수정 에이전트(GLOBAL)다.', 1)
    prompt = prompt.replace('"action":"EDIT|MOVE|UNDO|STOP"', '"action":"EDIT|MOVE|INSERT|SPLIT|UNDO|STOP"', 1)
    anchor = 'UNDO: {}'
    added = ('INSERT: {"position":"before:S7 또는 after:S7 또는 before:P2 또는 after:P2","text":"한 문장"}; '
        '문단 ID는 해당 문단의 처음/끝이다. 글에 없는 내용은 만들지 않는다.\n'
        'SPLIT: {"sentence_id":"S7","text_1":"첫 문장","text_2":"둘째 문장"}; '
        '내용을 보존하며 한 문장을 둘로 나누고 경계의 어미·접속어만 필요한 만큼 복원한다. '
        '이 문장 분리는 위 구조 수정 범위에 추가되며 UNDO로 취소할 수 있다.\n')
    return prompt.replace(anchor, added + anchor, 1)
