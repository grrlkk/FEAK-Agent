"""Versioned v4.1 editor prompts; the frozen v4 bytes remain untouched."""
from hashlib import sha256
import json
from pathlib import Path

from . import policy_prompts as v4

VERSION = 'v4.1_editors_20261010'
MASK_RULE = '#@...# 같은 가림 표시는 그대로 둔다.'
KOREAN_EDIT_RULE = 'EDIT는 문장 하나 안에서만 고치며 문장을 나누거나 합치지 않는다.'


def _extend(text, extra):
    text=text.replace('행동 JSON 하나만 출력한다.','행동 JSON 객체 하나만 출력한다.')
    first,rest=text.split('\n',1)
    return first+'\n'+extra+'\n'+rest


REVISION = _extend(v4.REVISION, MASK_RULE)
KOREAN = _extend(v4.KOREAN, MASK_RULE+'\n'+KOREAN_EDIT_RULE)
PROMPTS={'revision':REVISION,'korean':KOREAN}
FROZEN=Path(__file__).with_name('prompts')/'v41_editors.json'
EXPECTED_HASHES={
    'revision':'af13e45ebc23f0e4a7a8269a9687c2efa68130a87ece300d5a4ba282e13dbb98',
    'korean':'d8bc2086fc316fcd737abaa84f7e505f3c631d3087f55f6bd80526db8d8abd24',
}


def digest(text):
    return sha256(text.encode('utf-8')).hexdigest()


def freeze(tokenizer,tokenizer_path):
    """Local, no-model-inference prompt freeze; refuse replacement of any version."""
    from .common import atomic_new,file_sha
    v4.verify_frozen()
    value={'version':VERSION,'policy_version':'v4.1','tokenizer':str(tokenizer_path),
        'tokenizer_sha256':file_sha(Path(tokenizer_path)/'tokenizer.json'),
        'base_manifest_sha256':file_sha(v4.FROZEN),
        'token_definition':'pinned Kanana tokenizer; body without special tokens and system-message template both <=400',
        'roles':{role:{'text':text,'sha256':digest(text),
            'policy_tokens':len(tokenizer.encode(text,add_special_tokens=False)),
            'system_message_tokens':len(tokenizer.apply_chat_template([{'role':'system','content':text}],tokenize=True))}
            for role,text in PROMPTS.items()},'teacher_export_inference_exact_match':True,'legacy_v4_unchanged':True}
    if any(r['policy_tokens']>400 or r['system_message_tokens']>400 for r in value['roles'].values()):
        raise ValueError('v4.1 prompt token budget exceeded; no artifact written')
    if FROZEN.exists():
        if json.loads(FROZEN.read_text())!=value:
            raise ValueError('Refuse to overwrite a frozen v4.1 version')
    else:
        atomic_new(FROZEN,value)
    return verify_frozen()


def verify_frozen(*,version=VERSION):
    v4.verify_frozen()
    if version!=VERSION:
        raise ValueError('An explicit v4.1 editor version is required')
    manifest=json.loads(FROZEN.read_text())
    if manifest['version']!=VERSION or manifest['policy_version']!='v4.1':
        raise ValueError('Wrong frozen v4.1 prompt version')
    if manifest['base_manifest_sha256']!=sha256(v4.FROZEN.read_bytes()).hexdigest():
        raise ValueError('Frozen v4 base prompt artifact changed')
    for role,text in PROMPTS.items():
        saved=manifest['roles'][role]
        if saved['text']!=text or saved['sha256']!=digest(text) or digest(text)!=EXPECTED_HASHES[role]:
            raise ValueError('Frozen v4.1 prompt identity changed: '+role)
        if not (0<saved['policy_tokens']<=400 and 0<saved['system_message_tokens']<=400):
            raise ValueError('Frozen v4.1 prompt exceeds the 400-token limit')
        if MASK_RULE not in text or 'JSON 객체 하나만 출력한다' not in text:
            raise ValueError('Required v4.1 preservation/JSON instruction missing')
    if KOREAN_EDIT_RULE not in KOREAN:
        raise ValueError('Required Korean EDIT instruction missing')
    return manifest


def prompt_for(role,*,version=VERSION):
    verify_frozen(version=version)
    return PROMPTS[role]


def assert_messages(messages,role,*,version=VERSION):
    if not messages or messages[0]!={'role':'system','content':prompt_for(role,version=version)}:
        raise ValueError('Teacher/export/inference must use the exact frozen v4.1 prompt')


def assert_editor_request(messages,*,version=VERSION):
    verify_frozen(version=version)
    for role,text in PROMPTS.items():
        if messages and messages[0]=={'role':'system','content':text}:
            return role
    raise ValueError('v4.1 teacher requests must use the exact frozen v4.1 prompt')
