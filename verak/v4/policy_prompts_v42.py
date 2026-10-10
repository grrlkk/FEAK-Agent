"""v4.2 prompt/environment identity; prior frozen versions stay immutable."""
from hashlib import sha256
import json
from pathlib import Path

from . import policy_prompts_v41 as v41

VERSION='v4.2_editors_20261010'
REVISION_SPLIT_RULE='긴 문장은 EDIT로 두 문장까지 나눌 수 있다.'
REVISION=v41.REVISION.replace(v41.MASK_RULE,v41.MASK_RULE+'\n'+REVISION_SPLIT_RULE)
KOREAN=v41.KOREAN
PROMPTS={'revision':REVISION,'korean':KOREAN}
FROZEN=Path(__file__).with_name('prompts')/'v42_editors.json'
ENV_FILES=('environment_v42.py','runtime_v42.py')


def digest(text): return sha256(text.encode('utf-8')).hexdigest()


def environment_hashes():
    from .common import file_sha
    return {name:file_sha(Path(__file__).with_name(name)) for name in ENV_FILES}


def freeze(tokenizer,tokenizer_path):
    from .common import file_sha,atomic_new
    from .environment_v42 import ENVIRONMENT_VERSION,RULES
    v41.verify_frozen()
    value={'version':VERSION,'policy_version':'v4.2','environment_version':ENVIRONMENT_VERSION,
        'environment_sha256':environment_hashes(),'environment_rules':RULES,
        'tokenizer':str(tokenizer_path),'tokenizer_sha256':file_sha(Path(tokenizer_path)/'tokenizer.json'),
        'base_manifest_sha256':file_sha(v41.FROZEN),'roles':{role:{'text':text,'sha256':digest(text),
            'policy_tokens':len(tokenizer.encode(text,add_special_tokens=False)),
            'system_message_tokens':len(tokenizer.apply_chat_template([{'role':'system','content':text}],tokenize=True))}
            for role,text in PROMPTS.items()},'teacher_export_inference_exact_match':True}
    if any(r['policy_tokens']>400 or r['system_message_tokens']>400 for r in value['roles'].values()):
        raise ValueError('v4.2 prompt exceeds 400 tokens')
    if FROZEN.exists():
        if json.loads(FROZEN.read_text())!=value: raise ValueError('Frozen v4.2 contract differs')
    else: atomic_new(FROZEN,value)
    return verify_frozen()


def verify_frozen(*,version=VERSION):
    v41.verify_frozen()
    if version!=VERSION: raise ValueError('Explicit v4.2 version required')
    value=json.loads(FROZEN.read_text())
    if value['version']!=VERSION or value['base_manifest_sha256']!=sha256(v41.FROZEN.read_bytes()).hexdigest():
        raise ValueError('Frozen v4.2/base identity changed')
    if value['environment_sha256']!=environment_hashes(): raise ValueError('Frozen v4.2 environment changed')
    for role,text in PROMPTS.items():
        saved=value['roles'][role]
        if saved['text']!=text or saved['sha256']!=digest(text): raise ValueError('Frozen v4.2 prompt changed')
        if not (0<saved['policy_tokens']<=400 and 0<saved['system_message_tokens']<=400): raise ValueError('Prompt token limit')
    return value


def prompt_for(role,*,version=VERSION):
    verify_frozen(version=version);return PROMPTS[role]


def assert_messages(messages,role,*,version=VERSION):
    if not messages or messages[0]!={'role':'system','content':prompt_for(role,version=version)}:
        raise ValueError('Teacher/export/inference require exact frozen v4.2 prompt')


def assert_editor_request(messages,*,version=VERSION):
    verify_frozen(version=version)
    for role,text in PROMPTS.items():
        if messages and messages[0]=={'role':'system','content':text}: return role
    raise ValueError('Teacher requires exact frozen v4.2 prompt')
