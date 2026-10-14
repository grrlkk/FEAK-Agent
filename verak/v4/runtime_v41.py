"""v4.1 composition over the unchanged Dv3 environment and procedure.

Function bindings are private copies of globals; the original Dv3 code objects,
module globals, observations, tool validators, step limits and export masks are
not modified. The bindings change only output roots and prompt identities.
"""
import json
from types import FunctionType

from verak.v3.eval.api import Phase6API
from .paid import PrepAPI
from .policy_env import V4Environment
from . import policy_prompts_v41 as prompts


class V41Environment(V4Environment):
    def __init__(self,row,analysis,*,prompt_version=prompts.VERSION):
        prompts.verify_frozen(version=prompt_version)
        self.prompt_version=prompt_version
        super().__init__(row,analysis)

    def public_messages(self,role,steps_left=None):
        messages=super().public_messages(role,steps_left)
        messages[0]={'role':'system','content':prompts.prompt_for(role,version=self.prompt_version)}
        prompts.assert_messages(messages,role,version=self.prompt_version)
        return messages


def bind(function,**replacements):
    if any(key not in function.__globals__ for key in replacements):
        raise ValueError('Unknown frozen-procedure binding')
    value=FunctionType(function.__code__,{**function.__globals__,**replacements},function.__name__,
                       function.__defaults__,function.__closure__)
    value.__kwdefaults__=function.__kwdefaults__
    return value


def make_plan(row,api,*,output_root):
    from .prep3_content import make_plan as original
    return bind(original,ROOT=output_root)(row,api)


def teach(row,plan,attempt,api,tokenizer,*,output_root,prompt_version=prompts.VERSION):
    from .prep3_content import teach as original
    prompts.verify_frozen(version=prompt_version)
    result=bind(original,ROOT=output_root,V4Environment=V41Environment,
                assert_messages=prompts.assert_messages,verify_frozen=prompts.verify_frozen)(row,plan,attempt,api,tokenizer)
    if result.get('prompt_version')!=prompt_version:
        raise ValueError('Cached teacher attempt has the wrong prompt version')
    for call in result.get('calls',[]):
        prompts.assert_messages(call['public_messages'],call['role'],version=prompt_version)
    return result


def export(cases,tokenizer,*,output_root,prompt_version=prompts.VERSION):
    from .prep3_content_report import export as original
    prompts.verify_frozen(version=prompt_version)
    return bind(original,ROOT=output_root,assert_messages=prompts.assert_messages,
                verify_frozen=prompts.verify_frozen)(cases,tokenizer)


class VersionedAPI(PrepAPI):
    """Same durable reservations/retries, with a v4.1-specific teacher assertion."""
    def request(self,messages,*,stage,item_id,effort='low',max_output=2048,schema=None):
        from .common import sha_text
        prompts.verify_frozen()
        if 'teacher' in stage:
            prompts.assert_editor_request(messages)
        stage='v41_'+stage
        contract={'stage':stage,'item_id':item_id,'model':self.model,'reasoning_effort':effort,
                  'max_output_tokens':max_output,'messages':messages,'schema':schema}
        fingerprint=sha_text(json.dumps(contract,ensure_ascii=False,sort_keys=True))
        with self.db() as db:
            prior=db.execute("SELECT id,status FROM calls WHERE fingerprint=? AND status!='blocked_before_send' ORDER BY id",
                             (fingerprint,)).fetchall()
        if prior and not any(status=='completed' for _,status in prior):
            raise RuntimeError(f'Preserved prior v4.1 API outcome {prior}; no silent regeneration')
        return Phase6API.request(self,messages,stage=stage,item_id=item_id,effort=effort,max_output=max_output,schema=schema)
