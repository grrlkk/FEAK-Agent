"""Same Dv3 procedure with explicitly prepared v4.2 inputs and frozen contract."""
import json
from verak.v3.eval.api import Phase6API
from .paid import PrepAPI
from .runtime_v41 import bind
from .environment_v42 import V42Environment,BoundaryParagraphs,ENVIRONMENT_VERSION
from . import policy_prompts_v42 as prompts


def require_prepared(row):
    if row.get('environment_version')!=ENVIRONMENT_VERSION:
        raise ValueError('Call normalize_row and freeze its paragraphs before planner or teacher')


def make_plan(row,api,*,output_root):
    from .prep3_content import make_plan as original
    require_prepared(row);prompts.verify_frozen()
    return bind(original,ROOT=output_root)(row,api)


def teach(row,plan,attempt,api,tokenizer,*,output_root,prompt_version=prompts.VERSION):
    from .prep3_content import teach as original
    require_prepared(row);prompts.verify_frozen(version=prompt_version)
    result=bind(original,ROOT=output_root,V4Environment=V42Environment,BoostParagraphs=BoundaryParagraphs,
        assert_messages=prompts.assert_messages,verify_frozen=prompts.verify_frozen)(row,plan,attempt,api,tokenizer)
    if result.get('prompt_version')!=prompt_version: raise ValueError('Cached attempt has wrong v4.2 identity')
    for call in result.get('calls',[]): prompts.assert_messages(call['public_messages'],call['role'])
    return result


def export(cases,tokenizer,*,output_root,prompt_version=prompts.VERSION):
    from .prep3_content_report import export as original
    prompts.verify_frozen(version=prompt_version)
    return bind(original,ROOT=output_root,assert_messages=prompts.assert_messages,
        verify_frozen=prompts.verify_frozen)(cases,tokenizer)


class VersionedAPI(PrepAPI):
    def request(self,messages,*,stage,item_id,effort='low',max_output=2048,schema=None):
        from .common import sha_text
        prompts.verify_frozen()
        if 'teacher' in stage: prompts.assert_editor_request(messages)
        stage='v42_'+stage
        contract={'stage':stage,'item_id':item_id,'model':self.model,'reasoning_effort':effort,
            'max_output_tokens':max_output,'messages':messages,'schema':schema}
        fingerprint=sha_text(json.dumps(contract,ensure_ascii=False,sort_keys=True))
        with self.db() as db:
            prior=db.execute("SELECT id,status FROM calls WHERE fingerprint=? AND status!='blocked_before_send' ORDER BY id",(fingerprint,)).fetchall()
        if prior and not any(status=='completed' for _,status in prior):
            raise RuntimeError(f'Preserved v4.2 API outcome {prior}; no silent regeneration')
        return Phase6API.request(self,messages,stage=stage,item_id=item_id,effort=effort,max_output=max_output,schema=schema)
