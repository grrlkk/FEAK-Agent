"""Local Kanana reconstructions from the same context-only payload as Luna."""
import json
import time

from .common import file_sha, read_json, sha_text, write_json
from .reconstruction_data import reconstruction_payload
from .reconstruction_api import RECONSTRUCTION_PROMPT
from .phase2 import write_jsonl
from .reconstruction_format import first_sentence


def parse_sentence(raw):
    clean=raw.strip()
    if clean.startswith('```'):
        clean='\n'.join(clean.splitlines()[1:-1])
    try:
        result=json.loads(clean)
    except json.JSONDecodeError:
        # Preserve unwrapped output, then apply the same fixed first-sentence
        # extraction as Luna. Never regenerate or select by judged quality.
        if clean.startswith(('{','[')):
            raise
        result={'sentence':clean}
    if isinstance(result,str):
        result={'sentence':result}
    if not isinstance(result,dict) or not isinstance(result.get('sentence'),str):
        raise ValueError('Expected a sentence object or unwrapped sentence')
    sentence=result['sentence'].strip()
    return first_sentence(sentence)


def generate_local(cases,config,output,*,limit=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    rows={}
    path=config['paths']['policy_base']
    selected=cases[:limit] if limit else cases
    model=tokenizer=None
    settings=config['reconstruction']
    for i,case in enumerate(selected):
        payload=reconstruction_payload(case)
        messages=[{'role':'system','content':RECONSTRUCTION_PROMPT},
                  {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]
        seed=settings['seed']+int(sha_text(case['item_id'])[:8],16)%1000000
        contract={'messages':messages,'base':str(path.resolve()),'config_sha256':file_sha(path/'config.json'),
                  'temperature':.7,'top_p':.95,'top_k':50,'seed':seed,
                  'max_new_tokens':settings['local_max_new_tokens'],'dtype':'bfloat16'}
        fingerprint=sha_text(json.dumps(contract,ensure_ascii=False,sort_keys=True))
        cache=output/'local_reconstruction'/(sha_text(case['item_id'])+'.json')
        if cache.exists():
            result=read_json(cache)
            if result['fingerprint']!=fingerprint:
                raise ValueError('Local generation contract changed')
        else:
            if model is None:
                tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
                model=AutoModelForCausalLM.from_pretrained(str(path),torch_dtype=torch.bfloat16,
                    device_map={'':settings['local_gpu']},local_files_only=True,attn_implementation='sdpa')
                model.eval()
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            encoded=tokenizer(prompt,add_special_tokens=False,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]+settings['local_max_new_tokens']>model.config.max_position_embeddings:
                raise ValueError('Local reconstruction exceeds context; no truncation')
            encoded={k:v.to(model.device) for k,v in encoded.items()}
            set_seed(seed)
            start=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**encoded,do_sample=True,temperature=.7,top_p=.95,top_k=50,
                    max_new_tokens=settings['local_max_new_tokens'],pad_token_id=tokenizer.eos_token_id)
            raw=tokenizer.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
            result={'item_id':case['item_id'],'source':'kanana','model':'kakaocorp/kanana-1.5-8b-instruct-2505',
                    'fingerprint':fingerprint,'contract':contract,'raw':raw,
                    'elapsed_s':time.monotonic()-start,'input_tokens':encoded['input_ids'].shape[1],
                    'output_tokens':generated.shape[1]-encoded['input_ids'].shape[1]}
            # Persist every attempt before parsing. Never silently regenerate failures.
            write_json(cache,result)
        result['sentence']=parse_sentence(result['raw'])
        result['extraction']='first_complete_sentence_v1'
        write_json(cache,result)
        rows[case['item_id']]=result
        print(f'Local reconstruction {i+1}/{len(selected)}',flush=True)
    write_jsonl(output/'local_reconstructions.jsonl',list(rows.values()))
    return rows
