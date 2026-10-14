"""Bounded V5 API requests with durable reservations; secrets never enter logs."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from .common import ROOT, REPO, atomic_new, read_json, sha_text, write_json


class BudgetStop(RuntimeError):
    pass


def credentials():
    from dotenv import dotenv_values
    values=dotenv_values(REPO/'.env')
    return {k:os.environ.get(k) or values.get(k) for k in ('OPENAI_API_KEY','ANTHROPIC_API_KEY')}


class PaidAPI:
    """One ledger for both providers. Unknown sent outcomes keep their reservation."""
    def __init__(self,kind,*,root=None):
        if kind not in {'luna','claude'}:raise ValueError('Unknown provider')
        root=Path(root) if root is not None else ROOT
        self.kind=kind;self.contract=read_json(root/'api_contract.json')
        if self.contract.get('user_approved') is not True:raise PermissionError('No V5 API budget authorization')
        self.spec=self.contract[kind];self.model=self.spec['model']
        if kind=='luna' and self.spec.get('endpoint','responses') not in {'responses','chat_completions'}:
            raise ValueError('Unsupported Luna endpoint')
        for cap in (self.contract['total_cap_usd'],self.spec['cap_usd']):
            if not isinstance(cap,(int,float)) or not math.isfinite(cap) or cap<=0:
                raise ValueError('Positive finite budget caps required')
        if self.spec['input_per_million']<=0 or self.spec['output_per_million']<=0:
            raise ValueError('Known rates required before a paid call')
        self.directory=root/'api';self.directory.mkdir(parents=True,exist_ok=True)
        import httpx
        self.client=httpx.Client(timeout=600)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS calls (fingerprint TEXT PRIMARY KEY, kind TEXT, '
                'status TEXT, reserved REAL, confirmed REAL DEFAULT 0, path TEXT, created REAL)')

    @contextmanager
    def db(self):
        db=sqlite3.connect(self.directory/'ledger.sqlite',timeout=60)
        try:
            with db:yield db
        finally:db.close()

    def request(self,messages,*,item_id,max_output=4096):
        request={'model':self.model,'messages':messages,'max_output':max_output,'settings':self.spec,'item_id':item_id}
        digest=sha_text(json.dumps(request,sort_keys=True,ensure_ascii=False))
        path=self.directory/'calls'/(digest+'.json')
        if path.exists():
            old=read_json(path)
            if old['status']!='completed':raise RuntimeError('Prior terminal API failure preserved: '+digest)
            return old
        keys=credentials();key=keys['OPENAI_API_KEY' if self.kind=='luna' else 'ANTHROPIC_API_KEY']
        if not key:raise RuntimeError('Missing provider credential; no request sent')
        # UTF-8 bytes overbound text tokens; no cache discount in reservations.
        raw=json.dumps(messages,ensure_ascii=False)
        bound=((len(raw.encode())+4096)*max(self.spec['input_per_million'],self.spec.get('cache_write_per_million',0))+
               max_output*self.spec['output_per_million'])/1e6
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM calls WHERE fingerprint=?',(digest,)).fetchone():
                raise RuntimeError('Prior in-flight/unknown API outcome; no automatic retry')
            total=db.execute('SELECT COALESCE(SUM(reserved+confirmed),0) FROM calls').fetchone()[0]
            own=db.execute('SELECT COALESCE(SUM(reserved+confirmed),0) FROM calls WHERE kind=?',(self.kind,)).fetchone()[0]
            if total+bound>self.contract['total_cap_usd'] or own+bound>self.spec['cap_usd']:
                raise BudgetStop('V5 budget cap would be exceeded before dispatch')
            db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?)',(digest,self.kind,'pending',bound,0,str(path),time.time()))
        atomic_new(self.directory/'requests'/(digest+'.json'),request)
        started=time.monotonic();value={'fingerprint':digest,'request':request,'status':'error','reserved_usd':bound}
        try:
            if self.kind=='luna':
                if self.spec.get('endpoint')=='chat_completions':
                    endpoint='chat/completions'
                    body={'model':self.model,'messages':messages,'reasoning_effort':self.spec['reasoning_effort'],
                          'max_completion_tokens':max_output,'store':False,'n':1}
                    if self.spec.get('text_format') is not None:body['response_format']=self.spec['text_format']
                else:
                    endpoint='responses'
                    body={'model':self.model,'input':messages,'reasoning':{'effort':self.spec['reasoning_effort']},
                          'max_output_tokens':max_output,'truncation':'disabled','store':False}
                    if self.spec.get('text_format') is not None:body['text']={'format':self.spec['text_format']}
                if self.spec.get('temperature') is not None:body['temperature']=self.spec['temperature']
                response=self.client.post('https://api.openai.com/v1/'+endpoint,headers={'Authorization':'Bearer '+key},json=body)
            else:
                system='\n'.join(m['content'] for m in messages if m['role']=='system')
                body={'model':self.model,'system':system,'messages':[m for m in messages if m['role']!='system'],
                      'max_tokens':max_output}
                body.update(self.spec.get('generation_options',{}))
                response=self.client.post('https://api.anthropic.com/v1/messages',
                    headers={'x-api-key':key,'anthropic-version':'2023-06-01'},json=body)
            value['http_status']=response.status_code
            if response.status_code!=200:
                # Never persist SDK/HTTP exception bodies or authentication headers.
                value['error_type']='HTTPStatusError'
                if 400<=response.status_code<500 and response.status_code not in (408,429):value['reserved_usd']=0
            else:
                data=response.json();usage=data.get('usage',{})
                if self.kind=='luna':
                    if self.spec.get('endpoint')=='chat_completions':
                        value['provider_usage']=usage
                        choices=data.get('choices',[])
                        if len(choices)!=1:raise ValueError('Exactly one completion choice required')
                        choice=choices[0];text=choice['message'].get('content') or ''
                        stop=choice.get('finish_reason');completed=stop=='stop'
                        usage={**usage,'input_tokens':usage.get('prompt_tokens'),
                            'output_tokens':usage.get('completion_tokens'),
                            'input_tokens_details':usage.get('prompt_tokens_details',{})}
                    else:
                        value['provider_output']=data.get('output',[])
                        text=''.join(c.get('text','') for x in data.get('output',[]) for c in x.get('content',[]) if c.get('type')=='output_text')
                        completed=data.get('status')=='completed';stop=data.get('status')
                    inp=usage.get('input_tokens',0);out=usage.get('output_tokens',0)
                    cached=(usage.get('input_tokens_details') or {}).get('cached_tokens',0)
                    written=(usage.get('input_tokens_details') or {}).get('cache_write_tokens',0)
                    uncached=inp-cached-written
                else:
                    text=''.join(c.get('text','') for c in data.get('content',[]) if c.get('type')=='text')
                    completed=data.get('stop_reason')=='end_turn';stop=data.get('stop_reason')
                    inp=usage.get('input_tokens',0);out=usage.get('output_tokens',0)
                    cached=usage.get('cache_read_input_tokens',0);written=usage.get('cache_creation_input_tokens',0)
                    uncached=inp
                cost=(uncached*self.spec['input_per_million']+out*self.spec['output_per_million']+
                    cached*self.spec.get('cache_read_per_million',self.spec['input_per_million'])+
                    written*self.spec.get('cache_write_per_million',self.spec['input_per_million']))/1e6
                value.update(raw=text,usage=usage,response_model=data.get('model'),response_id=data.get('id'),
                    finish_reason='stop' if completed else stop,status='completed' if completed else 'incomplete',
                    confirmed_usd=cost,reserved_usd=0)
                if data.get('model')!=self.model:
                    value.update(status='model_identity_mismatch')
                if not all(isinstance(usage.get(k),int) and usage[k]>=0 for k in ('input_tokens','output_tokens')) or uncached<0:
                    value.update(status='usage_missing_or_invalid',reserved_usd=bound,confirmed_usd=0)
        except Exception as exc:
            value['error_type']=type(exc).__name__
        value['elapsed_s']=time.monotonic()-started
        atomic_new(path,value)
        with self.db() as db:
            db.execute('UPDATE calls SET status=?,reserved=?,confirmed=? WHERE fingerprint=?',
                       (value['status'],value['reserved_usd'],value.get('confirmed_usd',0),digest))
        if value['status']!='completed':raise RuntimeError('Paid request terminal outcome: '+value['status']+' '+digest)
        return value

    def close(self):
        self.client.close()
        with self.db() as db:
            records=db.execute('SELECT kind,status,reserved,confirmed FROM calls').fetchall()
        write_json(self.directory/'accounting.json',{'calls':len(records),'confirmed_usd':sum(r[3] for r in records),
            'reserved_usd':sum(r[2] for r in records),'by_provider':{k:{'calls':sum(r[0]==k for r in records),
                'confirmed_usd':sum(r[3] for r in records if r[0]==k),'reserved_usd':sum(r[2] for r in records if r[0]==k)}
                for k in ('luna','claude')}})


class LunaWorker:
    def __init__(self,api,case):self.api=api;self.case=case

    def generate(self,messages,role,max_tokens=None):
        # Stable identity includes full observation; completed requests replay.
        identity=self.case+':'+role+':'+sha_text(json.dumps(messages,ensure_ascii=False))
        value=self.api.request(messages,item_id=identity,max_output=max_tokens or 1024)
        return {**value,'generation':{'model':self.api.model,'settings':self.api.spec}}
