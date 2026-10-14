"""Provider Batch APIs sharing PaidAPI's durable $5 reservation ledger."""
import json
import time

from .api import PaidAPI, BudgetStop, credentials
from .common import sha_text, read_json, write_json, atomic_new


class Batch(PaidAPI):
    def headers(self):
        key = credentials()['OPENAI_API_KEY' if self.kind == 'luna' else 'ANTHROPIC_API_KEY']
        if not key: raise RuntimeError('Provider credential missing')
        return {'Authorization': 'Bearer '+key} if self.kind == 'luna' else {'x-api-key': key, 'anthropic-version': '2023-06-01'}

    def body(self, prompt, max_output):
        messages = [{'role': 'user', 'content': prompt}]
        if self.kind == 'luna':
            return {'model': self.model, 'messages': messages, 'reasoning_effort': self.spec['reasoning_effort'],
                    'temperature': self.spec['temperature'], 'max_completion_tokens': max_output, 'store': False, 'n': 1}
        return {'model': self.model, 'messages': messages, 'max_tokens': max_output, **self.spec['generation_options']}

    def send(self, name, rows):
        """rows contain opaque custom_id, prompt, max_output; never system labels."""
        if len({r['custom_id'] for r in rows}) != len(rows): raise ValueError('Duplicate batch custom IDs')
        bundle = {'kind': self.kind, 'name': name, 'spec': self.spec, 'rows': rows}
        digest = sha_text(json.dumps(bundle, sort_keys=True, ensure_ascii=False))
        path = self.directory/'batches'/(digest+'.json')
        if path.exists():
            saved = read_json(path)
            if not saved.get('batch_id'): raise RuntimeError('Unknown batch dispatch: reconcile before retry')
            return saved
        prepared = []
        for row in rows:
            body = self.body(row['prompt'], row['max_output'])
            fp = sha_text(json.dumps([self.kind, name, row['custom_id'], body, 'batch50'], sort_keys=True, ensure_ascii=False))
            bound = ((len(json.dumps(body, ensure_ascii=False).encode())+4096)*max(self.spec['input_per_million'], self.spec.get('cache_write_per_million',0)) + row['max_output']*self.spec['output_per_million'])/2e6
            prepared.append({'fingerprint': fp, 'row': row, 'body': body, 'bound': bound})
        bound = sum(p['bound'] for p in prepared)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            total = db.execute('SELECT COALESCE(SUM(reserved+confirmed),0) FROM calls').fetchone()[0]
            own = db.execute('SELECT COALESCE(SUM(reserved+confirmed),0) FROM calls WHERE kind=?',(self.kind,)).fetchone()[0]
            if total+bound > self.contract['total_cap_usd'] or own+bound > self.spec['cap_usd']:
                raise BudgetStop('Batch reservation would exceed the shared experiment cap')
            for p in prepared:
                db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?)', (p['fingerprint'], self.kind, 'batch_pending', p['bound'], 0,
                           str(self.directory/'calls'/(p['fingerprint']+'.json')), time.time()))
        saved = {'digest': digest, 'kind': self.kind, 'name': name, 'started_at': time.time(), 'prepared': prepared,
                 'status': 'dispatching', 'batch_id': None, 'manifest': str(path)}
        atomic_new(path, saved)
        headers = self.headers()
        if self.kind == 'luna':
            payload = '\n'.join(json.dumps({'custom_id': p['row']['custom_id'], 'method':'POST', 'url':'/v1/chat/completions', 'body':p['body']}, ensure_ascii=False) for p in prepared)+'\n'
            response = self.client.post('https://api.openai.com/v1/files', headers=headers,
                                        data={'purpose':'batch'}, files={'file':('requests.jsonl', payload.encode(), 'application/jsonl')})
            if response.status_code != 200: raise RuntimeError('Batch file upload HTTP '+str(response.status_code))
            saved['input_file_id'] = response.json()['id']; write_json(path, saved)
            response = self.client.post('https://api.openai.com/v1/batches', headers=headers,
                json={'input_file_id':saved['input_file_id'], 'endpoint':'/v1/chat/completions', 'completion_window':'24h', 'metadata':{'experiment':'luna_v2'}})
        else:
            response = self.client.post('https://api.anthropic.com/v1/messages/batches', headers=headers,
                json={'requests':[{'custom_id':p['row']['custom_id'], 'params':p['body']} for p in prepared]})
        if response.status_code not in (200, 201):
            saved['http_status'] = response.status_code; write_json(path, saved)
            # Unknown dispatch remains reserved; never silently resubmit.
            raise RuntimeError('Batch submission HTTP '+str(response.status_code))
        saved.update(batch_id=response.json()['id'], status='submitted', provider=response.json())
        write_json(path, saved); return saved

    def poll(self, saved):
        if saved.get('status') == 'completed': return saved
        prefix = 'https://api.openai.com/v1/batches/' if self.kind == 'luna' else 'https://api.anthropic.com/v1/messages/batches/'
        response = self.client.get(prefix+saved['batch_id'], headers=self.headers())
        if response.status_code != 200: raise RuntimeError('Batch poll HTTP '+str(response.status_code))
        data = response.json(); saved['provider'] = data
        done = data.get('status') == 'completed' if self.kind == 'luna' else data.get('processing_status') == 'ended'
        if not done:
            saved['status'] = data.get('status') or data.get('processing_status')
            write_json(saved['manifest'], saved); return saved
        if self.kind == 'luna':
            files = [data[k] for k in ('output_file_id', 'error_file_id') if data.get(k)]
            lines = []
            for file_id in files:
                r = self.client.get('https://api.openai.com/v1/files/'+file_id+'/content', headers=self.headers())
                if r.status_code != 200: raise RuntimeError('Batch output download HTTP '+str(r.status_code))
                lines.extend(json.loads(line) for line in r.text.splitlines() if line)
        else:
            r = self.client.get(prefix+saved['batch_id']+'/results', headers=self.headers())
            if r.status_code != 200: raise RuntimeError('Batch results HTTP '+str(r.status_code))
            lines = [json.loads(line) for line in r.text.splitlines() if line]
        by_id = {line['custom_id']: line for line in lines}
        if len(by_id) != len(lines) or set(by_id) != {p['row']['custom_id'] for p in saved['prepared']}:
            raise ValueError('Missing/duplicate/unknown batch results; reservations retained')
        outputs = {}
        for p in saved['prepared']:
            line = by_id[p['row']['custom_id']]
            success = line.get('response', {}).get('status_code') == 200 if self.kind == 'luna' else line['result']['type'] == 'succeeded'
            if not success:
                # Failed batch items are not counted as successful comparisons.
                outputs[p['row']['custom_id']] = {'status':'provider_error', 'provider_result':line}
                continue
            body = line['response']['body'] if self.kind == 'luna' else line['result']['message']
            u = body['usage']
            if self.kind == 'luna':
                inp, out = u['prompt_tokens'], u['completion_tokens']
                details = u.get('prompt_tokens_details') or {}
                cached, written = details.get('cached_tokens',0), details.get('cache_write_tokens',0)
                uncached = inp-cached-written
                raw = body['choices'][0]['message']['content']; stop = body['choices'][0]['finish_reason']
                complete = stop == 'stop'
            else:
                inp, out = u['input_tokens'], u['output_tokens']
                cached, written = u.get('cache_read_input_tokens',0), u.get('cache_creation_input_tokens',0)
                uncached = inp
                raw = ''.join(x.get('text','') for x in body['content'] if x['type']=='text')
                stop = body['stop_reason']; complete = stop == 'end_turn'
            if body['model'] != self.model: raise ValueError('Batch response model identity mismatch')
            if not all(isinstance(x,int) and x>=0 for x in [inp,out,cached,written,uncached]): raise ValueError('Invalid batch usage')
            cost = (uncached*self.spec['input_per_million']+out*self.spec['output_per_million']+
                    cached*self.spec.get('cache_read_per_million',self.spec['input_per_million'])+
                    written*self.spec.get('cache_write_per_million',self.spec['input_per_million']))/2e6
            value = {'fingerprint':p['fingerprint'], 'status':'completed' if complete else 'incomplete', 'raw':raw,
                     'usage':{'input_tokens':inp, 'output_tokens':out, 'cached_tokens':cached, 'cache_write_tokens':written},
                     'provider_usage':u, 'response_model':body['model'], 'confirmed_usd':cost, 'reserved_usd':0,
                     'batch_id':saved['batch_id'], 'provider_result':line, 'stop_reason':stop}
            write_json(self.directory/'calls'/(p['fingerprint']+'.json'), value)
            with self.db() as db:
                db.execute('UPDATE calls SET status=?,reserved=0,confirmed=? WHERE fingerprint=?',(value['status'],cost,p['fingerprint']))
            outputs[p['row']['custom_id']] = value
        saved.update(status='completed', outputs=outputs, ended_at=time.time(), wall_seconds=time.time()-saved['started_at'])
        write_json(saved['manifest'], saved); self.close(); return saved
