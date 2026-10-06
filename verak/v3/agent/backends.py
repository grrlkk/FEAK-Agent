"""Conversation extension of the existing environment-only API adapter."""
import copy
import json
import os
from pathlib import Path
import time

from feak_tc.agent.schemas import OpenAIConfig
from feak_tc.runtime.openai import CallBudgetExceeded
from ..api import EnvironmentJSONClient, PhaseBudget
from ..common import read_json, sha_text, write_json
from ..phase2 import read_jsonl
from ..reconstruction_api import usage_cost


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class ConversationAdapter(EnvironmentJSONClient):
    def complete(self, messages):
        self._load()
        # System and every earlier message are passed verbatim, as real turns.
        response = self._client.responses.create(model=self.cfg.model, input=messages,
            reasoning={'effort': self.cfg.reasoning_effort},
            max_output_tokens=self.cfg.max_output_tokens, truncation='disabled', store=False)
        return {'raw': response.output_text, 'status': response.status,
                'response_id': response.id, 'response_model': response.model,
                'usage': response.usage.model_dump() if response.usage else None}


class TeacherBackend:
    name = 'teacher'
    context_limit = 250000

    def __init__(self, config, output, *, max_api_calls, max_cost_usd=10., adapter_factory=ConversationAdapter):
        if not 0 < max_cost_usd <= 10:
            raise ValueError('Phase 5 confirmed cost cap is $10')
        settings = config['runtime']
        self.model = config['role_teacher']['model']
        if (self.model, config['role_teacher']['reasoning_effort']) != ('gpt-6.1-sol', 'low'):
            raise ValueError('Phase 5 teacher must be gpt-6.1-sol / low')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cap = max_cost_usd
        self.budget = PhaseBudget(self.output/'api_budget.json', max_api_calls,
            authorized_ceiling=settings['phase_api_ceiling'], phase='v3_phase5')
        self.cfg = OpenAIConfig(model=self.model, reasoning_effort='low',
            max_output_tokens=config['policy'].get('generation_reserve', 1024), timeout_s=600,
            max_input_chars=1000000, max_calls_total=settings['phase_api_ceiling'],
            max_calls_per_sample=settings['phase_api_ceiling'])
        self.adapter = adapter_factory(self.cfg)
        self.records = read_jsonl(self.output/'calls.jsonl') if (self.output/'calls.jsonl').exists() else []
        reservations = {r['phase_call'] for r in self.records if r.get('event') == 'reserved'}
        responses = {r['phase_call'] for r in self.records if 'status' in r}
        if reservations != responses or len(reservations) != self.budget.used:
            raise ValueError('Unresolved Phase 5 API reservation; inspect ledger before resuming')

    def accounting(self):
        rows = [r for r in self.records if 'status' in r]
        return {'calls': self.budget.used,
                'confirmed_usd': sum(r.get('cost', {}).get('confirmed_usd', 0) for r in rows),
                'timeout_reservations_usd': sum(r.get('timeout_reservation_usd', 0) for r in rows)}

    def log(self, record):
        append_jsonl(self.output/'calls.jsonl', record)
        self.records.append(record)

    def generate(self, messages, *, episode_id, role, turn):
        messages = copy.deepcopy(messages)
        context = {'episode_id': episode_id, 'role': role, 'turn': turn,
                   'model': self.model, 'reasoning_effort': 'low', 'messages': messages,
                   'max_output_tokens': self.cfg.max_output_tokens}
        key = sha_text(json.dumps(context, ensure_ascii=False, sort_keys=True))
        cache = self.output/'request_cache'/(key+'.json')
        if cache.exists():
            return {**read_json(cache), 'replayed': True}
        successful = next((r for r in self.records if r.get('fingerprint') == key and
                           r.get('status') == 'completed'), None)
        if successful:
            write_json(cache, successful)
            return {**successful, 'replayed': True}
        self.adapter._load()  # Existing launcher supplies environment; never log the key.
        serialized = json.dumps(messages, ensure_ascii=False)
        bound = ((len(serialized.encode('utf-8'))+4096)*2.5 + self.cfg.max_output_tokens*10) / 1e6
        budget = self.accounting()
        if budget['confirmed_usd'] + budget['timeout_reservations_usd'] + bound > self.cap:
            raise CallBudgetExceeded('Phase 5 $10 cap: insufficient room for the next request')
        phase_call = self.budget.reserve()
        self.log({'event': 'reserved', 'phase_call': phase_call, 'fingerprint': key,
                  'episode_id': episode_id, 'bound_usd': bound})
        started = time.monotonic()
        record = {**context, 'fingerprint': key, 'phase_call': phase_call, 'replayed': False}
        try:
            record.update(self.adapter.complete(messages))
            if record.get('usage'):
                record['cost'] = usage_cost(self.model, record['usage'])
            if record['status'] != 'completed':
                record['error_type'] = 'IncompleteResponse'
        except Exception as error:
            record.update(status='error', error_type=type(error).__name__,
                          http_status=getattr(error, 'status_code', None))
            if 'Timeout' in type(error).__name__:
                record['timeout_reservation_usd'] = bound
        record['elapsed_s'] = time.monotonic()-started
        self.log(record)
        if record['status'] != 'completed':
            raise RuntimeError('Teacher request failed: ' + record.get('error_type', record['status']))
        write_json(cache, record)
        return record

    def close(self):
        self.adapter.close()


class PolicyBackend:
    name = 'policy'
    context_limit = 32768

    def __init__(self, config, output):
        import httpx
        self.settings = config['policy']
        self.context_limit = self.settings.get('context_limit', 32768)
        self.output = Path(output)
        self.model = self.settings['model']
        self.client = httpx.Client(timeout=600)

    def generate(self, messages, *, episode_id, role, turn):
        model = self.settings.get('adapters', {}).get(role) or self.model
        payload = {'model': model, 'messages': copy.deepcopy(messages),
                   'temperature': 0., 'top_p': 1., 'seed': 47,
                   'max_tokens': self.settings.get('generation_reserve', 1024)}
        # No JSON-constrained decoder: measure the untrained model's own output.
        started = time.monotonic()
        response = self.client.post(self.settings['base_url']+'/chat/completions', json=payload)
        response.raise_for_status()
        result = response.json()
        record = {'episode_id': episode_id, 'role': role, 'turn': turn,
            'model': model, 'messages': payload['messages'], 'raw': result['choices'][0]['message']['content'],
            'usage': result.get('usage', {}), 'finish_reason': result['choices'][0]['finish_reason'],
            'elapsed_s': time.monotonic()-started, 'cost': {'confirmed_usd': 0}, 'replayed': False}
        append_jsonl(self.output/'policy_calls.jsonl', record)
        return record

    def close(self):
        self.client.close()
