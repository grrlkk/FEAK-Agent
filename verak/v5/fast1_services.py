"""FAST1 local scorer and base-model perplexity HTTP services.

The scorer endpoint preserves ReferenceScores result/cache/fingerprint semantics.
No essay normalization happens here; callers own their frozen normalization.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from . import common

FAST1 = common.REPO / 'verak/v5/outputs/fast1'
SCORES_ROOT = FAST1 / 'shared'
REFERENCE_FALLBACK = common.REPO / 'verak/v5/outputs/pipe1'
VLLM_URL = 'http://127.0.0.1:8051'

SCORE_LOCK = threading.Lock()
PPL_LOCK = threading.Lock()
_score = None
_ppl_cache: dict[str, dict] = {}
_policy_tokenizer = None


def _json_response(handler, status, payload):
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _score_call(question, text):
    global _score
    if not isinstance(question, str) or not isinstance(text, str):
        raise ValueError('question and text must be strings')
    if not question.strip() or not text.strip():
        raise ValueError('question and text must be nonblank')
    # ReferenceScores uses per-pair JSON cache files and the established scorer
    # identity fingerprint. Its lazy model initialization and CUDA use are
    # serialized with every score request.
    with SCORE_LOCK:
        if _score is None:
            _score = common.ReferenceScores(root=SCORES_ROOT, fallback=REFERENCE_FALLBACK)
        result = _score(question, text)
        identity_path = SCORES_ROOT / 'scorer_identity.json'
        if identity_path.exists():
            fingerprint = common.read_json(identity_path)['fingerprint']
        else:
            identity = common.read_json(common.REPO / 'verak/v4/outputs/train44/smoke/scorer_identity.json')
            common.write_json(identity_path, identity)
            fingerprint = identity['fingerprint']
    return {'result': result, 'fingerprint': fingerprint}


def _completion_perplexities(texts):
    """Score one or more prompts with the base policy model via echo logprobs."""
    keys = [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in texts]
    output = [None] * len(texts)
    missing = []
    with PPL_LOCK:
        for i, key in enumerate(keys):
            if key in _ppl_cache:
                output[i] = {**_ppl_cache[key], 'cache_hit': True}
            else:
                missing.append((i, key, texts[i]))
    if not missing:
        return output
    payload = {'model': 'kanana-v5', 'prompt': [row[2] for row in missing],
               'echo': True, 'max_tokens': 1, 'temperature': 0,
               'logprobs': 1, 'seed': 47}
    request = urllib.request.Request(VLLM_URL + '/v1/completions',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f'base-model completion request failed: {exc}') from exc
    choices = data.get('choices') or []
    if len(choices) != len(missing):
        raise RuntimeError('vLLM returned a different number of completion choices')
    for (i, key, prompt), choice in zip(missing, choices):
        logprobs = choice.get('logprobs') or {}
        values = logprobs.get('token_logprobs')
        if not values:
            raise RuntimeError('vLLM echo response has no token_logprobs')
        # Batch usage.prompt_tokens is aggregate, so count each prompt locally
        # with the same pinned tokenizer. vLLM echo arrays contain one entry per
        # prompt token (including the BOS position) followed by the generated
        # probe token; the BOS logprob is null and is naturally excluded below.
        global _policy_tokenizer
        if _policy_tokenizer is None:
            _policy_tokenizer = common.tokenizer()
        prompt_count = len(_policy_tokenizer.encode(prompt, add_special_tokens=True))
        prompt_values = values[:prompt_count]
        finite = [float(x) for x in prompt_values if x is not None]
        if not finite:
            raise RuntimeError('vLLM returned no prompt token log probabilities')
        mean_nll = -sum(finite) / len(finite)
        result = {'mean_nll': mean_nll,
                  'perplexity': float(__import__('math').exp(mean_nll)),
                  'mean_log_probability': -mean_nll, 'token_count': len(finite),
                  'model': 'kanana-v5-base', 'adapter': None,
                  'prompt_sha256': key, 'cache_hit': False}
        with PPL_LOCK:
            _ppl_cache[key] = result
        output[i] = result
    return output


class Handler(BaseHTTPRequestHandler):
    server_version = 'FAST1Services/1.0'

    def log_message(self, fmt, *args):
        line = '%s - %s\n' % (self.log_date_time_string(), fmt % args)
        with open(FAST1 / 'services_http.log', 'a', encoding='utf-8') as stream:
            stream.write(line)

    def do_GET(self):
        if self.path == '/health':
            _json_response(self, 200, {'ok': True, 'service': 'fast1-services',
                'score_ready': _score is not None, 'perplexity_model': 'kanana-v5 base'})
        else:
            _json_response(self, 404, {'error': 'not found'})

    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > 20_000_000:
                raise ValueError('body must be between 1 byte and 20 MB')
            body = json.loads(self.rfile.read(size))
            if self.path == '/score':
                value = _score_call(body.get('question'), body.get('text'))
                # Direct scorer-result keys keep this drop-in for score callers;
                # fingerprint is additive provenance.
                _json_response(self, 200, {**value['result'], 'fingerprint': value['fingerprint']})
            elif self.path == '/perplexity':
                if 'texts' in body:
                    texts = body['texts']
                    if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
                        raise ValueError('texts must be a list of strings')
                    values = _completion_perplexities(texts)
                    _json_response(self, 200, {'results': values})
                else:
                    text = body.get('text')
                    if not isinstance(text, str):
                        raise ValueError('text must be a string')
                    _json_response(self, 200, _completion_perplexities([text])[0])
            else:
                _json_response(self, 404, {'error': 'not found'})
        except (ValueError, json.JSONDecodeError) as exc:
            _json_response(self, 400, {'error': str(exc)})
        except Exception as exc:
            _json_response(self, 500, {'error': str(exc)})


def serve(host='127.0.0.1', port=8052):
    FAST1.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    common.write_json(FAST1 / 'scorer_service.json', {'pid': os.getpid(),
        'host': host, 'port': port, 'physical_gpu': 1,
        'reference_cache': str(SCORES_ROOT), 'fallback_cache': str(REFERENCE_FALLBACK),
        'started_at': time.time(), 'training': False})
    server.serve_forever()


if __name__ == '__main__':
    serve()
