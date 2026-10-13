"""V5_PIPE1 artifacts, frozen inference runtime and reference GPU scores."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from verak.v4.common import REPO, atomic_new, file_sha, load_config, read_json, safe_id, sha_text, write_json
from verak.v4.dryrun_common import BASE, VLLM_PYTHON, REVISION, tokenizer, local_analysis_environment
from verak.v4.full_data import ref, verified
from verak.v4.full_run import INFERENCE_PYTHON

ROOT = REPO / 'verak/v5/outputs/pipe1'
REPORT = REPO / 'reports/V5_PIPE1.md'
WORKTREE = Path(__file__).resolve().parents[2]
PORT = 8051
ADAPTERS = {r: REPO / 'verak/v4/outputs/train44/adapters' / r / f'epoch_{n}'
            for r, n in {'orchestrator': 2, 'revision': 1, 'korean': 2, 'oneshot': 1}.items()}
PUBLIC_FIELDS = ('source_id', 'genre', 'question', 'text', 'profile', 'layout', 'normalization',
                 'paragraphs', 'environment_version')


def public_row(row):
    return {k: row[k] for k in PUBLIC_FIELDS if k in row}


def append_report(title, body, *, seconds):
    import fcntl
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stamp = datetime.now(timezone.utc).isoformat()
        stream.write(f'\n## {title}\n\nUTC {stamp}; wall-clock {seconds:.1f} s.\n\n{body}\n')
        stream.flush()


def check_runtime():
    """Fail before inference in an interpreter without the local SEARCH stack."""
    from verak.v4.search_v43 import verify_index, local_search
    started = time.monotonic()
    if Path(sys.executable).absolute() != INFERENCE_PYTHON.absolute():
        raise RuntimeError(f'Run with the SEARCH-enabled interpreter: {INFERENCE_PYTHON}')
    value = {'python': sys.executable,
             'packages': {n: importlib.metadata.version(n) for n in ('bm25s', 'kiwipiepy')},
             'index': verify_index(),
             'hits': [{k: p[k] for k in ('title', 'passage_id')} for p in local_search('한글 창제 세종')],
             'elapsed_s': time.monotonic() - started, 'new_API_calls': 0, 'GPU_used': False}
    if len(value['hits']) != 3:
        raise RuntimeError('Local SEARCH did not return three passages')
    return value


def preflight():
    runtime = check_runtime()
    config = load_config()
    if config['bareun'].get('host') not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('V5 uses the existing local Bareun server only')
    from verak.v4 import orchestrator_v44, policy_prompts_v43
    orchestrator_v44.verify(); policy_prompts_v43.verify_frozen()
    for path in ADAPTERS.values():
        p = read_json(path/'provenance.json')
        if p['base_revision'] != REVISION or p['adapter_sha256'] != file_sha(path/'adapter_model.safetensors'):
            raise ValueError('Frozen adapter provenance differs')
    manifest = {
        'base_revision': REVISION, 'environment': 'v4.6', 'temperature': 0., 'top_p': 1., 'seed': 47,
        'training': False, 'test_cohort_untouched': True,
        'adapters': {r: ref(p/'adapter_model.safetensors') for r,p in ADAPTERS.items()},
        'A_design': ref(REPO/'verak/v3/outputs/phase7_sft/evaluation_design.json'),
        'C_design': ref(REPO/'verak/v4/outputs/dryrun/evaluation/manifest.json'),
        'environment_files': {p.name: ref(p) for p in sorted((WORKTREE/'verak/v4').glob('environment_v4*.py'))},
        'worker_prompt': ref(policy_prompts_v43.FROZEN), 'orchestrator_prompt': ref(orchestrator_v44.FROZEN)}
    path = ROOT/'preflight.json'
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError('V5 frozen preflight changed')
    else:
        atomic_new(path, manifest)
        atomic_new(ROOT/'search_runtime.json', runtime)
        append_report('Runtime and frozen contract',
            'No training. O epoch 2 / Revision epoch 1 / Korean epoch 2 / v4 one-pass epoch 1; '
            'v4.6, temperature 0. The older FIX1 continuation workers are not used.\n\n'
            f'SEARCH: `{runtime["python"]}`, bm25s {runtime["packages"]["bm25s"]}, '
            f'Kiwi {runtime["packages"]["kiwipiepy"]}; local top-3 retrieval passed. '
            'Every V5 inference subprocess uses this interpreter and checks SEARCH before dispatch. '
            'No source or query leaves the server for SEARCH. Model/API evaluation is recorded separately.\n\n'
            'A uses the frozen dev100. C uses smoke20 + agent_dev10. No test-cohort inference. '
            'Artifact root: `verak/v5/outputs/pipe1`. Results below are appended as parts finish.',
            seconds=runtime['elapsed_s'])
    return manifest


class Backend:
    def __init__(self):
        import httpx
        self.client = httpx.Client(timeout=600)

    def generate(self, messages, role, max_tokens=None):
        from verak.v4.environment_v46 import OUTPUT
        started = time.monotonic()
        request = {'model': 'v5-'+role, 'messages': messages, 'temperature': 0., 'top_p': 1.,
                   'seed': 47, 'max_tokens': max_tokens or OUTPUT[role]}
        r = self.client.post(f'http://127.0.0.1:{PORT}/v1/chat/completions', json=request)
        r.raise_for_status(); data = r.json()
        return {'raw': data['choices'][0]['message']['content'], 'usage': data.get('usage', {}),
                'finish_reason': data['choices'][0]['finish_reason'], 'response_model': data.get('model'),
                'generation': {k:v for k,v in request.items() if k != 'messages'},
                'elapsed_s': time.monotonic()-started}

    def close(self):
        self.client.close()


class ReferenceScores:
    def __init__(self):
        self.scorer = None

    def __call__(self, question, text):
        path = ROOT/'scores'/(sha_text(json.dumps([question, text], ensure_ascii=False))+'.json')
        if path.exists():
            value = read_json(path)
            if value['question'] != question or value['text'] != text:
                raise ValueError('Score cache mismatch')
            return value['result']
        if self.scorer is None:
            from verak.v3.score.kanana import KananaScorer
            config = load_config(); config['scorer']['gpu'] = 0
            config['paths']['output'] = ROOT/'scorer'
            self.scorer = KananaScorer(config, cache_enabled=False)
            prior = read_json(REPO/'verak/v4/outputs/train44/smoke/scorer_identity.json')
            if self.scorer.fingerprint != prior['fingerprint']:
                raise ValueError('GPU reference scorer fingerprint differs')
            write_json(ROOT/'scorer_identity.json', {'physical_gpu': 1, 'fingerprint': self.scorer.fingerprint,
                'settings': config['scorer'], 'GPU_reference_only': True})
        result = self.scorer.score(question, text, use_cache=False).to_dict()
        atomic_new(path, {'question': question, 'text': text, 'result': result,
                          'fingerprint': self.scorer.fingerprint, 'physical_gpu': 1})
        return result

    def close(self):
        if self.scorer is not None:
            self.scorer.close()
