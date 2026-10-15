"""Shared low-priority Bareun slots for independent CPU/API data tasks.

Cache hits do not acquire the service slot. Cache misses are serialized and at
least one second apart across both boosts. New busy/timeout failures from this
service or task A pause misses throughout A's current inference stage.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from threading import Lock
import time

from ..common import read_json, sha_text, write_json
from ..corrupt.document import BareunBank
from ..env.analysis import ParagraphAnalyzer
from ..v2_ops.retry_resources import priority_active
from .cpu_score import shared_root

_THREAD_LOCK = Lock()


def cache_roots(config, kind):
    outputs = config['paths']['repo'] / 'verak/v3/outputs'
    return [outputs / name / kind for name in ('phase5', 'phase6', 'phase7_teacher',
        'teacher_bulk_two_stage', 'phase7_sft', 'v2_ops', 'v2_ops_retry')]


def new_priority_errors(config, root):
    """Only read new bytes; historical errors cannot trigger a false pause."""
    state_path = root / 'bareun_rft_log_offsets.json'
    offsets = read_json(state_path) if state_path.exists() else {}
    errors = []
    for name in ('rollouts.log', 'controller.log'):
        path = config['paths']['repo'] / 'verak/v3/outputs/phase8_rft1' / name
        if not path.exists():
            continue
        length = path.stat().st_size
        previous = offsets.get(name, length)
        if previous > length:
            previous = 0
        with path.open('rb') as stream:
            stream.seek(previous)
            payload = stream.read().decode('utf-8', errors='replace')
        offsets[name] = length
        for line in payload.splitlines():
            lower = line.lower()
            if any(s in lower for s in ('bareun', 'grpc')) and any(
                    s in lower for s in ('error', 'timeout', 'busy', 'resource_exhausted')):
                errors.append({'log': name, 'line': line[:2000]})
    write_json(state_path, offsets)
    return errors


@contextmanager
def bareun_slot(config, *, kind):
    root = shared_root(config)
    root.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK, (root / 'bareun_service.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        errors = new_priority_errors(config, root)
        pause = root / 'bareun_pause.json'
        if errors:
            write_json(pause, {'reason': 'new_task_A_Bareun_error', 'errors': errors,
                'at': time.time(), 'pid': os.getpid()})
        while pause.exists() and priority_active(config):
            write_json(root / 'bareun_wait.json', {'waiting_for': 'task_A_inference_after_service_error',
                'pid': os.getpid(), 'at': time.time(), 'pause': read_json(pause)})
            time.sleep(10)
        stamp = root / 'bareun_last_call.json'
        previous = read_json(stamp)['at'] if stamp.exists() else 0
        delay = 1.0-(time.time()-previous)
        if delay > 0:
            time.sleep(delay)
        before = time.time()
        error = None
        try:
            yield
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
            lower = str(exc).lower()
            if any(s in lower for s in ('busy', 'timeout', 'timed out', 'resource_exhausted', 'deadline_exceeded')):
                write_json(pause, {'reason': 'boost_Bareun_busy_or_timeout', 'error': error,
                                  'at': time.time(), 'pid': os.getpid()})
            raise
        finally:
            row = {'at': time.time(), 'seconds': time.time()-before, 'kind': kind,
                'pid': os.getpid(), 'task_root': str(config['paths'].get('v2_ops_output', 'global_boost')),
                'error': error, 'gpu_used': False}
            write_json(stamp, row)
            with (root / 'bareun_latency.jsonl').open('a') as log:
                log.write(json.dumps(row, ensure_ascii=False)+'\n')


class BoostParagraphs(ParagraphAnalyzer):
    def profile(self, text, neighbors=()):
        key = sha_text(json.dumps([text, sha_text(json.dumps(neighbors, ensure_ascii=False))], ensure_ascii=False))
        path = self.cache_dir / (key + '.json')
        if not path.exists():
            for directory in cache_roots(self.config, 'bareun_paragraphs'):
                old = directory / path.name
                if old.exists():
                    value = read_json(old)
                    if value['text'] != text or value['neighbors'] != list(neighbors):
                        raise ValueError('Exact Bareun profile cache key mismatch')
                    write_json(path, value)
                    break
        if path.exists():
            return super().profile(text, neighbors)
        with bareun_slot(self.config, kind='paragraph'):
            return super().profile(text, neighbors)


class BoostBank(BareunBank):
    def __init__(self, config, **kwargs):
        prior = kwargs.pop('read_cache_dirs', ())
        super().__init__(config, read_cache_dirs=tuple(prior)+tuple(cache_roots(config, 'bareun_units')), **kwargs)

    def tokens(self, text):
        name = sha_text(text)+'.json'
        if text in self.memory or any((root / name).exists() for root in (self.cache,)+self.read_cache_dirs):
            return super().tokens(text)
        with bareun_slot(self.config, kind='unit'):
            return super().tokens(text)
