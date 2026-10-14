"""CPU analysis that yields the shared Bareun service to task A inference."""
from contextlib import contextmanager
import fcntl
import json
import os
from threading import Lock
import time

from ..common import read_json, sha_text, write_json
from ..corrupt.document import BareunBank
from .candidates import PriorityParagraphs
from .config import PHASE

_LOCK = Lock()


def priority_active(config):
    path = config['paths']['repo'] / 'verak/v3/outputs/phase8_rft1/status.json'
    if not path.exists():
        return False
    state = read_json(path)
    stage = state.get('stage', '')
    # Training uses neither the Bareun service nor this CPU affinity allocation.
    return stage in {'rollouts', 'evaluating', 'evaluation'} or stage.startswith(('evaluate', 'evaluating', 'rollout'))


@contextmanager
def bareun_slot(config):
    root = config['paths'][PHASE + '_output']
    with _LOCK, (root / 'bareun_service.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        while priority_active(config):
            write_json(root / 'bareun_wait.json', {'waiting_for': 'task_A_inference', 'pid': os.getpid(), 'at': time.time()})
            time.sleep(10)
        stamp = root / 'bareun_last_call.json'
        previous = read_json(stamp)['at'] if stamp.exists() else 0
        remaining = 1.0 - (time.time()-previous)
        if remaining > 0:
            time.sleep(remaining)
        try:
            yield
        finally:
            write_json(stamp, {'at': time.time(), 'pid': os.getpid(), 'gpu_used': False})


class RetryParagraphs(PriorityParagraphs):
    def profile(self, text, neighbors=()):
        key = sha_text(json.dumps([text, sha_text(json.dumps(neighbors, ensure_ascii=False))], ensure_ascii=False))
        if (self.cache_dir / (key + '.json')).exists():
            return super().profile(text, neighbors)
        with bareun_slot(self.config):
            return super().profile(text, neighbors)


class RetryBank(BareunBank):
    def tokens(self, text):
        name = sha_text(text) + '.json'
        if text in self.memory or any((path / name).exists() for path in (self.cache,) + self.read_cache_dirs):
            return super().tokens(text)
        with bareun_slot(self.config):
            return super().tokens(text)
