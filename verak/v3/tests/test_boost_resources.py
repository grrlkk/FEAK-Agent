from concurrent.futures import ThreadPoolExecutor
import time

from verak.v3.common import write_json
from verak.v3.insertion_boost.resources import bareun_slot, new_priority_errors


def test_shared_bareun_slots_rate_limit_two_tasks(tmp_path):
    config = {'paths': {'repo': tmp_path}}
    starts = []
    def one():
        with bareun_slot(config, kind='test'):
            starts.append(time.monotonic())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: one(), range(2)))
    assert starts[1]-starts[0] >= .99


def test_only_new_bareun_errors_pause_priority_inference(tmp_path):
    config = {'paths': {'repo': tmp_path}}
    shared = tmp_path / 'shared'
    log = tmp_path / 'verak/v3/outputs/phase8_rft1/rollouts.log'
    log.parent.mkdir(parents=True)
    log.write_text('old Bareun timeout\n')
    assert new_priority_errors(config, shared) == []
    with log.open('a') as stream:
        stream.write('new unrelated error\nBareun RPC TimeoutError\n')
    assert new_priority_errors(config, shared) == [{'log': 'rollouts.log', 'line': 'Bareun RPC TimeoutError'}]
    assert new_priority_errors(config, shared) == []
