from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import pytest

from verak.v3.agent.runner import system_prompt
from verak.v3.common import load_config
from verak.v3.eval.api import Phase6API
from verak.v3.eval.filter import filter_rows
from feak_tc.runtime.openai import CallBudgetExceeded


@pytest.mark.parametrize('role', ['global', 'korean', 'single'])
def test_content_scope_and_stop_in_every_role(role):
    text = system_prompt(role)
    assert '일반 상식이어도 새로 넣지 않는다' in text
    assert 'STOP summary' in text and '무엇을 바꿨는지' in text
    assert '글쓴이가 추가해야 할 내용' in text
    variant = system_prompt(role, 'check_once')
    assert variant.startswith(text) and 'CHECK를 정확히 한 번' in variant


def test_filter_drops_whole_essay_and_keeps_order_without_mutating_source():
    rows = [{'episode_id': 'a', 'records': [{'op': 'G_DELETE_SUPPORT', 'record_id': 'r'}]},
            {'episode_id': 'b', 'records': [{'op': 'L_REGISTER', 'record_id': 'r'}]},
            {'episode_id': 'c', 'records': [{'op': 'G_DELETE_SUPPORT', 'record_id': 'r'}]}]
    original = deepcopy(rows)
    kept, dropped = filter_rows(rows, {'a:r': {'recoverable': False, 'note': 'absent'},
                                      'c:r': {'recoverable': True, 'note': 'present'}})
    assert [r['episode_id'] for r in kept] == ['b', 'c'] and dropped == ['a']
    assert rows == original
    with pytest.raises(ValueError):
        filter_rows(rows, {})
    with pytest.raises(ValueError):
        filter_rows(rows, {'a:r': {'recoverable': 'false', 'note': 'invalid'}})


def test_phase6_reservations_enforce_shared_cost_and_concurrency(tmp_path):
    config = load_config()
    config['paths']['phase6_output'] = tmp_path
    api = Phase6API(config, 100)
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda i: api.reserve('x', str(i), str(i), 1.), range(8)))
    assert sum(i is not None for i in ids) == 4
    assert api.accounting()['reserved_usd'] == 4
    with api.db() as db:
        db.execute("UPDATE calls SET status='completed', reserved=0, confirmed=8")
    with pytest.raises(CallBudgetExceeded):
        api.reserve('x', 'too_expensive', 'x', 4)


def test_filter_multiple_records_all_must_be_recoverable():
    row = {'episode_id': 'a', 'records': [
        {'op': 'G_DELETE_SUPPORT', 'record_id': '1'}, {'op': 'G_DELETE_SUPPORT', 'record_id': '2'}]}
    kept, dropped = filter_rows([row], {'a:1': {'recoverable': True, 'note': 'present'},
                                      'a:2': {'recoverable': False, 'note': 'missing'}})
    assert not kept and dropped == ['a']


def test_one_shot_alignment_keeps_moved_ids_and_does_not_use_hidden_source():
    from verak.v3.corrupt.document import Document, Paragraph, Unit
    from verak.v3.eval.measurement import align_rewrite, changes
    def doc(paragraphs):
        return Document([Paragraph(pid, [Unit(sid, text, [], ' ' if i else '')
            for i, (sid, text) in enumerate(units)]) for pid, units in paragraphs], ['', '\n'])
    original = doc([('P1', [('S1', '상어는 바다에 산다.'), ('S2', '바다 환경을 보호한다.')]),
                    ('P2', [('S3', '책을 읽고 감상을 기록한다.')])])
    rewrite = doc([('P1', [('S1', '책을 읽고 감상을 기록한다.')]),
                   ('P2', [('S2', '상어는 바다에 산다.'), ('S3', '바다 환경을 보호한다.')])])
    aligned, info = align_rewrite(original, rewrite)
    assert [p.pid for p in aligned.paragraphs] == ['P2', 'P1']
    assert [u.sid for u in aligned.units] == ['S3', 'S1', 'S2']
    assert aligned.text == rewrite.text and info['matched'] == 3
    assert changes(original, aligned)['changed_share'] == 1
    assert changes(original, original)['changed_share'] == 0


def test_paired_bootstrap_keeps_pairs_together_and_missing_level_is_not_zero():
    from verak.v3.eval.summary import bootstrap_difference, get_metric
    ci = bootstrap_difference([100., 2., 24., 9.], [99., 1., 23., 8.], ['L1', 'L1', 'L4', 'L4'])
    assert ci['difference'] == 1 and ci['ci95'] == [1, 1]
    row = {'reward': {'combined': {'per_record': [{'level': 'WORD', 'main': .5}]}}}
    assert get_metric(row, 'rec_SENTENCE') is None
    assert get_metric(row, 'rec_WORD') == .5
    row['reward']['combined']['per_record'] = [{'level': 'GLOBAL', 'main': 0., 'coupled': 1., 'recovery': .3}]
    assert get_metric(row, 'rec_GLOBAL') == 0
    assert get_metric(row, 'recovery_GLOBAL') == .3


def test_cost_api_cache_does_not_repeat_calls(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from verak.v3.eval import api as module
    calls = []
    usage = {'input_tokens': 100, 'output_tokens': 20,
             'input_tokens_details': {'cached_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 10}}
    class Adapter:
        def __init__(self, cfg):
            self._client = SimpleNamespace(responses=SimpleNamespace(create=self.create))
        def _load(self):
            pass
        def close(self):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='{}', status='completed', id='fake', model='gpt-6.1-sol',
                                   usage=SimpleNamespace(model_dump=lambda: usage))
    monkeypatch.setattr(module.socket, 'getaddrinfo', lambda *args: [])
    config = load_config()
    config['paths']['phase6_output'] = tmp_path
    api = Phase6API(config, 1, adapter_factory=Adapter)
    first = api.request([{'role': 'user', 'content': 'synthetic'}], stage='test', item_id='a')
    second = api.request([{'role': 'user', 'content': 'synthetic'}], stage='test', item_id='a')
    assert len(calls) == 1 and not first['replayed'] and second['replayed']
    assert api.accounting()['calls'] == 1
    assert api.accounting()['confirmed_usd'] == pytest.approx(.0004)
    assert api.accounting()['reserved_usd'] == 0
