"""Durable cumulative spending and once-only request contracts, without live calls."""

import pytest
from concurrent.futures import ThreadPoolExecutor

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.corrupt.full_judging import CostLedger, CostUnavailable, request_bound, usage_nanos


def pilot():
    return {"calls":100, "cost_nanos":1_474_082_500, "ids":["pilot:x"]}


def test_pilot_cost_and_requests_are_cumulative(tmp_path):
    ledger = CostLedger(tmp_path/'ledger.json', pilot(), max_api_calls=102)
    assert ledger.snapshot()['cost_usd'] == pytest.approx(1.4740825)
    assert ledger.reserve('new:1',100_000_000)==101
    ledger.settle('new:1',{'status':'completed','usage':{'input_tokens':1000,'output_tokens':100}})
    assert ledger.snapshot()['cost_usd']==pytest.approx(1.4770825)
    assert ledger.reserve('new:2',100_000_000)==102
    with pytest.raises(CallBudgetExceeded): ledger.reserve('new:3',100_000_000)
    with pytest.raises(ValueError,match='already'): ledger.reserve('pilot:x',1)
    with pytest.raises(ValueError,match='already'): ledger.reserve('new:1',1)


def test_cost_reservations_include_inflight_and_survive_resume(tmp_path):
    p=pilot(); p['cost_nanos']=44_800_000_000
    ledger=CostLedger(tmp_path/'ledger.json',p,max_api_calls=3033)
    ledger.reserve('one',150_000_000)
    with pytest.raises(CostUnavailable): ledger.reserve('two',150_000_000)
    assert ledger.snapshot()['calls']==101
    record={'status':'completed','usage':{'input_tokens':1000,'output_tokens':100}}
    ledger.settle('one',record)
    ledger.settle('one',record)  # Crash replay must not double charge.
    resumed=CostLedger(tmp_path/'ledger.json',p,max_api_calls=3033)
    assert resumed.snapshot()['cost_usd']==pytest.approx(44.803)
    assert resumed.reserve('two',150_000_000)==102


def test_concurrent_workers_cannot_overreserve(tmp_path):
    p=pilot(); p['cost_nanos']=44_800_000_000
    ledger=CostLedger(tmp_path/'ledger.json',p,max_api_calls=3033)
    def reserve(i):
        try: return ledger.reserve(str(i),100_000_000)
        except CostUnavailable: return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(reserve,range(8)))
    assert len([r for r in results if r is not None])==2
    assert ledger.snapshot()['outstanding_nanos']==200_000_000


def test_unknown_usage_is_not_counted_as_free(tmp_path):
    ledger=CostLedger(tmp_path/'ledger.json',pilot(),max_api_calls=3033)
    ledger.reserve('x',100_000_000)
    ledger.settle('x',{'status':'error'})
    state=ledger.snapshot()
    assert state['cost_usd']==pytest.approx(1.5740825)
    assert state['confirmed_cost_usd']==pytest.approx(1.4740825)
    assert state['unknown_cost_upper_usd']==pytest.approx(.1)
    assert state['entries']['x']['cost_known'] is False


def test_bound_includes_entire_output_and_cache_write_cost():
    bound,byte_count=request_bound({'sentence':'실험용 합성 문장이다.'})
    assert bound==(byte_count+4096)*2500+8192*10000
    assert bound > usage_nanos({'input_tokens':byte_count,'output_tokens':8192,
                               'input_tokens_details':{'cache_write_tokens':byte_count}})


@pytest.mark.parametrize('calls,cost',[(3034,45),(99,45),(3033,46)])
def test_approval_cannot_be_silently_extended(tmp_path,calls,cost):
    with pytest.raises(ValueError):
        CostLedger(tmp_path/'ledger.json',pilot(),max_api_calls=calls,max_cost_usd=cost)
