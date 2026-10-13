import httpx
import pytest

from verak.v5 import api, finish
from verak.v5.common import read_json, write_json


def test_metadata_refresh_preserves_failure_and_never_uses_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(finish,'ROOT',tmp_path)
    monkeypatch.setattr(api,'credentials',lambda:{'OPENAI_API_KEY':'test-only','ANTHROPIC_API_KEY':'test-only'})
    write_json(tmp_path/'api_contract.json',{'user_approved':True,
        'luna':{'model':'test-luna'},'claude':{'model':'test-claude'}})
    calls=[];unauthorized=[True];client_type=httpx.Client
    def respond(request):
        calls.append(request)
        assert request.method=='GET' and '/models/' in request.url.path
        if unauthorized[0] and request.url.host=='api.openai.com':return httpx.Response(401)
        return httpx.Response(200,json={'id':request.url.path.rsplit('/',1)[-1]})
    monkeypatch.setattr(httpx,'Client',lambda **kw:client_type(transport=httpx.MockTransport(respond),**kw))
    with pytest.raises(RuntimeError,match='not available'):finish.api_access()
    failure=read_json(tmp_path/'api_model_access.json')
    with pytest.raises(RuntimeError,match='Prior'):finish.api_access()
    assert len(calls)==2
    unauthorized[0]=False
    result=finish.api_access(refresh=True)
    assert result['all_available'] and len(calls)==4 and result['paid_generation_requests']==0
    archives=list((tmp_path/'api_model_access_history').glob('*.json'))
    assert len(archives)==1 and read_json(archives[0])==failure
    assert finish.api_access()==result and len(calls)==4
