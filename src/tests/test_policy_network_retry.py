from types import SimpleNamespace

import httpx
from openai import APIConnectionError
import pytest

from src.agent import shopping


def disconnected():
    return APIConnectionError(request=httpx.Request('POST','https://example.invalid/chat/completions'))


def client_for(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_transient_retry_waits_without_changing_prompt_or_model_request(monkeypatch):
    requests=[];delays=[];trace={}
    def create(**kwargs):
        requests.append(kwargs)
        if len(requests)<3:raise disconnected()
        return {'choices':[{'message':{'content':'click[Buy Now]'},'finish_reason':'stop'}]}
    monkeypatch.setattr(shopping.time,'sleep',delays.append)
    action=shopping.llm('fixed prompt','fixed model',client_for(create),'minimax',
                        max_tokens=2048,max_retries=3,trace=trace)
    assert action=='click[Buy Now]' and delays==[15,30]
    assert requests[0]==requests[1]==requests[2]
    assert [a['RetryDelaySeconds'] for a in trace['ResponseAttempts'][:-1]]==[15,30]
    assert trace['CallAttempts']==3


def test_transient_retry_remains_bounded_and_preserves_failure_evidence(monkeypatch):
    attempts=[];delays=[];trace={}
    def create(**kwargs):
        attempts.append(kwargs);raise disconnected()
    monkeypatch.setattr(shopping.time,'sleep',delays.append)
    with pytest.raises(APIConnectionError):
        shopping.llm('fixed','fixed',client_for(create),'minimax',max_retries=3,trace=trace)
    assert len(attempts)==3 and delays==[15,30]
    assert len(trace['ResponseAttempts'])==3
    assert all(a['ErrorType']=='APIConnectionError' for a in trace['ResponseAttempts'])


def test_nontransport_errors_keep_existing_retry_delay():
    assert shopping.policy_retry_delay(ValueError('bad input'),1)==1
    assert shopping.policy_retry_delay(disconnected(),10)==60
