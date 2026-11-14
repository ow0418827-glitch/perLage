import json
from types import SimpleNamespace

import pytest

from src.agent import shopping


def client_for(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def response(content='click[Buy Now]', *, reasoning='已确认当前商品符合需求。', details=None,
             finish='stop'):
    message = {'content': content, 'reasoning_content': reasoning}
    if details is not None:
        message['reasoning_details'] = details
    return {
        'choices': [{'message': message, 'finish_reason': finish}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18,
                  'completion_tokens_details': {'reasoning_tokens': 5}},
    }


def test_reasoning_is_saved_by_default_without_changing_action_or_request():
    requests = []
    traces = []
    for enabled in (True, False):
        trace = {}

        def create(**kwargs):
            requests.append(kwargs)
            return response()

        arguments = {} if enabled else {'save_reasoning': False}
        action = shopping.llm('fixed prompt', 'fixed model', client_for(create), 'minimax',
                              trace=trace, max_tokens=2048, max_retries=1, **arguments)
        assert action == 'click[Buy Now]'
        assert trace['RawResponse'] == 'click[Buy Now]'
        assert trace['ReasoningCaptureEnabled'] is enabled
        metadata = trace['ResponseAttempts'][0]
        assert metadata['ReasoningPresent'] is True
        assert metadata['ReasoningSaved'] is enabled
        assert metadata['Usage']['ReasoningTokens'] == 5
        if enabled:
            assert metadata['ReasoningContent'] == '已确认当前商品符合需求。'
        else:
            assert 'ReasoningContent' not in metadata
            assert 'ReasoningDetails' not in metadata
        traces.append(trace)
    assert requests[0] == requests[1]
    assert requests[0]['extra_body'] == {'reasoning_split': True}
    assert json.loads(json.dumps(traces, ensure_ascii=False)) == traces


def test_structured_sdk_reasoning_is_preserved_as_serializable_data():
    class ReasoningDetail:
        def model_dump(self, *, mode):
            assert mode == 'json'
            return {'type': 'reasoning.text', 'text': '先比较商品特点。', 'index': 0,
                    'metadata': {'provider': 'fixture', 'ids': [1, 2]}}

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='search[fixture]', reasoning_content=None,
            reasoning_details=[ReasoningDetail()]), finish_reason='stop')], usage=None)
    text, metadata = shopping._policy_response_metadata(completion, 1, 2048)
    assert text == 'search[fixture]'
    assert metadata['ReasoningDetails'] == [{
        'type': 'reasoning.text', 'text': '先比较商品特点。', 'index': 0,
        'metadata': {'provider': 'fixture', 'ids': [1, 2]}}]
    assert metadata['ReasoningPresent'] is True
    assert metadata['ReasoningSaved'] is True
    assert json.loads(json.dumps(metadata, ensure_ascii=False)) == metadata


def test_empty_action_retry_preserves_each_returned_reasoning(monkeypatch):
    trace = {}
    requests = []
    responses = iter([
        response('', reasoning='第一次已分析，但动作尚未返回。', finish='length'),
        response('search[fixture]', reasoning='第二次继续后返回动作。',
                 details=[{'type': 'reasoning.text', 'text': '核对检索词。'}]),
    ])

    def create(**kwargs):
        requests.append(kwargs)
        return next(responses)

    monkeypatch.setattr(shopping.time, 'sleep', lambda _: None)
    action = shopping.llm('fixed', 'fixed', client_for(create), 'minimax', trace=trace,
                          max_tokens=2048, max_retries=2)
    assert action == 'search[fixture]'
    attempts = trace['ResponseAttempts']
    assert len(attempts) == 2
    assert attempts[0]['ReasoningContent'] == '第一次已分析，但动作尚未返回。'
    assert attempts[0]['ErrorType'] == 'PolicyActionError'
    assert attempts[1]['ReasoningContent'] == '第二次继续后返回动作。'
    assert attempts[1]['ReasoningDetails'][0]['text'] == '核对检索词。'
    assert requests[0]['max_tokens'] == 2048
    assert requests[1]['max_tokens'] == 4096
    assert requests[0]['messages'] == requests[1]['messages']
    assert json.loads(json.dumps(trace, ensure_ascii=False)) == trace


def test_exhausted_empty_action_retains_reasoning_in_failure_trace(monkeypatch):
    monkeypatch.setattr(shopping.time, 'sleep', lambda _: None)
    trace = {}
    with pytest.raises(shopping.PolicyActionError) as caught:
        shopping.llm('fixed', 'fixed', client_for(lambda **_: response('', reasoning='失败前的返回正文。')),
                      'minimax', trace=trace, max_retries=1)
    assert caught.value.trace['ResponseAttempts'][0]['ReasoningContent'] == '失败前的返回正文。'


def test_absent_reasoning_is_recorded_without_inventing_text():
    _, metadata = shopping._policy_response_metadata(
        {'choices': [{'message': {'content': 'think[fixture]'}, 'finish_reason': 'stop'}]},
        1, 2048)
    assert metadata['ReasoningPresent'] is False
    assert metadata['ReasoningSaved'] is False
    assert 'ReasoningContent' not in metadata
    assert 'ReasoningDetails' not in metadata


@pytest.mark.parametrize('enabled', [True, False])
def test_episode_persists_reasoning_and_honors_explicit_opt_out(tmp_path, enabled):
    class FixtureEnvironment:
        sessions = {}

        def step(self, session, action):
            if action == 'reset':
                return 'Instruction: fixture request', 0, False, {'instruction': 'fixture request'}
            return 'Fixture finished', 1, True, {
                'instruction': 'fixture request', 'final_asin': 'fixture'}

    arguments = {} if enabled else {'policy_save_reasoning': False}
    _, record, _ = shopping.webshop_run_rap(
        'fixture', '', '', 'Fixture initial context', FixtureEnvironment(),
        client_for(lambda **_: response()), 'fixture model', 'minimax', None, [], {}, 1,
        benign=True, test=True, to_print=False, honest_reward=True, **arguments)
    path = tmp_path / 'episode.json'
    path.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    persisted = json.loads(path.read_text(encoding='utf-8'))
    trace = persisted['PolicyTrace'][0]
    assert trace['ReasoningCaptureEnabled'] is enabled
    assert trace['ResponseAttempts'][0]['ReasoningSaved'] is enabled
    if enabled:
        assert trace['ResponseAttempts'][0]['ReasoningContent'] == '已确认当前商品符合需求。'
    else:
        assert 'ReasoningContent' not in trace['ResponseAttempts'][0]
