"""Offline tests for model API and repair trajectory recording."""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent_trace import APIServerError, TraceHTTPClient, TraceRecorder


def entries(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_callbacks_record_model_and_tool_trajectory_and_redact(tmp_path):
    path = tmp_path / 'nested' / 'run.jsonl'
    recorder = TraceRecorder(path, secrets=['secret-key'])
    model_run, tool_run = uuid.uuid4(), uuid.uuid4()
    recorder.on_chat_model_start({}, [[HumanMessage(content='fix code')]], run_id=model_run,
                                 invocation_params={'model': 'example', 'api_key': 'secret-key'})
    recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content='inspect file'))]]),
                        run_id=model_run)
    recorder.on_tool_start({'name': 'read_file'}, 'src/lib.rs', run_id=tool_run,
                           parent_run_id=model_run, inputs={'path': 'src/lib.rs'})
    recorder.on_tool_end('file contains secret-key', run_id=tool_run)
    recorder.on_tool_error(ValueError('secret-key error'), run_id=tool_run)
    recorder.on_llm_error(RuntimeError('request failed'), run_id=model_run)

    events = entries(path)
    assert [event['event'] for event in events] == [
        'model_start', 'model_end', 'tool_start', 'tool_end', 'tool_error', 'model_error'
    ]
    assert events[0]['messages'][0][0]['content'] == 'fix code'
    assert events[0]['invocation_params']['api_key'] == '[REDACTED]'
    assert events[2]['input']['path'] == 'src/lib.rs'
    assert events[2]['parent_run_id'] == str(model_run)
    assert events[3]['output'] == 'file contains [REDACTED]'
    assert 'secret-key' not in path.read_text(encoding='utf-8')


def test_each_http_attempt_records_request_response_and_failure(tmp_path, monkeypatch):
    path = tmp_path / 'run.jsonl'
    recorder = TraceRecorder(path, secrets=['secret-key'])
    client = TraceHTTPClient(recorder)
    seen = []

    def send(_client, request, **kwargs):
        seen.append(request)
        if len(seen) == 2:
            raise httpx.ConnectError('connection failed for secret-key', request=request)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'fixed'}}]}, request=request)

    monkeypatch.setattr(httpx.Client, 'send', send)
    request = client.build_request('POST', 'https://example.test/v1/chat/completions?token=secret-key',
                                   headers={'Authorization': 'Bearer secret-key'},
                                   json={'messages': [{'role': 'user', 'content': 'repair'}]})
    assert client.send(request).json()['choices'][0]['message']['content'] == 'fixed'
    with pytest.raises(httpx.ConnectError):
        client.send(request)
    client.close()

    events = entries(path)
    assert [event['event'] for event in events] == ['api_request', 'api_response', 'api_request', 'api_error']
    assert events[0]['body']['messages'][0]['content'] == 'repair'
    assert events[0]['path'] == '/v1/chat/completions'
    assert events[1]['body']['choices'][0]['message']['content'] == 'fixed'
    assert events[0]['request_id'] == events[1]['request_id']
    assert events[2]['request_id'] == events[3]['request_id']
    assert events[0]['request_id'] != events[2]['request_id']
    assert 'secret-key' not in path.read_text(encoding='utf-8')


def test_http_trace_against_local_server(tmp_path, monkeypatch):
    for name in ('ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy'):
        monkeypatch.delenv(name, raising=False)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers['Content-Length']))
            assert json.loads(payload)['model'] == 'sample'
            body = json.dumps({'choices': [{'message': {'content': 'answer'}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / 'network.jsonl'
    try:
        with TraceHTTPClient(TraceRecorder(path)) as client:
            response = client.post(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                   json={'model': 'sample', 'messages': [{'content': 'issue'}]})
            assert response.json()['choices'][0]['message']['content'] == 'answer'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    request, response = entries(path)
    assert request['body']['model'] == 'sample'
    assert response['body']['choices'][0]['message']['content'] == 'answer'


def test_chatopenai_records_sdk_call(tmp_path):
    path = tmp_path / 'sdk.jsonl'
    recorder = TraceRecorder(path, secrets=['secret-key'])
    client = TraceHTTPClient(recorder)

    def respond(request):
        assert request.headers['authorization'] == 'Bearer secret-key'
        return httpx.Response(200, request=request, json={
            'id': 'completion-1', 'object': 'chat.completion', 'created': 1,
            'model': 'sample', 'choices': [{'index': 0, 'finish_reason': 'stop',
                                             'message': {'role': 'assistant', 'content': 'answer'}}],
        })

    client._transport = httpx.MockTransport(respond)
    client._mounts = {}
    model = ChatOpenAI(model='sample', api_key='secret-key', base_url='https://example.test/v1',
                       http_client=client, max_retries=0)
    try:
        response = model.invoke('issue', config={'callbacks': [recorder]})
    finally:
        client.close()
    assert response.content == 'answer'
    events = entries(path)
    assert {event['event'] for event in events} >= {'model_start', 'api_request', 'api_response', 'model_end'}
    request = next(event for event in events if event['event'] == 'api_request')
    assert request['body']['messages'][0]['content'] == 'issue'
    assert 'secret-key' not in path.read_text(encoding='utf-8')


def test_sdk_retry_is_recorded_as_two_http_attempts(tmp_path, monkeypatch):
    path = tmp_path / 'retry.jsonl'
    recorder = TraceRecorder(path)
    client = TraceHTTPClient(recorder)
    attempts = []

    def respond(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(500, request=request, json={'error': {'message': 'retry'}})
        return httpx.Response(200, request=request, json={
            'id': 'completion-1', 'object': 'chat.completion', 'created': 1,
            'model': 'sample', 'choices': [{'index': 0, 'finish_reason': 'stop',
                                             'message': {'role': 'assistant', 'content': 'answer'}}],
        })

    client._transport = httpx.MockTransport(respond)
    client._mounts = {}
    model = ChatOpenAI(model='sample', api_key='key', base_url='https://example.test/v1',
                       http_client=client, max_retries=1)
    monkeypatch.setattr('time.sleep', lambda *_: None)
    try:
        assert model.invoke('issue', config={'callbacks': [recorder]}).content == 'answer'
    finally:
        client.close()
    events = entries(path)
    requests = [event for event in events if event['event'] == 'api_request']
    responses = [event for event in events if event['event'] == 'api_response']
    assert len(requests) == 2
    assert [event['status'] for event in responses] == [500, 200]
    assert [event['request_id'] for event in requests] == [event['request_id'] for event in responses]


def test_api_limit_stops_additional_http_attempts(tmp_path):
    path = tmp_path / 'limit.jsonl'
    client = TraceHTTPClient(TraceRecorder(path), max_calls=2)
    calls = []
    client._transport = httpx.MockTransport(lambda request: (calls.append(request) or httpx.Response(200, json={})))
    client._mounts = {}
    try:
        client.post('https://example.test/v1/chat/completions', json={})
        client.post('https://example.test/v1/chat/completions', json={})
        with pytest.raises(RuntimeError, match='call limit'):
            client.post('https://example.test/v1/chat/completions', json={})
    finally:
        client.close()
    assert len(calls) == 2
    assert [e['event'] for e in entries(path)].count('api_request') == 2
    assert entries(path)[-1]['event'] == 'api_limit'


def test_metrics_count_retries_and_provider_usage(tmp_path):
    recorder = TraceRecorder(tmp_path / 'metrics.jsonl')
    client = TraceHTTPClient(recorder, server_error_retries=0)
    responses = iter([
        httpx.Response(500, json={'error': 'retry'}),
        httpx.Response(200, json={'usage': {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17}}),
        httpx.Response(200, json={'choices': []}),
        httpx.Response(200, json={'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': None}}),
        httpx.Response(200, json={'usage': {'input_tokens': 4, 'output_tokens': 6}}),
    ])
    client._transport = httpx.MockTransport(lambda request: next(responses))
    client._mounts = {}
    try:
        for _ in range(5):
            client.post('https://example.test/v1/chat/completions', json={})
    finally:
        client.close()
    assert recorder.metrics() == {'api_calls': 5, 'input_tokens': 18, 'output_tokens': 14,
                                  'total_tokens': 32, 'responses_without_usage': 1,
                                  'transient_failures': 1}


def test_5xx_retries_every_two_minutes_and_recovers(tmp_path, monkeypatch):
    path = tmp_path / 'retry_5xx.jsonl'
    recorder = TraceRecorder(path)
    client = TraceHTTPClient(recorder)
    sleeps = []
    monkeypatch.setattr('time.sleep', sleeps.append)
    responses = iter([
        httpx.Response(502, json={'error': 'bad gateway'}),
        httpx.Response(503, json={'error': 'unavailable'}),
        httpx.Response(504, json={'error': 'gateway timeout'}),
        httpx.Response(200, json={'choices': [{'message': {'content': 'recovered'}}]}),
    ])
    client._transport = httpx.MockTransport(lambda request: next(responses))
    client._mounts = {}
    try:
        res = client.post('https://example.test/v1/chat/completions', json={'messages': []})
        assert res.status_code == 200
        assert res.json()['choices'][0]['message']['content'] == 'recovered'
    finally:
        client.close()
    assert sleeps == [120.0, 120.0, 120.0]
    events = entries(path)
    assert [e['event'] for e in events].count('api_request') == 4
    assert [e['status'] for e in events if e['event'] == 'api_retry'] == [502, 503, 504]


def test_5xx_aborts_after_three_retries_without_sdk_double_retry(tmp_path, monkeypatch):
    path = tmp_path / 'abort_5xx.jsonl'
    recorder = TraceRecorder(path)
    client = TraceHTTPClient(recorder)
    sleeps = []
    monkeypatch.setattr('time.sleep', sleeps.append)
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(503, request=request, json={'error': {'message': 'down'}})

    client._transport = httpx.MockTransport(respond)
    client._mounts = {}
    model = ChatOpenAI(model='sample', api_key='key', base_url='https://example.test/v1',
                       http_client=client, max_retries=3)
    try:
        with pytest.raises(APIServerError) as exc_info:
            model.invoke('issue', config={'callbacks': [recorder]})
        assert exc_info.value.is_api_server_error is True
        assert exc_info.value.status_code == 503
    finally:
        client.close()
    assert sleeps == [120.0, 120.0, 120.0]
    assert len(attempts) == 4
    events = entries(path)
    assert [e['event'] for e in events].count('api_request') == 4
    assert any(e['event'] == 'api_server_abort' and e['status'] == 503 and e['retries'] == 3 for e in events)
