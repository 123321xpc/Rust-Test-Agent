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
