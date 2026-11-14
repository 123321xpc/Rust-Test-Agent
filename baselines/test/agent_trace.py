"""Append model requests, responses, and agent tool events to a run trace."""

import json
import threading
import uuid
from datetime import datetime, timezone

import httpx
from langchain_core.callbacks import BaseCallbackHandler


class TraceRecorder(BaseCallbackHandler):
    raise_error = True
    run_inline = True

    def __init__(self, path, secrets=()):
        self.path = path
        self.secrets = tuple(s for s in secrets if s)
        self.lock = threading.Lock()
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.responses_without_usage = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(self, value):
        if hasattr(value, 'model_dump'):
            value = value.model_dump(mode='json')
        if isinstance(value, dict):
            return {str(k): ('[REDACTED]' if str(k).lower() in
                    {'api_key', 'authorization', 'password', 'access_token'} else self._clean(v))
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, '[REDACTED]')
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self._clean(str(value))

    def record(self, event, **data):
        entry = self._clean({'event': event, 'time': datetime.now(timezone.utc).isoformat(), **data})
        with self.lock:
            if event == 'api_request':
                self.api_calls += 1
            elif event == 'api_response':
                body = data.get('body')
                usage = body.get('usage') if isinstance(body, dict) else None
                if isinstance(usage, dict) and isinstance(usage.get('prompt_tokens'), int) and isinstance(usage.get('completion_tokens'), int):
                    self.input_tokens += usage['prompt_tokens']
                    self.output_tokens += usage['completion_tokens']
                    total = usage.get('total_tokens')
                    self.total_tokens += total if isinstance(total, int) else usage['prompt_tokens'] + usage['completion_tokens']
                elif 200 <= data.get('status', 0) < 300:
                    self.responses_without_usage += 1
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def metrics(self):
        with self.lock:
            return dict(api_calls=self.api_calls, input_tokens=self.input_tokens,
                        output_tokens=self.output_tokens, total_tokens=self.total_tokens,
                        responses_without_usage=self.responses_without_usage)


    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self.record('model_start', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None,
                    messages=messages, invocation_params=kwargs.get('invocation_params', {}))

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self.record('model_end', run_id=str(run_id), response=response)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self.record('model_error', run_id=str(run_id), error=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        self.record('tool_start', run_id=str(run_id), parent_run_id=str(parent_run_id) if parent_run_id else None,
                    tool=serialized.get('name'), input=inputs if inputs is not None else input_str)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self.record('tool_end', run_id=str(run_id), output=output)

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self.record('tool_error', run_id=str(run_id), error=str(error))


def _body(data):
    text = data.decode('utf-8', errors='replace')
    try:
        return json.loads(text)
    except ValueError:
        return text


class TraceHTTPClient(httpx.Client):
    """Record each actual SDK HTTP attempt, including retries and errors."""

    def __init__(self, recorder, max_calls=None):
        super().__init__(timeout=180, follow_redirects=True)
        self.recorder = recorder
        self.max_calls = max_calls
        self._calls = 0
        self._calls_lock = threading.Lock()

    @property
    def calls_made(self):
        with self._calls_lock:
            return self._calls

    @property
    def remaining_calls(self):
        with self._calls_lock:
            return None if self.max_calls is None else max(0, self.max_calls - self._calls)

    def send(self, request, **kwargs):
        with self._calls_lock:
            if self.max_calls is not None and self._calls >= self.max_calls:
                if self.recorder:
                    self.recorder.record('api_limit', max_calls=self.max_calls)
                raise RuntimeError(f'Model API call limit reached ({self.max_calls})')
            self._calls += 1
        request_id = uuid.uuid4().hex
        if self.recorder:
            self.recorder.record('api_request', request_id=request_id, method=request.method,
                                 path=request.url.path, body=_body(request.content))
        try:
            response = super().send(request, **kwargs)
            response.read()
        except Exception as exc:
            if self.recorder:
                self.recorder.record('api_error', request_id=request_id, error=str(exc))
            raise
        if self.recorder:
            self.recorder.record('api_response', request_id=request_id, status=response.status_code,
                                 body=_body(response.content))
        return response


class RoleTraceCallback(BaseCallbackHandler):
    """Forward LangChain callback events to TraceRecorder tagged with agent_role."""

    raise_error = True
    run_inline = True

    def __init__(self, recorder: TraceRecorder, role: str, attempt: int = 1):
        self.recorder = recorder
        self.role = role
        self.attempt = attempt

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self.recorder.record(
            'model_start',
            agent_role=self.role,
            attempt=self.attempt,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            messages=messages,
            invocation_params=kwargs.get('invocation_params', {}),
        )

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self.recorder.record('model_end', agent_role=self.role, attempt=self.attempt, run_id=str(run_id), response=response)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self.recorder.record('model_error', agent_role=self.role, attempt=self.attempt, run_id=str(run_id), error=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        self.recorder.record(
            'tool_start',
            agent_role=self.role,
            attempt=self.attempt,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tool=serialized.get('name'),
            input=inputs if inputs is not None else input_str,
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self.recorder.record('tool_end', agent_role=self.role, attempt=self.attempt, run_id=str(run_id), output=output)

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self.recorder.record('tool_error', agent_role=self.role, attempt=self.attempt, run_id=str(run_id), error=str(error))
