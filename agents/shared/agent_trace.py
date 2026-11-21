"""Append model requests, responses, and agent tool events to a run trace."""

import json
import threading
import time
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
        self.transient_failures = 0
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
            elif event == 'api_error' and data.get('transient'):
                self.transient_failures += 1
            elif event == 'api_response':
                if data.get('transient'):
                    self.transient_failures += 1
                body = data.get('body')
                usage = body.get('usage') if isinstance(body, dict) else None
                in_tok = usage.get('prompt_tokens') if isinstance(usage, dict) else None
                if not isinstance(in_tok, int) or isinstance(in_tok, bool):
                    in_tok = usage.get('input_tokens') if isinstance(usage, dict) else None
                out_tok = usage.get('completion_tokens') if isinstance(usage, dict) else None
                if not isinstance(out_tok, int) or isinstance(out_tok, bool):
                    out_tok = usage.get('output_tokens') if isinstance(usage, dict) else None
                if isinstance(in_tok, int) and not isinstance(in_tok, bool) and isinstance(out_tok, int) and not isinstance(out_tok, bool):
                    self.input_tokens += in_tok
                    self.output_tokens += out_tok
                    total = usage.get('total_tokens')
                    self.total_tokens += total if isinstance(total, int) and not isinstance(total, bool) else in_tok + out_tok
                elif 200 <= data.get('status', 0) < 300:
                    self.responses_without_usage += 1
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def metrics(self):
        with self.lock:
            return dict(api_calls=self.api_calls, input_tokens=self.input_tokens,
                        output_tokens=self.output_tokens, total_tokens=self.total_tokens,
                        responses_without_usage=self.responses_without_usage,
                        transient_failures=self.transient_failures)


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


API_SERVER_ERROR_EXIT_CODE = 75
DEFAULT_SERVER_ERROR_RETRIES = 3
DEFAULT_SERVER_ERROR_RETRY_INTERVAL = 120.0


class APIServerError(RuntimeError):
    """Raised when the model API keeps returning HTTP 5xx after all retries are exhausted."""

    is_api_server_error = True

    def __init__(self, status_code, retries, message=""):
        self.status_code = status_code
        self.retries = retries
        super().__init__(
            message or f"Model API returned HTTP {status_code} after {retries} retries; aborting task"
        )


class TraceHTTPClient(httpx.Client):
    """Record each actual SDK HTTP attempt, including retries and errors."""

    def __init__(
        self,
        recorder,
        max_calls=None,
        *,
        server_error_retries=DEFAULT_SERVER_ERROR_RETRIES,
        server_error_retry_interval=DEFAULT_SERVER_ERROR_RETRY_INTERVAL,
    ):
        super().__init__(timeout=180, follow_redirects=True)
        self.recorder = recorder
        self.max_calls = max_calls
        self.server_error_retries = max(0, int(server_error_retries))
        self.server_error_retry_interval = float(server_error_retry_interval)
        self._calls = 0
        self._transient_failures = 0
        self._calls_lock = threading.Lock()

    @property
    def calls_made(self):
        with self._calls_lock:
            return self._calls

    @property
    def stage_calls_made(self):
        with self._calls_lock:
            return self._calls - self._transient_failures

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
        request.read()
        for attempt in range(self.server_error_retries + 1):
            with self._calls_lock:
                self._calls += 1
            request_id = uuid.uuid4().hex
            if self.recorder:
                self.recorder.record('api_request', request_id=request_id, method=request.method,
                                     path=request.url.path, body=_body(request.content))
            try:
                response = super().send(request, **kwargs)
                response.read()
            except Exception as exc:
                transient = isinstance(exc, httpx.TransportError)
                if transient:
                    with self._calls_lock:
                        self._transient_failures += 1
                if self.recorder:
                    self.recorder.record('api_error', request_id=request_id, error=str(exc), transient=transient)
                raise
            transient = response.status_code in (408, 409, 429) or response.status_code >= 500
            if transient:
                with self._calls_lock:
                    self._transient_failures += 1
            if self.recorder:
                self.recorder.record('api_response', request_id=request_id, status=response.status_code,
                                     body=_body(response.content), transient=transient)
            if 500 <= response.status_code < 600 and self.server_error_retries > 0:
                if attempt < self.server_error_retries:
                    if self.recorder:
                        self.recorder.record(
                            'api_retry',
                            request_id=request_id,
                            status=response.status_code,
                            retry=attempt + 1,
                            max_retries=self.server_error_retries,
                            wait_seconds=self.server_error_retry_interval,
                        )
                    print(
                        f"[API 5xx] HTTP {response.status_code}; retrying in "
                        f"{self.server_error_retry_interval:g}s ({attempt + 1}/{self.server_error_retries})...",
                        flush=True,
                    )
                    time.sleep(self.server_error_retry_interval)
                    continue
                if self.recorder:
                    self.recorder.record(
                        'api_server_abort',
                        request_id=request_id,
                        status=response.status_code,
                        retries=self.server_error_retries,
                    )
                raise APIServerError(response.status_code, self.server_error_retries)
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
