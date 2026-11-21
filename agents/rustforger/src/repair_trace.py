"""Incremental, per-attempt RustForger repair records."""
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class RepairTrace:
    def __init__(self, log_dir, instance_id, model):
        self.run_id = uuid.uuid4().hex
        self.directory = Path(log_dir) / "runs" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "trace.jsonl"
        self.instance_id = instance_id
        self.model = model
        self.secrets = tuple(v for v in (os.getenv("API_KEY"), os.getenv("GITHUB_TOKEN")) if v)
        self.lock = threading.Lock()
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.responses_without_usage = 0
        self.record("run_start", instance_id=instance_id, model=model, run_id=self.run_id)

    def clean(self, value):
        if isinstance(value, dict):
            return {str(k): ("[REDACTED]" if str(k).lower() in {"api_key", "authorization", "github_token"} else self.clean(v)) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        return value

    def record(self, event, **data):
        entry = self.clean({"event": event, "time": datetime.now(timezone.utc).isoformat(), **data})
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def request(self, call):
        with self.lock:
            self.api_calls = max(self.api_calls, call)
        self.record("api_request", api_call=call)

    def response(self, status, usage):
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens")) if isinstance(usage, dict) else None
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens")) if isinstance(usage, dict) else None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total = usage.get("total_tokens")
            if not isinstance(total, int):
                total = input_tokens + output_tokens
            with self.lock:
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
                self.total_tokens += total
        elif 200 <= status < 300:
            with self.lock:
                self.responses_without_usage += 1
        self.record("api_response", status=status, usage=usage)

    def metrics(self):
        with self.lock:
            return {"api_calls": self.api_calls, "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens, "total_tokens": self.total_tokens,
                    "responses_without_usage": self.responses_without_usage}
