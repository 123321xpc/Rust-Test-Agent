"""Shared module loader for rust_error_explain.py."""
from pathlib import Path

_SHARED_PATH = Path(__file__).resolve().parents[1] / "shared" / "rust_error_explain.py"
exec(compile(_SHARED_PATH.read_text(encoding="utf-8"), str(_SHARED_PATH), "exec"), globals())
