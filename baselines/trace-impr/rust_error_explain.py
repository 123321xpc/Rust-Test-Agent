"""Look up compiler error explanations with the checkout's Rust toolchain."""

import re
import subprocess

from langchain_core.tools import tool


def make_rust_error_explain_tool(root, env=None):
    """Create the explanation tool, using the isolated checkout as cwd."""
    @tool
    def rust_error_explain(code: str) -> str:
        """Explain a Rust compiler error code, such as E0382, using rustc --explain.

        The Rust toolchain selected for the working copy must be installed.
        """
        code = code.strip().upper()
        if not re.fullmatch(r"E[0-9]{4}", code):
            return "Error: provide a Rust error code in the form E0382."
        try:
            result = subprocess.run(
                ["rustc", "--explain", code], cwd=root, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=30, env=env,
            )
        except subprocess.TimeoutExpired:
            return "Error: rustc --explain timed out after 30 seconds."
        except OSError as exc:
            return f"Error running rustc --explain: {exc}"
        if result.returncode:
            return f"Error explaining {code}: {result.stderr.strip() or result.stdout.strip() or 'rustc failed'}"
        return result.stdout

    return rust_error_explain
