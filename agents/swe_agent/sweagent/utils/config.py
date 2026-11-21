from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Locate the shared repository configuration independently of the working directory.
for parent in Path(__file__).resolve().parents:
    if (parent / "pyproject.toml").is_file() or (parent / ".env").is_file():
        load_dotenv(parent / ".env")
        break
from sweagent import REPO_ROOT
from sweagent.utils.log import get_logger

logger = get_logger("config")


def convert_path_to_abspath(path: Path | str) -> Path:
    """If path is not absolute, convert it to an absolute path
    using the SWE_AGENT_CONFIG_ROOT environment variable (if set) or
    REPO_ROOT as base.
    """
    path = Path(path)
    root = Path(keys_config.get("SWE_AGENT_CONFIG_ROOT", REPO_ROOT))
    assert root.is_dir()
    if not path.is_absolute():
        path = root / path
    assert path.is_absolute()
    return path.resolve()


def convert_paths_to_abspath(paths: list[Path | str]) -> list[Path]:
    return [convert_path_to_abspath(p) for p in paths]


class Config:
    def __init__(self, *, keys_cfg_path: Path | None = None):
        """Read settings from the environment, populated by the root .env file."""
        pass

    def get(self, key: str, default=None, choices: list[Any] | None = None) -> Any:
        """Get a key from the environment.

        Args:
            key: The key to retrieve.
            default: The default value to return if the key is not found.
            choices: If provided, the value must be one of the choices.
        """

        def check_choices(value):
            if choices is not None and value not in choices:
                msg = f"Value {value} for key {key} not in {choices}"
                raise ValueError(msg)
            return value

        if key in os.environ:
            return check_choices(os.environ[key])
        return check_choices(default)

    def __getitem__(self, key: str) -> Any:
        if key in os.environ:
            return os.environ[key]
        msg = f"Key {key} not found in environment variables"
        raise KeyError(msg)

    def __contains__(self, key: str) -> bool:
        return key in os.environ


keys_config = Config()
