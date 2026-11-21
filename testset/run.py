#!/usr/bin/env python3
"""Run the frozen test set using the shared validation experiment runner."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('validation_runner', HERE.parent / 'testset/runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE = HERE
runner.SET_NAME = 'testset'

if __name__ == '__main__':
    try:
        sys.exit(runner.main())
    except (RuntimeError, runner.subprocess.CalledProcessError) as exc:
        print(f'Test set failed: {exc}', file=sys.stderr)
        sys.exit(1)
