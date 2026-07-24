#!/usr/bin/env python3
"""Runs the dllm-bench test suite. Extra args pass straight through to pytest.

    python run_tests.py
    python run_tests.py -k gsm8k
    python run_tests.py tests/test_registry.py -v
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *sys.argv[1:]])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
