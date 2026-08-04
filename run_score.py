#!/usr/bin/env python3
"""Local entry point: score transferred ``model_output`` artifacts."""

from __future__ import annotations

import sys

from venv_scripts.root import run_in_root_venv


def main() -> int:
    if not any(value in sys.argv[1:] for value in ("-h", "--help")):
        run_in_root_venv(__file__, sys.argv[1:])
    from dllm_bench.runner.local_pipeline import main as run_local_stage
    return run_local_stage("score", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
