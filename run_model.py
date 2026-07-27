#!/usr/bin/env python3
"""Server entry point: generate every selected model's persisted outputs.

The caller may use the server's system Python. Each model is dispatched into
its own ``.venvs/<model>`` environment. No scoring or visualization runs here.
"""

from __future__ import annotations

import sys

import run_bench


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--stage" in arguments:
        raise SystemExit("run_model.py always runs the generate stage; remove --stage")
    if "--demo" not in arguments and "--real-data" not in arguments:
        arguments.append("--real-data")
    if "--measure-compute" not in arguments and "--no-measure-compute" not in arguments:
        arguments.append("--measure-compute")
    if "--require-all-metrics" not in arguments and "--allow-missing-metrics" not in arguments:
        arguments.append("--require-all-metrics")
    return run_bench.main([*arguments, "--stage", "generate"])


if __name__ == "__main__":
    raise SystemExit(main())
