#!/usr/bin/env python3
"""Local entry point for optional pairwise resource sensitivity charts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from venv_scripts.root import run_in_root_venv


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", action="extend", nargs="+", required=True)
    parser.add_argument("--model-config", action="extend", nargs="+", default=[])
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("-d", "--dataset", action="extend", nargs="+", default=[])
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=50.0)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if argv is None and not any(value in arguments for value in ("-h", "--help")):
        run_in_root_venv(__file__, arguments)
    args = build_parser().parse_args(arguments)
    if not 0 <= args.beta <= 100:
        raise SystemExit("--beta must be in [0, 100]")
    if not 0 <= args.gamma <= 100:
        raise SystemExit("--gamma must be in [0, 100]")
    command = [
        sys.executable,
        "-m",
        "dllm_bench.cli",
        "pairwise-report",
        "--output-root",
        args.output_root,
        "--base-model",
        args.base_model,
        "--beta",
        str(args.beta),
        "--gamma",
        str(args.gamma),
    ]
    if args.base_config:
        command.extend(["--base-config", args.base_config])
    for model in args.model:
        for name in model.split(","):
            if name.strip():
                command.extend(["--model", name.strip()])
    for config in args.model_config:
        for name in config.split(","):
            if name.strip():
                command.extend(["--model-config", name.strip()])
    for dataset in args.dataset:
        for name in dataset.split(","):
            if name.strip():
                command.extend(["--dataset", name.strip()])
    print("- " + " ".join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
