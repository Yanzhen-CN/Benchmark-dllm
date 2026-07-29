#!/usr/bin/env python3
"""Prepare every environment, dataset, and model required by a matrix.

This is the one-shot server entry point. It may be launched with the system
Python and performs, in order:

1. create/check ``.venvs/root``;
2. create/check every selected ``.venvs/<model>``;
3. prepare every selected dataset under ``data/datasets/prepared``;
4. download every selected HF checkpoint under ``data/huggingface``.

No model is loaded onto CPU/GPU and no generation forward pass is executed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"


def _flatten(values: Sequence[Sequence[str]]) -> list[str]:
    result: list[str] = []
    for group in values:
        for value in group:
            result.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(result))


def _run(command: list[str], *, dry_run: bool) -> None:
    print(f"- {' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    python = sys.executable
    matrix = str(Path(args.matrix).resolve())
    models = _flatten(args.model)
    datasets = _flatten(args.dataset)

    root_command = [python, str(PROJECT_ROOT / "venv_scripts" / "root.py"), "setup"]
    setup_command = [
        python,
        str(PROJECT_ROOT / "setup_venv.py"),
        "--matrix",
        matrix,
        "--cuda-index",
        args.cuda_index,
    ]
    model_command = [
        python,
        str(PROJECT_ROOT / "prepare_model.py"),
        "--matrix",
        matrix,
    ]
    for model in models:
        setup_command.extend(["-m", model])
        model_command.extend(["-m", model])

    data_command = [
        python,
        str(PROJECT_ROOT / "prepare_data.py"),
        "--experiment-config",
        matrix,
    ]
    if datasets:
        data_command.extend(["-d", *datasets])
    if args.force_data:
        data_command.append("--force")

    commands = [root_command, setup_command, data_command, model_command]
    if args.skip_data:
        commands.remove(data_command)
    if args.skip_models:
        commands.remove(model_command)
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        nargs="+",
        default=[],
        help="Only prepare these model(s); repeat, space-separate, or comma-separate",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="append",
        nargs="+",
        default=[],
        help="Only prepare these dataset(s); repeat, space-separate, or comma-separate",
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument(
        "--cuda-index",
        default="cu124",
        choices=("cu118", "cu121", "cu124", "cu126"),
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Rebuild selected prepared dataset artifacts even when cached",
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the four preparation commands without executing them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Matrix: {Path(args.matrix).resolve()}")
    print("Preparing root environment, model environments, datasets, and checkpoints ...")
    for command in build_commands(args):
        _run(command, dry_run=args.dry_run)
    print("Preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
