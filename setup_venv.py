#!/usr/bin/env python3
"""Create or update one isolated venv per benchmark model.

This is a dispatcher only. Each ``venv_scripts/<model>.py`` owns that model's
venv path, dependency versions, installation and validation.

    python setup_venv.py
    python setup_venv.py -m illada
    python setup_venv.py -m dreamreasoner -m diffusiongemma
    python setup_venv.py --cuda-index cu126
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from run_bench import (
    DEFAULT_MATRIX,
    DEFAULT_VENV_SCRIPTS_DIR,
    dispatch_model_scripts,
    matrix_model_names,
    normalize_model_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", action="append", default=[], help="Model name; repeat or comma-separate (default: all matrix models)")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="Experiment matrix YAML")
    parser.add_argument("--venv-scripts-dir", dest="scripts_dir", default=str(DEFAULT_VENV_SCRIPTS_DIR), help="Directory containing per-model Python environment scripts")
    parser.add_argument("--cuda-index", default="cu124", choices=("cu118", "cu121", "cu124", "cu126"))
    parser.add_argument("--check", action="store_true", help="Run each model script's check action after setup")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix_path = Path(args.matrix).resolve()
    available = matrix_model_names(matrix_path)
    if args.list_models:
        print("\n".join(available))
        return 0
    try:
        selected = normalize_model_names(args.model, available)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    updates = {"CUDA_INDEX": args.cuda_index}
    dispatch_model_scripts(
        selected, action="setup", scripts_dir=args.scripts_dir,
        env_updates=updates, dry_run=args.dry_run,
    )
    if args.check:
        dispatch_model_scripts(
            selected, action="check", scripts_dir=args.scripts_dir,
            env_updates=updates, dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
