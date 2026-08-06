#!/usr/bin/env python3
"""Create or update one isolated venv per benchmark model.

This is a dispatcher only. Each ``venv_scripts/<model>.py`` owns that model's
venv path, dependency versions, installation and validation.

    python setup_venv.py
    python setup_venv.py -m illada
    python setup_venv.py -m llada2_1
    python setup_venv.py -m illada_vargen
    python setup_venv.py -m qwen3_8b
    python setup_venv.py -m dreamreasoner -m diffusiongemma
    python setup_venv.py -m gemma
    python setup_venv.py -m gemma_dflash
    python setup_venv.py --cuda-index cu126
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from run_bench import (
    DEFAULT_MATRIX,
    DEFAULT_VENV_SCRIPTS_DIR,
    dispatch_model_scripts,
    matrix_model_names,
    normalize_model_names,
)
from venv_scripts._model_script import PROFILES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", action="append", default=[], help="Model name; repeat or comma-separate (default: all matrix models)")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="Experiment matrix YAML")
    parser.add_argument("--venv-scripts-dir", dest="scripts_dir", default=str(DEFAULT_VENV_SCRIPTS_DIR), help="Directory containing per-model Python environment scripts")
    parser.add_argument("--cuda-index", default="cu124", choices=("cu118", "cu121", "cu124", "cu126"))
    parser.add_argument("--check", action="store_true", help="Run each model script's check action after setup")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of selected model environments to install concurrently",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the selected root/model environments",
    )
    parser.add_argument(
        "--no-root",
        action="store_true",
        help="Do not create/check the shared .venvs/root environment",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    matrix_path = Path(args.matrix).resolve()
    matrix_models = matrix_model_names(matrix_path)
    available = list(dict.fromkeys([*matrix_models, *PROFILES]))
    if args.list_models:
        print("\n".join(available))
        return 0
    try:
        selected = (
            normalize_model_names(args.model, available)
            if args.model
            else matrix_models
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    updates = {"CUDA_INDEX": args.cuda_index}
    action_args = ["--recreate"] if args.recreate else []
    if not args.no_root:
        root_script = Path(args.scripts_dir).resolve() / "root.py"
        root_command = [sys.executable, str(root_script), "setup", *action_args]
        print(f"[root] {' '.join(root_command)}", flush=True)
        if not args.dry_run:
            subprocess.run(root_command, cwd=Path(__file__).resolve().parent, check=True)
    dispatch_model_scripts(
        selected, action="setup", scripts_dir=args.scripts_dir,
        env_updates=updates, action_args=action_args, jobs=args.jobs,
        dry_run=args.dry_run,
    )
    if args.check:
        dispatch_model_scripts(
            selected, action="check", scripts_dir=args.scripts_dir,
            env_updates=updates, dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
