#!/usr/bin/env python3
"""Prepare every real dataset declared in an experiment matrix.

Normal benchmark runs call the same preparation code automatically when an
artifact is missing.  Running this script ahead of time moves all downloads,
validation, normalization, and generated-data work out of the run startup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from venv_scripts.root import REPO_ROOT, run_in_root_venv


PROJECT_ROOT = REPO_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="extend",
        nargs="+",
        default=[],
        help="Dataset name(s) to prepare, e.g. -d sudoku ruler (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild matching cached artifacts")
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Prepare Sudoku4/Sudoku9 with their original reasoning prompts",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)

    os.environ["DLLM_BENCH_ENABLE_REASONING"] = (
        "1" if args.enable_reasoning else "0"
    )

    run_in_root_venv(__file__, arguments)

    from dllm_bench.runner.data_preparation import (
        DataPreparationError,
        prepare_matrix_datasets,
    )

    dataset_names = [
        part.strip()
        for value in args.dataset
        for part in value.split(",")
        if part.strip()
    ]
    try:
        prepared = prepare_matrix_datasets(
            args.experiment_config,
            force=args.force,
            dataset_names=dataset_names,
        )
    except DataPreparationError as exc:
        raise SystemExit(str(exc)) from exc
    for item in prepared:
        action = "prepared" if item.prepared_now else "cached"
        print(f"[{item.dataset_name}] {action}: {item.sample_count} samples -> {item.samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
