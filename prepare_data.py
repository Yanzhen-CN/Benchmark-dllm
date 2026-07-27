#!/usr/bin/env python3
"""Prepare every real dataset declared in an experiment matrix.

Normal benchmark runs call the same preparation code automatically when an
artifact is missing.  Running this script ahead of time moves all downloads,
validation, normalization, and generated-data work out of the run startup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from venv_scripts.root import REPO_ROOT, run_in_root_venv


PROJECT_ROOT = REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild matching cached artifacts")
    args = parser.parse_args()

    run_in_root_venv(__file__, sys.argv[1:])

    from dllm_bench.runner.data_preparation import (
        DataPreparationError,
        prepare_matrix_datasets,
    )

    try:
        prepared = prepare_matrix_datasets(args.experiment_config, force=args.force)
    except DataPreparationError as exc:
        raise SystemExit(str(exc)) from exc
    for item in prepared:
        action = "prepared" if item.prepared_now else "cached"
        print(f"[{item.dataset_name}] {action}: {item.sample_count} samples -> {item.samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
