#!/usr/bin/env python3
"""Prepare every real dataset declared in an experiment matrix.

Normal benchmark runs call the same preparation code automatically when an
artifact is missing.  Running this script ahead of time moves all downloads,
validation, normalization, and generated-data work out of the run startup.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_VENV = Path(
    os.environ.get("DLLM_DATA_VENV_DIR", PROJECT_ROOT / ".venvs" / "data")
).expanduser().resolve()
_INSIDE_DATA_VENV = "DLLM_PREPARE_DATA_IN_VENV"


def _venv_python(directory: Path) -> Path:
    windows = directory / "Scripts" / "python.exe"
    return windows if os.name == "nt" else directory / "bin" / "python"


def _run_in_data_venv() -> None:
    """Create/reuse the lightweight data venv, then restart this script in it."""
    if os.environ.get(_INSIDE_DATA_VENV) == "1":
        return
    python = _venv_python(DATA_VENV)
    install_environment = os.environ.copy()
    data_root = Path(os.environ.get("DLLM_DATA_ROOT", PROJECT_ROOT / ".data"))
    pip_cache = Path(
        os.environ.get("DLLM_PIP_CACHE_DIR", data_root / "pip-cache")
    ).expanduser().resolve()
    pip_cache.mkdir(parents=True, exist_ok=True)
    install_environment["PIP_CACHE_DIR"] = str(pip_cache)

    created = not python.is_file()
    if created:
        print(f"Creating data environment: {DATA_VENV}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(DATA_VENV)], check=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            cwd=PROJECT_ROOT,
            env=install_environment,
            check=True,
        )

    import_check = subprocess.run(
        [str(python), "-c", "import dllm_bench, yaml"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if import_check.returncode != 0:
        print(f"Installing data dependencies into: {DATA_VENV}", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-e", "."],
            cwd=PROJECT_ROOT,
            env=install_environment,
            check=True,
        )

    environment = os.environ.copy()
    environment[_INSIDE_DATA_VENV] = "1"
    os.execve(
        str(python),
        [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild matching cached artifacts")
    args = parser.parse_args()

    _run_in_data_venv()

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
