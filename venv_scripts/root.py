#!/usr/bin/env python3
"""Manage the shared root venv for every non-model benchmark stage.

Model generation uses ``.venvs/<model>``. Data preparation, scoring,
visualization, and reporting all use ``.venvs/root`` through this module.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_VENV = Path(
    os.environ.get("DLLM_ROOT_VENV_DIR", REPO_ROOT / ".venvs" / "root")
).expanduser().resolve()
INSIDE_ROOT_VENV = "DLLM_ROOT_VENV_ACTIVE"


def venv_python(directory: Path | None = None) -> Path:
    directory = directory or ROOT_VENV
    windows = directory / "Scripts" / "python.exe"
    return windows if os.name == "nt" else directory / "bin" / "python"


def ensure_environment() -> Path:
    python = venv_python()
    install_environment = os.environ.copy()
    data_root = Path(os.environ.get("DLLM_DATA_ROOT", REPO_ROOT / ".data"))
    pip_cache = Path(
        os.environ.get("DLLM_PIP_CACHE_DIR", data_root / "pip-cache")
    ).expanduser().resolve()
    pip_cache.mkdir(parents=True, exist_ok=True)
    install_environment["PIP_CACHE_DIR"] = str(pip_cache)

    if not python.is_file():
        print(f"Creating root environment: {ROOT_VENV}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(ROOT_VENV)], check=True)
        subprocess.run(
            [
                str(python), "-m", "pip", "install", "--upgrade",
                "pip", "setuptools", "wheel",
            ],
            cwd=REPO_ROOT,
            env=install_environment,
            check=True,
        )

    import_check = subprocess.run(
        [str(python), "-c", "import dllm_bench, yaml, matplotlib"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if import_check.returncode != 0:
        print(f"Installing root dependencies into: {ROOT_VENV}", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-e", "."],
            cwd=REPO_ROOT,
            env=install_environment,
            check=True,
        )
    return python


def run_in_root_venv(script: str | Path, argv: Sequence[str]) -> None:
    """Replace the current process with ``script`` under `.venvs/root`."""
    if os.environ.get(INSIDE_ROOT_VENV) == "1":
        return
    python = ensure_environment()
    environment = os.environ.copy()
    environment[INSIDE_ROOT_VENV] = "1"
    script = Path(script).resolve()
    os.execve(
        str(python),
        [str(python), str(script), *argv],
        environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="setup", choices=("setup", "check"))
    args = parser.parse_args(argv)
    python = ensure_environment()
    if args.action == "check":
        subprocess.run([str(python), "-m", "pip", "check"], cwd=REPO_ROOT, check=True)
        subprocess.run(
            [str(python), "-c", "import dllm_bench, yaml, matplotlib; print('root environment OK')"],
            cwd=REPO_ROOT,
            check=True,
        )
    else:
        print(f"Root environment ready: {ROOT_VENV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
