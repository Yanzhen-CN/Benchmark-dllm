#!/usr/bin/env python3
"""Manage the shared root venv for every non-model benchmark stage.

Model generation uses ``.venvs/<model>``. Data preparation, scoring,
visualization, and reporting all use ``.venvs/root`` through this module.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Direct execution sets sys.path[0] to venv_scripts/, while imports and
    # editable installation are rooted one directory above it.
    sys.path.insert(0, str(REPO_ROOT))

from venv_scripts._model_script import (
    bootstrap_python,
    installation_environment,
    shared_data_environment,
)


ROOT_VENV = Path(
    Path(os.environ.get("DLLM_VENV_ROOT", REPO_ROOT / ".venvs")) / "root"
).expanduser().resolve()
INSIDE_ROOT_VENV = "DLLM_ROOT_VENV_ACTIVE"


def venv_python(directory: Path | None = None) -> Path:
    directory = directory or ROOT_VENV
    windows = directory / "Scripts" / "python.exe"
    return windows if os.name == "nt" else directory / "bin" / "python"


def ensure_environment(*, recreate: bool = False) -> Path:
    """Create or repair the root environment after explicit confirmation."""
    python = venv_python()
    install_environment = installation_environment()
    venv_root = Path(
        os.environ.get("DLLM_VENV_ROOT", REPO_ROOT / ".venvs")
    ).expanduser().resolve()
    if ROOT_VENV.parent != venv_root or ROOT_VENV == venv_root:
        raise SystemExit(f"refusing to manage root venv outside {venv_root}: {ROOT_VENV}")
    if recreate and ROOT_VENV.exists():
        print(f"Recreating root environment: {ROOT_VENV}", flush=True)
        shutil.rmtree(ROOT_VENV)

    if not python.is_file():
        print(f"Creating root environment: {ROOT_VENV}", flush=True)
        subprocess.run([bootstrap_python(), "-m", "venv", str(ROOT_VENV)], check=True)
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


def require_environment() -> Path:
    """Return a complete root environment without modifying it."""
    python = venv_python()
    if not python.is_file():
        raise SystemExit(f"root environment is missing or broken: {ROOT_VENV}")
    import_check = subprocess.run(
        [str(python), "-c", "import dllm_bench, yaml, matplotlib"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if import_check.returncode != 0:
        raise SystemExit(f"root environment is incomplete: {ROOT_VENV}")
    return python


def _confirm_setup(problem: str) -> bool:
    print(problem, file=sys.stderr, flush=True)
    try:
        interactive = sys.stdin.isatty()
    except (AttributeError, OSError):
        interactive = False
    if not interactive:
        return False
    try:
        answer = input("Run setup for the root environment now? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print("Setup cancelled.", file=sys.stderr, flush=True)
        return False
    return answer.strip().lower() in {"y", "yes"}


def data_preparation_environment() -> Path:
    """Resolve `.venvs/root`, asking before setup during data preparation."""
    try:
        return require_environment()
    except SystemExit as error:
        problem = str(error)
        if not _confirm_setup(problem):
            raise SystemExit(f"{problem}; setup was not run") from None
        return ensure_environment()


def run_in_root_venv(script: str | Path, argv: Sequence[str]) -> None:
    """Replace the current process with ``script`` under `.venvs/root`."""
    if os.environ.get(INSIDE_ROOT_VENV) == "1":
        return
    python = data_preparation_environment()
    environment = shared_data_environment()
    environment[INSIDE_ROOT_VENV] = "1"
    script = Path(script).resolve()
    command = [str(python), str(script), *argv]
    if os.name == "nt":
        # CPython's os.execve() can crash while replacing a venv interpreter
        # on Windows (observed as 0xC0000005). Preserve the same isolation and
        # exit status through a child process; POSIX keeps true exec below.
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        raise SystemExit(completed.returncode)
    os.execve(
        str(python),
        command,
        environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="setup", choices=("setup", "check"))
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the managed root venv (setup only)",
    )
    args = parser.parse_args(argv)
    if args.recreate and args.action != "setup":
        raise SystemExit("--recreate is only valid with the setup action")
    python = ensure_environment(recreate=args.recreate)
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
