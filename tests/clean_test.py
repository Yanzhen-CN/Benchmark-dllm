#!/usr/bin/env python3
"""Remove local test and Python cache artifacts without touching benchmark data."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def native_remove(path: Path) -> tuple[bool, OSError | None]:
    """Remove an empty directory with the Windows API as a locked-file fallback."""

    if os.name != "nt":
        return False, None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    remove_directory = kernel32.RemoveDirectoryW
    remove_directory.argtypes = [ctypes.c_wchar_p]
    remove_directory.restype = ctypes.c_int

    ctypes.set_last_error(0)
    if remove_directory(str(path)):
        return True, None

    error = ctypes.WinError(ctypes.get_last_error())
    error.filename = str(path)
    return False, error


def remove_target(path: Path, *, use_native_fallback: bool = True) -> tuple[int, int]:
    """Remove one known temporary target and report (removed, skipped)."""

    if not path.exists() and not path.is_symlink():
        return 0, 1

    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return 1, 0
    except OSError as error:
        last_error: OSError = error
        if use_native_fallback and path.is_dir():
            removed, native_error = native_remove(path)
            if removed:
                return 1, 0
            if native_error is not None:
                last_error = native_error
        print(f"Could not remove {path}; last error: {last_error}", file=sys.stderr)
        return 0, 0


def find_targets(root: Path = PROJECT_ROOT) -> list[Path]:
    """Find only generated test, coverage, and Python cache artifacts."""

    targets: set[Path] = set()
    for name in (".pytest_cache", ".pytest_tmp", ".test-tmp", "htmlcov"):
        targets.add(root / name)
    targets.update(root.glob(".pytest-*"))
    for name in (".coverage", "coverage.xml"):
        targets.add(root / name)

    ignored_parts = {".git", ".venv", ".venvs", "output"}
    for candidate in root.rglob("__pycache__"):
        if not ignored_parts.intersection(candidate.relative_to(root).parts):
            targets.add(candidate)

    data_tmp = root / "data" / "tmp"
    if data_tmp.is_dir():
        targets.update(data_tmp.glob("pytest-*"))
        targets.update(data_tmp.glob(".pytest-*"))

    return sorted(
        (path for path in targets if path.exists() or path.is_symlink()),
        key=lambda path: (len(path.parts), str(path)),
        reverse=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove test caches and temporary artifacts. Benchmark results are never touched."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching artifacts without removing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = find_targets()
    if not targets:
        print("No test artifacts found.")
        return 0

    if args.dry_run:
        for path in targets:
            print(path.relative_to(PROJECT_ROOT))
        print(f"Found {len(targets)} test artifacts.")
        return 0

    removed = 0
    skipped = 0
    for path in targets:
        removed_delta, skipped_delta = remove_target(path)
        removed += removed_delta
        skipped += skipped_delta
    print(f"Removed {removed} test artifacts; skipped {skipped} missing artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
