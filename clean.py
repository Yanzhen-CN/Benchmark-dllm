"""Utility script for cleaning temporary pytest/tmp artifacts."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
import ctypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean temporary pytest/tmp artifacts in this workspace."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[".pytest*", ".tmp*"],
        help="Glob pattern to match files/directories. Can be repeated.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan all subdirectories recursively.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show matched paths without deleting.")
    parser.add_argument(
        "--no-force-native",
        action="store_true",
        help="Disable Windows-native fallback deletion.",
    )
    parser.add_argument("--yes", action="store_true", help="Delete without interactive confirmation.")
    return parser.parse_args()


def collect_targets(root: Path, patterns: list[str], recursive: bool) -> list[Path]:
    all_targets: list[Path] = []
    walk = root.rglob if recursive else root.glob
    for pattern in patterns:
        all_targets.extend(walk(pattern))
    return sorted(set(all_targets), key=lambda p: str(p))


def ask_confirmation(count: int) -> bool:
    reply = input(f"Delete {count} path(s)? [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def make_writable(path: Path, recursive: bool = False) -> None:
    if recursive and path.exists():
        for child in sorted(path.rglob("*"), reverse=True):
            _clear_windows_readonly(child)
            _chmod_user_writable(child)
    if path.exists():
        _clear_windows_readonly(path)
        _chmod_user_writable(path)


def _chmod_user_writable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        if not (mode & stat.S_IWUSR):
            path.chmod(mode | stat.S_IWUSR)
    except Exception:
        # Some entries can be ephemeral in race windows while deleting.
        pass


def _clear_windows_readonly(path: Path) -> None:
    if os.name != "nt" or not path.exists():
        return
    try:
        FILE_ATTRIBUTE_READONLY = 0x1
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs != -1 and attrs & FILE_ATTRIBUTE_READONLY:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), attrs & ~FILE_ATTRIBUTE_READONLY)
    except Exception:
        pass


def native_remove(path: Path) -> bool:
    if os.name != "nt":
        return False

    quoted = f'"{path}"'
    subprocess.run(
        ["attrib", "-R", "-H", "-S", str(path), "/S", "/D"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if path.is_dir():
        proc = subprocess.run(
            ["cmd", "/c", f"rmdir /S /Q {quoted}"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        proc = subprocess.run(
            ["cmd", "/c", f"del /F /Q {quoted}"],
            check=False,
            capture_output=True,
            text=True,
        )

    if proc.returncode != 0 and not path.exists():
        # Some commands return non-zero for already gone paths.
        return True
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        print(f"[native-fallback-fail] {path}: {msg}", file=sys.stderr)
    return proc.returncode == 0 and not path.exists()


def remove_target(target: Path, use_native_fallback: bool) -> tuple[int, int]:
    try:
        if not target.exists():
            return 0, 0

        if target.is_dir() and not target.is_symlink():
            make_writable(target, recursive=True)
            for attempt in range(2):
                try:
                    for child in sorted(target.rglob("*"), reverse=True):
                        if child.is_file() or child.is_symlink():
                            child.unlink(missing_ok=True)
                        else:
                            child.rmdir()
                    if target.exists():
                        target.rmdir()
                    break
                except Exception:
                    if attempt == 1:
                        break
                    time.sleep(0.25)

            if use_native_fallback and target.exists():
                for _ in range(4):
                    make_writable(target, recursive=True)
                    if native_remove(target):
                        break
                    time.sleep(0.25)

            if target.exists():
                raise OSError(f"failed to remove directory after cleanup: {target}")
            return 1, 0

        make_writable(target)
        target.unlink()
        if target.exists():
            if use_native_fallback and not native_remove(target):
                raise OSError(f"failed to remove file: {target}")
            if target.exists():
                raise OSError(f"failed to remove file: {target}")
        return 0, 1
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"[skip] {target}: {exc}", file=sys.stderr)
        return 0, 0


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Root path is not a directory: {root}", file=sys.stderr)
        return 1

    targets = collect_targets(root, args.pattern, args.recursive)
    if not targets:
        print("No matching paths found.")
        return 0

    dirs = [p for p in targets if p.is_dir()]
    files = [p for p in targets if not p.is_dir()]
    print(f"Found {len(dirs)} directories and {len(files)} files.")
    for target in targets:
        print(f" - {target}")

    if args.dry_run:
        print("Dry run mode, no files deleted.")
        return 0

    if not args.yes and not ask_confirmation(len(targets)):
        print("Cancelled.")
        return 0

    removed_dirs = removed_files = skipped = 0
    use_native = not args.no_force_native
    for target in targets:
        d, f = remove_target(target, use_native_fallback=use_native)
        removed_dirs += d
        removed_files += f
        if d == 0 and f == 0:
            skipped += 1

    print(f"Deleted {removed_dirs} directories and {removed_files} files.")
    if skipped:
        print(f"Skipped {skipped} path(s).")
        print("Common cause: paths are still open in another process (VS Code explorer/search, terminal, or indexer).")
        print("Run again after closing heavy file scanners, or run once from a fresh shell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
