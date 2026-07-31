"""Utility script for cleaning temporary pytest/tmp artifacts."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path


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
        default=None,
        help=(
            "Glob pattern to match files/directories. Can be repeated; "
            "default: .pytest* and .tmp*."
        ),
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


def _extended_windows_path(path: Path) -> str:
    """Return a Win32 long-path form without passing through ``cmd.exe``."""
    absolute = str(path.absolute())
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def native_remove(path: Path) -> tuple[bool, OSError | None]:
    """Make one direct Win32 deletion attempt and preserve its real error."""
    if os.name != "nt":
        return False, None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    remove_directory = kernel32.RemoveDirectoryW
    remove_directory.argtypes = [ctypes.c_wchar_p]
    remove_directory.restype = ctypes.c_int
    delete_file = kernel32.DeleteFileW
    delete_file.argtypes = [ctypes.c_wchar_p]
    delete_file.restype = ctypes.c_int

    native_path = _extended_windows_path(path)
    removed = remove_directory(native_path) if path.is_dir() else delete_file(native_path)
    if removed or not path.exists():
        return True, None

    error_code = ctypes.get_last_error()
    return False, OSError(
        None,
        ctypes.FormatError(error_code).strip(),
        str(path),
        error_code,
    )


def _repair_windows_acl(path: Path) -> bool:
    """Reset a broken temp-path ACL when the script is already elevated."""
    if os.name != "nt":
        return False
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            return False
    except Exception:
        return False

    take_ownership = subprocess.run(
        ["takeown", "/F", str(path), "/R", "/D", "Y"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    reset_acl = subprocess.run(
        ["icacls", str(path), "/reset", "/T", "/C", "/Q"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return take_ownership.returncode == 0 and reset_acl.returncode == 0


def _rmtree_on_error(function, path: str, _exc_info) -> None:
    blocked = Path(path)
    _clear_windows_readonly(blocked)
    _chmod_user_writable(blocked)
    function(path)


def remove_target(target: Path, use_native_fallback: bool) -> tuple[int, int]:
    try:
        if not target.exists():
            return 0, 0

        if target.is_dir() and not target.is_symlink():
            last_error: BaseException | None = None
            try:
                make_writable(target, recursive=True)
            except Exception as exc:
                last_error = exc
            for attempt in range(2):
                try:
                    shutil.rmtree(target, onerror=_rmtree_on_error)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 1:
                        break
                    time.sleep(0.25)

            if use_native_fallback and target.exists():
                try:
                    make_writable(target, recursive=True)
                except Exception as exc:
                    last_error = exc
                removed, native_error = native_remove(target)
                if not removed and native_error is not None:
                    last_error = native_error
                    windows_error = getattr(native_error, "winerror", None)
                    if windows_error == 5 and _repair_windows_acl(target):
                        try:
                            make_writable(target, recursive=True)
                            shutil.rmtree(target, onerror=_rmtree_on_error)
                        except Exception as exc:
                            last_error = exc
                        if target.exists():
                            removed, repaired_error = native_remove(target)
                            if not removed and repaired_error is not None:
                                last_error = repaired_error

            if target.exists():
                raise OSError(
                    f"failed to remove directory after cleanup; last error: "
                    f"{last_error or 'unknown error'}"
                )
            return 1, 0

        last_error = None
        try:
            make_writable(target)
            target.unlink()
        except Exception as exc:
            last_error = exc
        if target.exists():
            if use_native_fallback:
                removed, native_error = native_remove(target)
                if (
                    not removed
                    and native_error is not None
                    and getattr(native_error, "winerror", None) == 5
                    and _repair_windows_acl(target)
                ):
                    removed, native_error = native_remove(target)
                if not removed:
                    raise OSError(
                        f"failed to remove file; last error: "
                        f"{native_error or last_error or 'unknown error'}"
                    )
            if target.exists():
                raise OSError(
                    f"failed to remove file; last error: "
                    f"{last_error or 'unknown error'}"
                )
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

    patterns = args.pattern or [".pytest*", ".tmp*"]
    targets = collect_targets(root, patterns, args.recursive)
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
        print(
            "WinError 5 means the path ACL denies access; WinError 32 means "
            "another process still has it open."
        )
        print(
            "For WinError 5, run this script once from an elevated terminal. "
            "For WinError 32, close the owning process and retry."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
