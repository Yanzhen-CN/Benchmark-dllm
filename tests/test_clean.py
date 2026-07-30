from __future__ import annotations

from pathlib import Path

import clean


def test_remove_target_deletes_nested_temporary_directory(tmp_path: Path):
    target = tmp_path / ".tmp_nested"
    nested = target / "one" / "two"
    nested.mkdir(parents=True)
    (nested / "result.json").write_text("{}", encoding="utf-8")

    assert clean.remove_target(target, use_native_fallback=True) == (1, 0)
    assert not target.exists()


def test_native_remove_uses_direct_win32_delete_for_empty_directory(tmp_path: Path):
    target = tmp_path / ".tmp_native"
    target.mkdir()

    removed, error = clean.native_remove(target)

    if clean.os.name == "nt":
        assert removed is True
        assert error is None
        assert not target.exists()
    else:
        assert removed is False
        assert error is None


def test_native_remove_preserves_real_win32_failure(tmp_path: Path):
    target = tmp_path / ".tmp_nonempty_native"
    target.mkdir()
    (target / "still-here.txt").write_text("locked", encoding="utf-8")

    removed, error = clean.native_remove(target)

    if clean.os.name == "nt":
        assert removed is False
        assert error is not None
        assert error.winerror == 145
        assert error.filename == str(target)
    else:
        assert removed is False
        assert error is None


def test_remove_target_reports_one_real_native_error(
    tmp_path: Path, monkeypatch, capsys
):
    target = tmp_path / ".tmp_denied"
    target.mkdir()

    def deny_rmtree(*_args, **_kwargs):
        raise PermissionError(5, "Access is denied", str(target))

    native_error = PermissionError(5, "Access is denied", str(target))
    monkeypatch.setattr(clean.shutil, "rmtree", deny_rmtree)
    monkeypatch.setattr(
        clean, "native_remove", lambda _path: (False, native_error)
    )

    assert clean.remove_target(target, use_native_fallback=True) == (0, 0)
    error_output = capsys.readouterr().err
    assert error_output.count("Access is denied") == 1
    assert "last error" in error_output
