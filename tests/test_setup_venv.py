from __future__ import annotations

from pathlib import Path

import pytest

import setup_venv
from venv_scripts import _model_script


def test_setup_venv_dispatches_selected_model_script(capsys):
    assert setup_venv.main(["--dry-run", "-m", "dreamreasoner"]) == 0
    output = capsys.readouterr().out
    assert "dreamreasoner.py setup" in output
    assert "illada.py" not in output


def test_setup_venv_defaults_to_every_matrix_model(capsys):
    assert setup_venv.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    for model in ("qwen3_4b", "illada", "dreamreasoner", "w1", "diffusiongemma"):
        assert f"{model}.py setup" in output


def test_model_run_uses_the_model_venv_python(monkeypatch):
    commands = []
    model_python = Path("model-specific-python")
    monkeypatch.setattr(_model_script, "ensure_environment", lambda profile, cuda: model_python)
    monkeypatch.setattr(_model_script, "repair_project_installation", lambda profile, python: None)
    monkeypatch.setattr(_model_script, "run", lambda command, **kwargs: commands.append(command))

    assert _model_script.main("illada", ["run"]) == 0
    assert commands[0][0] == model_python
    assert commands[0][1:4] == ["-m", "dllm_bench.cli", "matrix"]


def test_model_venvs_share_one_parent_directory(monkeypatch):
    monkeypatch.delenv("DLLM_VENV_DIR", raising=False)
    profile = _model_script.PROFILES["illada"]
    assert _model_script.venv_dir(profile) == _model_script.REPO_ROOT / ".venvs" / "illada"


def test_model_run_repair_avoids_reinstalling_dependencies(
    monkeypatch, tmp_path
):
    model_venv = tmp_path / "qwen-venv"
    python = _model_script.venv_python(model_venv)
    python.parent.mkdir(parents=True)
    python.touch()
    commands = []
    monkeypatch.setenv("DLLM_VENV_DIR", str(model_venv))
    monkeypatch.setattr(_model_script, "_project_importable", lambda executable: False)
    monkeypatch.setattr(
        _model_script,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    _model_script.repair_project_installation(
        _model_script.PROFILES["qwen3_4b"], python
    )

    assert commands[0] == [
        python, "-m", "pip", "install", "--no-deps", "-e", "."
    ]
    assert "import dllm_bench" in commands[1][2]
    assert not any("torch" in str(part) for command in commands for part in command)


def test_prepare_does_not_repair_or_install_the_project(monkeypatch, tmp_path):
    model_venv = tmp_path / "qwen-venv"
    model_python = _model_script.venv_python(model_venv)
    model_python.parent.mkdir(parents=True)
    model_python.touch()
    commands = []
    monkeypatch.setenv("DLLM_VENV_DIR", str(model_venv))
    monkeypatch.setattr(
        _model_script,
        "repair_project_installation",
        lambda profile, python: (_ for _ in ()).throw(
            AssertionError("prepare must not repair the venv")
        ),
    )
    monkeypatch.setattr(
        _model_script,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    assert _model_script.main("qwen3_4b", ["prepare"]) == 0
    assert commands == [
        [
            model_python,
            "prepare_model.py",
            "--model-config",
            "configs/models/qwen3_4b.yaml",
        ]
    ]


def test_prepare_requires_setup_when_model_environment_is_missing(
    monkeypatch, tmp_path
):
    missing_venv = tmp_path / "missing-qwen-venv"
    monkeypatch.setenv("DLLM_VENV_DIR", str(missing_venv))

    with pytest.raises(SystemExit, match="setup_venv.py -m qwen3_4b"):
        _model_script.main("qwen3_4b", ["prepare"])
