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
    for model in (
        "qwen3_4b",
        "qwen3_8b",
        "illada",
        "dreamreasoner",
        "w1",
        "diffusiongemma",
        "gemma4_26b_a4b",
    ):
        assert f"{model}.py setup" in output


def test_setup_venv_supports_separate_dg_comparison_matrix(capsys):
    matrix = Path("configs/experiments/dg_comparison.yaml")
    assert setup_venv.main(["--dry-run", "--matrix", str(matrix)]) == 0
    output = capsys.readouterr().out
    assert "diffusiongemma.py setup" in output
    assert "gemma4_26b.py setup" in output


def test_model_run_uses_the_model_venv_python(monkeypatch):
    commands = []
    model_python = Path("model-specific-python")
    monkeypatch.setattr(_model_script, "ensure_environment", lambda profile, cuda: model_python)
    monkeypatch.setattr(_model_script, "repair_project_installation", lambda profile, python: None)
    monkeypatch.setattr(_model_script, "run", lambda command, **kwargs: commands.append(command))

    assert _model_script.main("illada", ["run"]) == 0
    assert commands[0][0] == model_python
    assert commands[0][1:4] == ["-m", "dllm_bench.cli", "matrix"]


def test_generate_stage_does_not_receive_visualization_sample_limit(monkeypatch):
    monkeypatch.setenv("STAGE", "generate")
    monkeypatch.setenv("N_REPRESENTATIVE", "3")

    arguments = _model_script.benchmark_arguments(_model_script.PROFILES["illada"])

    assert "--n-representative" not in arguments


def test_visualize_stage_receives_visualization_sample_limit(monkeypatch):
    monkeypatch.setenv("STAGE", "visualize")
    monkeypatch.setenv("N_REPRESENTATIVE", "7")

    arguments = _model_script.benchmark_arguments(_model_script.PROFILES["illada"])

    index = arguments.index("--n-representative")
    assert arguments[index + 1] == "7"


def test_model_venvs_share_one_parent_directory(monkeypatch):
    monkeypatch.delenv("DLLM_VENV_DIR", raising=False)
    profile = _model_script.PROFILES["illada"]
    assert _model_script.venv_dir(profile) == _model_script.REPO_ROOT / ".venvs" / "illada"


def test_dreamreasoner_matches_checkpoint_transformers_version():
    assert _model_script.PROFILES["dreamreasoner"].transformers_version == "5.7.0"


def test_gemma4_ar_matches_diffusiongemma_runtime():
    gemma_ar = _model_script.PROFILES["gemma4_26b_a4b"]
    diffusion = _model_script.PROFILES["diffusiongemma"]
    assert gemma_ar.torch_version == diffusion.torch_version
    assert gemma_ar.transformers_version == diffusion.transformers_version


def test_qwen3_8b_matches_qwen3_4b_runtime():
    qwen_4b = _model_script.PROFILES["qwen3_4b"]
    qwen_8b = _model_script.PROFILES["qwen3_8b"]
    assert qwen_8b.torch_version == qwen_4b.torch_version
    assert qwen_8b.transformers_version == qwen_4b.transformers_version


def test_gpu_profile_env_defaults_expandable_segments(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    environment = _model_script.model_environment(_model_script.PROFILES["illada"])
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_gpu_profile_env_never_overrides_an_operator_supplied_value(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")
    environment = _model_script.model_environment(_model_script.PROFILES["illada"])
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "garbage_collection_threshold:0.6"


def test_non_gpu_profile_env_does_not_set_alloc_conf(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    environment = _model_script.model_environment(_model_script.PROFILES["w1"])
    assert "PYTORCH_CUDA_ALLOC_CONF" not in environment


def test_cuda_torch_local_version_matches_public_pin(monkeypatch):
    installed = {"torch": "2.6.0+cu124", "transformers": "5.14.1"}
    monkeypatch.setattr(
        _model_script,
        "_installed_distribution_version",
        lambda python, distribution: installed[distribution],
    )

    mismatches = _model_script._profile_version_mismatches(
        _model_script.PROFILES["qwen3_4b"], Path("python")
    )

    assert mismatches == {}


def test_different_torch_public_version_is_still_stale(monkeypatch):
    installed = {"torch": "2.5.1+cu124", "transformers": "5.14.1"}
    monkeypatch.setattr(
        _model_script,
        "_installed_distribution_version",
        lambda python, distribution: installed[distribution],
    )

    mismatches = _model_script._profile_version_mismatches(
        _model_script.PROFILES["qwen3_4b"], Path("python")
    )

    assert mismatches == {"torch": ("2.5.1+cu124", "2.6.0")}


def test_existing_model_venv_repairs_stale_profile_pins(monkeypatch, tmp_path):
    model_venv = tmp_path / "dreamreasoner-venv"
    python = _model_script.venv_python(model_venv)
    python.parent.mkdir(parents=True)
    python.touch()
    profile = _model_script.PROFILES["dreamreasoner"]
    repaired = []
    monkeypatch.setenv("DLLM_VENV_DIR", str(model_venv))
    monkeypatch.setattr(
        _model_script,
        "_profile_version_mismatches",
        lambda selected, executable: {"transformers": ("4.46.2", "5.7.0")},
    )
    monkeypatch.setattr(
        _model_script,
        "repair_profile_dependencies",
        lambda selected, executable, cuda, mismatches: repaired.append(
            (selected, executable, cuda, mismatches)
        ),
    )

    assert _model_script.ensure_environment(profile, "cu124") == python
    assert repaired == [
        (profile, python, "cu124", {"transformers": ("4.46.2", "5.7.0")})
    ]


def test_existing_model_venv_keeps_matching_profile_pins(monkeypatch, tmp_path):
    model_venv = tmp_path / "dreamreasoner-venv"
    python = _model_script.venv_python(model_venv)
    python.parent.mkdir(parents=True)
    python.touch()
    profile = _model_script.PROFILES["dreamreasoner"]
    monkeypatch.setenv("DLLM_VENV_DIR", str(model_venv))
    monkeypatch.setattr(
        _model_script, "_profile_version_mismatches", lambda selected, executable: {}
    )
    monkeypatch.setattr(
        _model_script,
        "repair_profile_dependencies",
        lambda *args: (_ for _ in ()).throw(AssertionError("matching venv changed")),
    )

    assert _model_script.ensure_environment(profile, "cu124") == python


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
