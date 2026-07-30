from __future__ import annotations

import subprocess
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
        "illada_vargen",
        "dreamreasoner",
        "w1",
        "diffusiongemma",
        "gemma",
        "gemma_dflash",
    ):
        assert f"{model}.py setup" in output


def test_setup_venv_supports_separate_dg_comparison_matrix(capsys):
    matrix = Path("configs/experiments/dg_comparison.yaml")
    assert setup_venv.main(["--dry-run", "--matrix", str(matrix)]) == 0
    output = capsys.readouterr().out
    assert "diffusiongemma.py setup" in output
    assert "gemma.py setup" in output
    assert "gemma_dflash.py setup" in output


@pytest.mark.parametrize(
    "matrix",
    [
        Path("configs/experiments/full_matrix.yaml"),
        Path("configs/experiments/dg_comparison.yaml"),
    ],
)
def test_every_matrix_model_has_an_environment_profile(matrix):
    for model in setup_venv.matrix_model_names(matrix):
        assert model in _model_script.PROFILES
        assert (Path("venv_scripts") / f"{model}.py").is_file()


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


def test_illada_vargen_has_a_separate_but_version_matched_environment():
    fixed = _model_script.PROFILES["illada"]
    vargen = _model_script.PROFILES["illada_vargen"]

    assert vargen.venv_subdir == "illada_vargen"
    assert vargen.model_config == "configs/models/illada_vargen.yaml"
    assert vargen.torch_version == fixed.torch_version
    assert vargen.transformers_version == fixed.transformers_version


def test_gemma_reuses_legacy_environment_after_public_rename(tmp_path, monkeypatch):
    monkeypatch.delenv("DLLM_VENV_DIR", raising=False)
    monkeypatch.setattr(_model_script, "REPO_ROOT", tmp_path)
    legacy = tmp_path / ".venvs" / "gemma4_26b_a4b"
    legacy.mkdir(parents=True)

    assert _model_script.venv_dir(_model_script.PROFILES["gemma"]) == legacy


def test_model_run_forwards_sampling_variant_filter(monkeypatch):
    monkeypatch.setenv("MATRIX_VARIANTS", "fast")
    arguments = _model_script.benchmark_arguments(_model_script.PROFILES["illada"])
    variants_index = arguments.index("--variants")
    assert arguments[variants_index + 1] == "fast"


def test_model_run_forwards_each_hellobench_length(monkeypatch):
    monkeypatch.setenv("HELLOBENCH_LENGTHS", "2k,4k")
    arguments = _model_script.benchmark_arguments(_model_script.PROFILES["illada"])

    assert arguments.count("--hellobench-length") == 2
    first = arguments.index("--hellobench-length")
    second = arguments.index("--hellobench-length", first + 1)
    assert arguments[first + 1] == "2k"
    assert arguments[second + 1] == "4k"


def test_model_run_forwards_temporary_output_length_override(monkeypatch):
    monkeypatch.setenv("MAX_NEW_TOKENS", "512")
    arguments = _model_script.benchmark_arguments(
        _model_script.PROFILES["qwen3_4b"]
    )

    index = arguments.index("--max-new-tokens")
    assert arguments[index + 1] == "512"


def test_dreamreasoner_matches_checkpoint_transformers_version():
    assert _model_script.PROFILES["dreamreasoner"].transformers_version == "5.7.0"


def test_gemma4_ar_matches_diffusiongemma_runtime():
    gemma_ar = _model_script.PROFILES["gemma"]
    diffusion = _model_script.PROFILES["diffusiongemma"]
    assert gemma_ar.torch_version == diffusion.torch_version
    assert gemma_ar.transformers_version == diffusion.transformers_version
    assert gemma_ar.torchvision_version == "0.21.0"
    assert diffusion.torchvision_version == "0.21.0"


def test_gemma_dflash_uses_its_own_vllm_environment():
    profile = _model_script.PROFILES["gemma_dflash"]

    assert profile.venv_subdir == "gemma_dflash"
    assert profile.model_config == "configs/models/gemma_dflash.yaml"
    assert profile.torch_version is None
    assert profile.transformers_version == "5.14.1"
    assert (
        profile.transformers_version
        == _model_script.PROFILES["gemma"].transformers_version
    )
    assert profile.required_distributions == ("vllm", "transformers", "torch")
    assert "8cb2db16072cebbb944564f84f21045a90151ad1" in profile.setup_requirements[0]
    assert profile.cuda_runtime == "12.9"
    assert profile.minimum_driver_version == "575.51.03"
    assert profile.uv_torch_backend == "cu129"
    assert profile.precompiled_wheel_variant == "cu129"
    assert profile.precompiled_wheel_commit == "84f7a55340601ddc77b850025ea1ca03f6b1fd82"


def test_installation_environment_keeps_large_build_files_under_data_root(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "persistent-data"
    monkeypatch.setenv("DLLM_DATA_ROOT", str(data_root))
    monkeypatch.delenv("DLLM_PIP_CACHE_DIR", raising=False)
    monkeypatch.delenv("DLLM_UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("DLLM_BUILD_TMPDIR", raising=False)
    monkeypatch.delenv("DLLM_TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.delenv("VLLM_USE_PRECOMPILED", raising=False)
    monkeypatch.delenv("UV_NO_CACHE", raising=False)

    environment = _model_script.installation_environment(
        prefer_vllm_precompiled=True,
        avoid_uv_cache=True,
    )

    assert environment["PIP_CACHE_DIR"] == str(data_root / "pip-cache")
    assert environment["UV_CACHE_DIR"] == str(data_root / "uv-cache")
    assert environment["TMPDIR"] == str(data_root / "tmp")
    assert environment["TMP"] == str(data_root / "tmp")
    assert environment["TEMP"] == str(data_root / "tmp")
    assert environment["TORCH_EXTENSIONS_DIR"] == str(
        data_root / "torch-extensions"
    )
    assert environment["VLLM_USE_PRECOMPILED"] == "1"
    assert environment["UV_NO_CACHE"] == "1"
    assert all(
        (data_root / name).is_dir()
        for name in ("pip-cache", "uv-cache", "tmp", "torch-extensions")
    )


def test_installation_environment_respects_explicit_build_overrides(
    tmp_path, monkeypatch
):
    custom_tmp = tmp_path / "custom-tmp"
    custom_uv = tmp_path / "custom-uv"
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DLLM_BUILD_TMPDIR", str(custom_tmp))
    monkeypatch.setenv("DLLM_UV_CACHE_DIR", str(custom_uv))
    monkeypatch.setenv("VLLM_USE_PRECOMPILED", "0")
    monkeypatch.setenv("UV_NO_CACHE", "0")

    environment = _model_script.installation_environment(
        prefer_vllm_precompiled=True,
        avoid_uv_cache=True,
    )

    assert environment["TMPDIR"] == str(custom_tmp)
    assert environment["UV_CACHE_DIR"] == str(custom_uv)
    assert environment["VLLM_USE_PRECOMPILED"] == "0"
    assert environment["UV_NO_CACHE"] == "0"


def test_gemma_dflash_allows_known_xgrammar_transformers_metadata_conflict(
    monkeypatch, capsys
):
    conflict = (
        "xgrammar 0.2.4 has requirement transformers<5,>=4.38.0, "
        "but you have transformers 5.14.1."
    )
    monkeypatch.setattr(
        _model_script.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout=f"{conflict}\n", stderr=""
        ),
    )

    _model_script.check_installed_dependencies(
        _model_script.PROFILES["gemma_dflash"], Path("python")
    )

    assert conflict in capsys.readouterr().out


def test_gemma_dflash_still_rejects_other_dependency_conflicts(monkeypatch):
    conflict = "some-package requires other-package<2, but you have other-package 3.0."
    monkeypatch.setattr(
        _model_script.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout=f"{conflict}\n", stderr=""
        ),
    )

    with pytest.raises(subprocess.CalledProcessError):
        _model_script.check_installed_dependencies(
            _model_script.PROFILES["gemma_dflash"], Path("python")
        )


def test_gemma_dflash_uses_installed_cuda_forward_compatibility(
    tmp_path, monkeypatch
):
    compatibility_dir = tmp_path / "cuda-12.9" / "compat"
    compatibility_dir.mkdir(parents=True)
    (compatibility_dir / "libcuda.so.1").touch()
    (compatibility_dir / "libcuda.so.575.51.03").touch()
    monkeypatch.setenv("DLLM_CUDA_COMPAT_DIR", str(compatibility_dir))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/libs")
    monkeypatch.setattr(
        _model_script, "_detected_nvidia_driver_version", lambda: "570.124.06"
    )

    environment = _model_script.apply_cuda_compatibility(
        _model_script.PROFILES["gemma_dflash"], {"LD_LIBRARY_PATH": "/existing/libs"}
    )

    assert environment["LD_LIBRARY_PATH"].split(_model_script.os.pathsep) == [
        str(compatibility_dir),
        "/existing/libs",
    ]
    assert environment["DLLM_CUDA_COMPAT_ACTIVE"] == str(compatibility_dir)


def test_gemma_dflash_rejects_old_driver_without_compatibility_package(
    tmp_path, monkeypatch
):
    stale_compatibility_dir = tmp_path / "cuda-12.4" / "compat"
    stale_compatibility_dir.mkdir(parents=True)
    (stale_compatibility_dir / "libcuda.so.1").touch()
    (stale_compatibility_dir / "libcuda.so.550.90.12").touch()
    monkeypatch.setenv("DLLM_CUDA_COMPAT_DIR", str(stale_compatibility_dir))
    monkeypatch.setattr(
        _model_script, "_detected_nvidia_driver_version", lambda: "560.124.06"
    )

    with pytest.raises(SystemExit, match="cuda-compat-12-9"):
        _model_script.apply_cuda_compatibility(
            _model_script.PROFILES["gemma_dflash"], {}
        )


def test_gemma_dflash_accepts_native_cuda_129_driver(monkeypatch):
    monkeypatch.setenv("DLLM_CUDA_COMPAT_DIR", "/missing")
    monkeypatch.setattr(
        _model_script, "_detected_nvidia_driver_version", lambda: "575.51.03"
    )
    environment = {"existing": "value"}

    assert _model_script.apply_cuda_compatibility(
        _model_script.PROFILES["gemma_dflash"], environment
    ) == {"existing": "value"}


def test_diffusiongemma_repairs_missing_torchvision(monkeypatch):
    installed = {
        "torch": "2.6.0+cu124",
        "transformers": "5.14.1",
        "torchvision": None,
    }
    monkeypatch.setattr(
        _model_script,
        "_installed_distribution_version",
        lambda python, distribution: installed[distribution],
    )

    mismatches = _model_script._profile_version_mismatches(
        _model_script.PROFILES["diffusiongemma"], Path("python")
    )

    assert mismatches == {"torchvision": (None, "0.21.0")}


def test_torchvision_cuda_local_version_matches_public_pin(monkeypatch):
    installed = {
        "torch": "2.6.0+cu124",
        "transformers": "5.14.1",
        "torchvision": "0.21.0+cu124",
    }
    monkeypatch.setattr(
        _model_script,
        "_installed_distribution_version",
        lambda python, distribution: installed[distribution],
    )

    assert _model_script._profile_version_mismatches(
        _model_script.PROFILES["diffusiongemma"], Path("python")
    ) == {}


def test_qwen3_8b_matches_qwen3_4b_runtime():
    qwen_4b = _model_script.PROFILES["qwen3_4b"]
    qwen_8b = _model_script.PROFILES["qwen3_8b"]
    assert qwen_8b.torch_version == qwen_4b.torch_version
    assert qwen_8b.transformers_version == qwen_4b.transformers_version


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
