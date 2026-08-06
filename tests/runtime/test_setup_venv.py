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


def test_setup_venv_recreate_is_explicit_for_root_and_selected_model(capsys):
    assert setup_venv.main(
        ["--dry-run", "--recreate", "-m", "diffusiongemma"]
    ) == 0
    output = capsys.readouterr().out
    assert "root.py setup --recreate" in output
    assert "diffusiongemma.py setup --recreate" in output


def test_setup_venv_supports_llada2_1(capsys):
    assert setup_venv.main(["--dry-run", "-m", "llada2_1"]) == 0
    output = capsys.readouterr().out
    assert "llada2_1.py setup" in output


def test_setup_venv_defaults_to_every_matrix_model(capsys):
    assert setup_venv.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    for model in (
        "qwen3_4b",
        "qwen3_8b",
        "illada",
        "illada_vargen",
        "dreamreasoner",
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
        Path("configs/experiments/illada_entropy.yaml"),
    ],
)
def test_every_matrix_model_has_an_environment_profile(matrix):
    for model in setup_venv.matrix_model_names(matrix):
        assert model in _model_script.PROFILES
        assert (Path("venv_scripts") / f"{model}.py").is_file()


def test_model_run_uses_the_model_venv_python(monkeypatch):
    commands = []
    model_python = Path("model-specific-python")
    monkeypatch.setattr(_model_script, "require_environment", lambda profile: model_python)
    monkeypatch.setattr(_model_script, "_project_importable", lambda python: True)
    monkeypatch.setattr(_model_script, "run", lambda command, **kwargs: commands.append(command))

    assert _model_script.main("illada", ["run"]) == 0
    assert commands[0][0] == model_python
    assert commands[0][1:4] == ["-m", "dllm_bench.cli", "matrix"]


def test_model_run_never_creates_repairs_or_updates_environment(monkeypatch):
    model_python = Path("model-specific-python")
    commands = []
    monkeypatch.setattr(_model_script, "require_environment", lambda profile: model_python)
    monkeypatch.setattr(_model_script, "_project_importable", lambda python: True)
    monkeypatch.setattr(
        _model_script,
        "ensure_environment",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("run must not ensure or update an environment")
        ),
    )
    monkeypatch.setattr(
        _model_script,
        "setup_environment",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("run must not create an environment")
        ),
    )
    monkeypatch.setattr(
        _model_script,
        "repair_project_installation",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("run must not repair an environment")
        ),
    )
    monkeypatch.setattr(_model_script, "run", lambda command, **kwargs: commands.append(command))

    assert _model_script.main("illada_entropy", ["run"]) == 0
    assert commands[0][0] == model_python


def test_model_run_fails_cleanly_when_its_venv_python_is_missing(
    monkeypatch, tmp_path
):
    model_venv = tmp_path / "venvs" / "illada_entropy"
    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
    monkeypatch.setattr(
        _model_script,
        "setup_environment",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("missing venv must not be rebuilt during run")
        ),
    )

    with pytest.raises(SystemExit, match="missing or broken.*setup_venv.py"):
        _model_script.main("illada_entropy", ["run"])


def test_interactive_model_run_can_setup_a_missing_environment(
    monkeypatch, tmp_path
):
    model_venv = tmp_path / "venvs" / "illada_entropy"
    model_python = _model_script.venv_python(model_venv)
    commands = []
    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
    monkeypatch.setattr(_model_script.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    monkeypatch.setattr(
        _model_script,
        "setup_environment",
        lambda profile, cuda_index: model_python,
    )
    monkeypatch.setattr(
        _model_script, "run", lambda command, **kwargs: commands.append(command)
    )

    assert _model_script.main("illada_entropy", ["run"]) == 0
    assert commands[0][0] == model_python


def test_interactive_model_run_declines_missing_environment_setup(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DLLM_VENV_ROOT", str(tmp_path / "missing"))
    monkeypatch.setattr(_model_script.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr(
        _model_script,
        "setup_environment",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("declined setup must not change the environment")
        ),
    )

    with pytest.raises(SystemExit, match="setup was not run"):
        _model_script.main("illada_entropy", ["run"])


def test_interactive_model_run_can_repair_only_missing_project_install(
    monkeypatch, tmp_path
):
    model_venv = tmp_path / "venvs" / "illada_entropy"
    model_python = _model_script.venv_python(model_venv)
    model_python.parent.mkdir(parents=True)
    model_python.touch()
    commands = []
    importable = False
    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
    monkeypatch.setattr(_model_script.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    monkeypatch.setattr(
        _model_script,
        "_project_importable",
        lambda python: importable,
    )

    def repair(profile, python):
        nonlocal importable
        importable = True

    monkeypatch.setattr(_model_script, "repair_project_installation", repair)
    monkeypatch.setattr(
        _model_script,
        "setup_environment",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("an intact venv must not be fully rebuilt")
        ),
    )
    monkeypatch.setattr(
        _model_script, "run", lambda command, **kwargs: commands.append(command)
    )

    assert _model_script.main("illada_entropy", ["run"]) == 0
    assert commands[0][0] == model_python


def test_setup_removes_environment_with_missing_python(monkeypatch, tmp_path):
    profile = _model_script.PROFILES["llada2_1"]
    model_venv = tmp_path / "llada2_1"
    stale_file = model_venv / "lib" / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale", encoding="utf-8")
    commands = []

    monkeypatch.setenv("DLLM_VENV_ROOT", str(tmp_path))

    def capture_run(command, **kwargs):
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            assert not model_venv.exists()
            python = _model_script.venv_python(model_venv)
            python.parent.mkdir(parents=True)
            python.touch()
        elif len(commands) == 2:
            raise RuntimeError("stop after venv recreation")

    monkeypatch.setattr(_model_script, "run", capture_run)

    with pytest.raises(RuntimeError, match="stop after venv recreation"):
        _model_script.setup_environment(profile, "cu124")

    assert not stale_file.exists()
    assert commands[0][1:3] == ["-m", "venv"]


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
    monkeypatch.delenv("DLLM_VENV_ROOT", raising=False)
    monkeypatch.setenv("DLLM_VENV_DIR", "ignored-legacy-single-model-path")
    profile = _model_script.PROFILES["illada"]
    assert _model_script.venv_dir(profile) == _model_script.REPO_ROOT / ".venvs" / "illada"


def test_illada_vargen_has_a_separate_but_version_matched_environment():
    fixed = _model_script.PROFILES["illada"]
    vargen = _model_script.PROFILES["illada_vargen"]

    assert vargen.venv_subdir == "illada_vargen"
    assert vargen.model_config == "configs/models/illada_vargen.yaml"
    assert vargen.torch_version == fixed.torch_version
    assert vargen.transformers_version == fixed.transformers_version


def test_illada_entropy_has_a_separate_but_version_matched_environment():
    fixed = _model_script.PROFILES["illada"]
    entropy = _model_script.PROFILES["illada_entropy"]

    assert entropy.venv_subdir == "illada_entropy"
    assert entropy.model_config == "configs/models/illada_entropy.yaml"
    assert entropy.torch_version == fixed.torch_version
    assert entropy.transformers_version == fixed.transformers_version


def test_model_environment_disables_unavailable_hf_transfer(monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setattr(
        _model_script,
        "_installed_distribution_version",
        lambda python, distribution: None,
    )

    environment = _model_script.model_environment(
        _model_script.PROFILES["illada_entropy"]
    )

    assert environment["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_model_environment_keeps_available_hf_transfer(monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setattr(
        _model_script,
        "_installed_distribution_version",
        lambda python, distribution: "0.1.9",
    )

    environment = _model_script.model_environment(
        _model_script.PROFILES["illada_entropy"]
    )

    assert environment["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_gemma_reuses_legacy_environment_after_public_rename(tmp_path, monkeypatch):
    monkeypatch.delenv("DLLM_VENV_ROOT", raising=False)
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


def test_model_run_forwards_each_output_length_override(monkeypatch):
    monkeypatch.setenv("MAX_NEW_TOKENS", "1024,2048")
    arguments = _model_script.benchmark_arguments(
        _model_script.PROFILES["illada_vargen"]
    )

    assert arguments.count("--max-new-tokens") == 2
    first = arguments.index("--max-new-tokens")
    second = arguments.index("--max-new-tokens", first + 1)
    assert arguments[first + 1] == "1024"
    assert arguments[second + 1] == "2048"


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
    assert profile.required_distributions == (
        "vllm",
        "transformers",
        "torch",
        "ninja",
    )
    assert "8cb2db16072cebbb944564f84f21045a90151ad1" in profile.setup_requirements[0]
    assert profile.setup_requirements[1] == "ninja>=1.11"
    assert profile.cuda_runtime == "12.9"
    assert profile.minimum_driver_version == "575.51.03"
    assert profile.uv_torch_backend == "cu129"
    assert profile.precompiled_wheel_variant == "cu129"
    assert profile.precompiled_wheel_commit == "84f7a55340601ddc77b850025ea1ca03f6b1fd82"


def test_existing_gemma_dflash_environment_repairs_missing_ninja_in_place(
    monkeypatch, tmp_path
):
    profile = _model_script.PROFILES["gemma_dflash"]
    model_venv = tmp_path / "venvs" / "gemma_dflash"
    python = _model_script.venv_python(model_venv)
    python.parent.mkdir(parents=True)
    python.touch()
    commands = []
    ninja_installed = False

    def installed_version(_python, distribution):
        if distribution == "ninja":
            return "1.11" if ninja_installed else None
        return "installed"

    def capture_run(command, **kwargs):
        nonlocal ninja_installed
        commands.append(command)
        if command[-1] == "ninja>=1.11":
            ninja_installed = True

    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
    monkeypatch.setattr(
        _model_script,
        "_installed_distribution_version",
        installed_version,
    )
    monkeypatch.setattr(
        _model_script, "_profile_version_mismatches", lambda *args: {}
    )
    monkeypatch.setattr(
        _model_script, "run", capture_run
    )

    assert _model_script.ensure_environment(profile, "cu124") == python
    assert commands == [
        [python, "-m", "pip", "install", "--upgrade", "ninja>=1.11"]
    ]


def test_installation_environment_keeps_large_build_files_under_data_root(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "persistent-data"
    monkeypatch.setenv("DLLM_DATA_ROOT", str(data_root))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "wrong-hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "wrong-hub"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "wrong-transformers"))
    monkeypatch.delenv("VLLM_USE_PRECOMPILED", raising=False)
    monkeypatch.delenv("UV_NO_CACHE", raising=False)

    environment = _model_script.installation_environment(
        prefer_vllm_precompiled=True,
        avoid_uv_cache=True,
    )

    assert environment["HF_HOME"] == str(data_root / "huggingface")
    assert environment["HF_HUB_CACHE"] == str(data_root / "huggingface" / "hub")
    assert environment["HF_XET_CACHE"] == str(data_root / "huggingface" / "xet")
    assert environment["TMPDIR"] == str(data_root / "tmp")
    assert environment["TMP"] == str(data_root / "tmp")
    assert environment["TEMP"] == str(data_root / "tmp")
    assert environment["TORCH_EXTENSIONS_DIR"] == str(
        data_root / "torch-extensions"
    )
    assert environment["VLLM_USE_PRECOMPILED"] == "1"
    assert environment["UV_NO_CACHE"] == "1"
    assert environment["PIP_NO_CACHE_DIR"] == "1"
    assert "TRANSFORMERS_CACHE" not in environment
    assert all(
        (data_root / name).is_dir()
        for name in ("huggingface", "tmp", "torch-extensions")
    )


def test_installation_environment_ignores_legacy_per_cache_overrides(
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

    assert environment["TMPDIR"] == str(tmp_path / "data" / "tmp")
    assert "UV_CACHE_DIR" not in environment
    assert "DLLM_BUILD_TMPDIR" not in environment
    assert "DLLM_UV_CACHE_DIR" not in environment
    assert environment["VLLM_USE_PRECOMPILED"] == "0"
    assert environment["UV_NO_CACHE"] == "1"


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
    model_venv = tmp_path / "venvs" / "dreamreasoner"
    python = _model_script.venv_python(model_venv)
    python.parent.mkdir(parents=True)
    python.touch()
    profile = _model_script.PROFILES["dreamreasoner"]
    repaired = []
    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
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
    model_venv = tmp_path / "venvs" / "dreamreasoner"
    python = _model_script.venv_python(model_venv)
    python.parent.mkdir(parents=True)
    python.touch()
    profile = _model_script.PROFILES["dreamreasoner"]
    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
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
    model_venv = tmp_path / "venvs" / "qwen3_4b"
    python = _model_script.venv_python(model_venv)
    python.parent.mkdir(parents=True)
    python.touch()
    commands = []
    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
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


def test_prepare_uses_the_complete_model_environment(monkeypatch, tmp_path):
    model_venv = tmp_path / "venvs" / "qwen3_4b"
    model_python = _model_script.venv_python(model_venv)
    model_python.parent.mkdir(parents=True)
    model_python.touch()
    commands = []
    monkeypatch.setenv("DLLM_VENV_ROOT", str(model_venv.parent))
    monkeypatch.setattr(_model_script, "_project_importable", lambda python: True)
    monkeypatch.setattr(
        _model_script,
        "repair_project_installation",
        lambda profile, python: (_ for _ in ()).throw(
            AssertionError("complete prepare venv must not be repaired")
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
    missing_venv_root = tmp_path / "missing-venvs"
    monkeypatch.setenv("DLLM_VENV_ROOT", str(missing_venv_root))

    with pytest.raises(SystemExit, match="setup_venv.py -m qwen3_4b"):
        _model_script.main("qwen3_4b", ["prepare"])
