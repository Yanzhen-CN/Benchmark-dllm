"""Shared implementation for the per-model Python environment scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
CUDA_INDEXES = ("cu118", "cu121", "cu124", "cu126")


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    venv_subdir: str
    model_config: str
    extras: str
    torch_version: str | None = None
    transformers_version: str | None = None
    torch_cuda_indexes: tuple[str, ...] = ()
    torchvision_version: str | None = None


PROFILES: Mapping[str, ModelProfile] = {
    "qwen3_4b": ModelProfile(
        "qwen3_4b", "qwen3_4b", "configs/models/qwen3_4b.yaml",
        "dev,hf,gpu", "2.6.0", "5.14.1", ("cu118", "cu124", "cu126"),
    ),
    "qwen3_8b": ModelProfile(
        "qwen3_8b", "qwen3_8b", "configs/models/qwen3_8b.yaml",
        "dev,hf,gpu", "2.6.0", "5.14.1", ("cu118", "cu124", "cu126"),
    ),
    "illada": ModelProfile(
        "illada", "illada", "configs/models/illada.yaml",
        "dev,hf,gpu", "2.6.0", "4.57.1", ("cu118", "cu124", "cu126"),
    ),
    "illada_optimized": ModelProfile(
        "illada_optimized", "illada", "configs/models/illada_optimized.yaml",
        "dev,hf,gpu", "2.6.0", "4.57.1", ("cu118", "cu124", "cu126"),
    ),
    "dreamreasoner": ModelProfile(
        "dreamreasoner", "dreamreasoner", "configs/models/dreamreasoner.yaml",
        "dev,hf,gpu", "2.5.1", "5.7.0", ("cu118", "cu121", "cu124"),
    ),
    "dreamreasoner_optimized": ModelProfile(
        "dreamreasoner_optimized", "dreamreasoner",
        "configs/models/dreamreasoner_optimized.yaml",
        "dev,hf,gpu", "2.5.1", "5.7.0", ("cu118", "cu121", "cu124"),
    ),
    "diffusiongemma": ModelProfile(
        "diffusiongemma", "diffusiongemma", "configs/models/diffusiongemma.yaml",
        "dev,diffusiongemma,gpu", "2.6.0", "5.14.1", ("cu118", "cu124", "cu126"),
        torchvision_version="0.21.0",
    ),
    "gemma4_26b_a4b": ModelProfile(
        "gemma4_26b_a4b", "gemma4_26b_a4b", "configs/models/gemma4_26b_a4b.yaml",
        "dev,gemma4,gpu", "2.6.0", "5.14.1", ("cu118", "cu124", "cu126"),
        torchvision_version="0.21.0",
    ),
    "w1": ModelProfile("w1", "w1", "configs/models/w1.yaml", "dev,api"),
}


def venv_dir(profile: ModelProfile) -> Path:
    override = os.environ.get("DLLM_VENV_DIR")
    return (
        Path(override).expanduser().resolve()
        if override
        else REPO_ROOT / ".venvs" / profile.venv_subdir
    )


def venv_python(directory: Path) -> Path:
    windows = directory / "Scripts" / "python.exe"
    return windows if os.name == "nt" else directory / "bin" / "python"


def run(command: Sequence[str | Path], *, env: Mapping[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"+ {printable}", flush=True)
    subprocess.run(
        [str(part) for part in command], cwd=REPO_ROOT, env=env, check=True
    )


def setup_environment(profile: ModelProfile, cuda_index: str) -> Path:
    if cuda_index not in CUDA_INDEXES:
        raise SystemExit(f"unsupported CUDA index {cuda_index}; use {', '.join(CUDA_INDEXES)}")
    if profile.torch_version and cuda_index not in profile.torch_cuda_indexes:
        supported = ", ".join(profile.torch_cuda_indexes)
        raise SystemExit(
            f"torch {profile.torch_version} for {profile.model_id} has no {cuda_index} "
            f"wheel; supported indexes: {supported}"
        )

    directory = venv_dir(profile)
    base_python = os.environ.get("PYTHON_BIN", sys.executable)
    data_root = Path(os.environ.get("DLLM_DATA_ROOT", REPO_ROOT / ".data"))
    cache_dir = Path(os.environ.get("DLLM_PIP_CACHE_DIR", data_root / "pip-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    install_env = os.environ.copy()
    install_env["PIP_CACHE_DIR"] = str(cache_dir)

    run([base_python, "-m", "venv", directory])
    python = venv_python(directory)
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], env=install_env)
    if profile.torch_version:
        run(
            [python, "-m", "pip", "install", "--upgrade", f"torch=={profile.torch_version}",
             "--index-url", f"https://download.pytorch.org/whl/{cuda_index}"],
            env=install_env,
        )
        uninstall = [str(python), "-m", "pip", "uninstall", "-y", "torchaudio"]
        if profile.torchvision_version is None:
            uninstall.append("torchvision")
        subprocess.run(uninstall, cwd=REPO_ROOT, env=install_env, check=False)
        if profile.torchvision_version is not None:
            run(
                [python, "-m", "pip", "install", "--upgrade",
                 f"torchvision=={profile.torchvision_version}",
                 "--index-url", f"https://download.pytorch.org/whl/{cuda_index}"],
                env=install_env,
            )
    if profile.transformers_version:
        run(
            [python, "-m", "pip", "install", "--upgrade",
             f"transformers=={profile.transformers_version}", "accelerate==1.14.0",
             "safetensors>=0.8.0", "sentencepiece"],
            env=install_env,
        )
    if "gpu" in profile.extras.split(","):
        subprocess.run(
            [str(python), "-m", "pip", "uninstall", "-y", "pynvml"],
            cwd=REPO_ROOT,
            env=install_env,
            check=False,
        )
    run([python, "-m", "pip", "install", "-e", f".[{profile.extras}]"], env=install_env)
    run([python, "-m", "pip", "check"], env=install_env)
    run([python, "-c", _IMPORT_CHECK])
    print(f"Environment ready: {profile.model_id}\nPath: {directory}")
    return python


_IMPORT_CHECK = """\
import importlib.util
print('Environment import check')
import dllm_bench
print('dllm_bench: OK')
if importlib.util.find_spec('torch'):
    import torch
    print('torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA runtime:', torch.version.cuda)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
if importlib.util.find_spec('torchvision'):
    import torchvision
    print('torchvision:', torchvision.__version__)
if importlib.util.find_spec('transformers'):
    import transformers
    print('transformers:', transformers.__version__)
if importlib.util.find_spec('requests'):
    import requests
    print('requests:', requests.__version__)
"""


def _project_importable(python: Path) -> bool:
    result = subprocess.run(
        [str(python), "-c", "import dllm_bench"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def repair_project_installation(profile: ModelProfile, python: Path) -> None:
    """Repair a legacy editable install immediately before model execution."""
    if not _project_importable(python):
        # A venv may survive an interrupted/older setup with its heavyweight
        # model dependencies intact but without this repository installed.
        # Repair only the editable project link; --no-deps deliberately avoids
        # replacing the profile's pinned Torch/Transformers/CUDA packages.
        print(
            f"Repairing missing dllm_bench installation in: {venv_dir(profile)}",
            flush=True,
        )
        install_env = os.environ.copy()
        data_root = Path(os.environ.get("DLLM_DATA_ROOT", REPO_ROOT / ".data"))
        cache_dir = Path(
            os.environ.get("DLLM_PIP_CACHE_DIR", data_root / "pip-cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        install_env["PIP_CACHE_DIR"] = str(cache_dir)
        run(
            [python, "-m", "pip", "install", "--no-deps", "-e", "."],
            env=install_env,
        )
        run([python, "-c", "import dllm_bench; print('dllm_bench: OK')"])


def ensure_environment(profile: ModelProfile, cuda_index: str) -> Path:
    python = venv_python(venv_dir(profile))
    if not python.is_file():
        return setup_environment(profile, cuda_index)

    mismatches = _profile_version_mismatches(profile, python)
    if mismatches:
        repair_profile_dependencies(profile, python, cuda_index, mismatches)
    return python


def _installed_distribution_version(python: Path, distribution: str) -> str | None:
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from importlib.metadata import version; "
                f"print(version({distribution!r}))"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _profile_version_mismatches(
    profile: ModelProfile, python: Path
) -> dict[str, tuple[str | None, str]]:
    expected = {
        name: version
        for name, version in (
            ("torch", profile.torch_version),
            ("transformers", profile.transformers_version),
            ("torchvision", profile.torchvision_version),
        )
        if version is not None
    }
    return {
        name: (installed, required)
        for name, required in expected.items()
        if not _installed_version_matches(
            name,
            installed := _installed_distribution_version(python, name),
            required,
        )
    }


def _installed_version_matches(
    distribution: str, installed: str | None, required: str
) -> bool:
    if installed is None:
        return False
    # CUDA wheels report a PEP 440 local suffix such as ``2.6.0+cu124`` even
    # though the pinned public Torch version and pip requirement are ``2.6.0``.
    # The CUDA index is validated separately, so this is not a stale package.
    if distribution in {"torch", "torchvision"}:
        installed = installed.partition("+")[0]
    return installed == required


def repair_profile_dependencies(
    profile: ModelProfile,
    python: Path,
    cuda_index: str,
    mismatches: Mapping[str, tuple[str | None, str]],
) -> None:
    """Update only stale pinned runtime packages in an existing model venv."""
    details = ", ".join(
        f"{name} {installed or 'missing'} -> {required}"
        for name, (installed, required) in mismatches.items()
    )
    print(f"Updating stale {profile.model_id} environment pins: {details}", flush=True)
    data_root = Path(os.environ.get("DLLM_DATA_ROOT", REPO_ROOT / ".data"))
    cache_dir = Path(os.environ.get("DLLM_PIP_CACHE_DIR", data_root / "pip-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    install_env = os.environ.copy()
    install_env["PIP_CACHE_DIR"] = str(cache_dir)

    if "torch" in mismatches:
        if cuda_index not in profile.torch_cuda_indexes:
            supported = ", ".join(profile.torch_cuda_indexes)
            raise SystemExit(
                f"torch {profile.torch_version} for {profile.model_id} has no "
                f"{cuda_index} wheel; supported indexes: {supported}"
            )
        run(
            [python, "-m", "pip", "install", "--upgrade", f"torch=={profile.torch_version}",
             "--index-url", f"https://download.pytorch.org/whl/{cuda_index}"],
            env=install_env,
        )
    if profile.torchvision_version is not None and (
        "torchvision" in mismatches or "torch" in mismatches
    ):
        run(
            [python, "-m", "pip", "install", "--upgrade",
             f"torchvision=={profile.torchvision_version}",
             "--index-url", f"https://download.pytorch.org/whl/{cuda_index}"],
            env=install_env,
        )
    if "transformers" in mismatches:
        run(
            [python, "-m", "pip", "install", "--upgrade",
             f"transformers=={profile.transformers_version}", "accelerate==1.14.0",
             "safetensors>=0.8.0", "sentencepiece"],
            env=install_env,
        )
    if "gpu" in profile.extras.split(","):
        # Old environments may still contain the deprecated `pynvml`
        # distribution. The maintained `nvidia-ml-py` package exports the
        # same import module and is what the gpu extra now requires.
        subprocess.run(
            [str(python), "-m", "pip", "uninstall", "-y", "pynvml"],
            cwd=REPO_ROOT,
            env=install_env,
            check=False,
        )
        run(
            [python, "-m", "pip", "install", "--upgrade", "nvidia-ml-py>=12.0"],
            env=install_env,
        )
    run([python, "-m", "pip", "check"], env=install_env)
    run([python, "-c", _IMPORT_CHECK], env=install_env)


def model_environment(profile: ModelProfile) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        DLLM_MODEL=profile.model_id,
        DLLM_MODEL_CONFIG=profile.model_config,
        DLLM_VENV=str(venv_dir(profile)),
    )
    if "gpu" in profile.extras.split(",") and "PYTORCH_CUDA_ALLOC_CONF" not in environment:
        # PyTorch's own OOM error message recommends this: a long sweep
        # (a dataset's full sample count, many diffusion steps each)
        # allocates and frees many similarly-but-not-identically-shaped
        # tensors in one long-lived process, which fragments the default
        # segment-based allocator over time — observed in practice as a
        # model's `best` variant (more steps/sample, so more allocations)
        # running an entire dataset successfully, and the very next variant
        # (`fast`, strictly less GPU work per sample) OOM-ing on its very
        # first sample right after. `expandable_segments` lets an existing
        # CUDA memory segment grow in place instead of needing a whole new
        # one to fit a request nothing else fits, which avoids that failure
        # mode without ever needing to clear the cache mid-run — so, unlike
        # a runtime `empty_cache()` + retry, this has no cold-allocation
        # cost to land inside any sample's measured time. Only sets a
        # default; an operator's own `PYTORCH_CUDA_ALLOC_CONF` always wins.
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return environment


def check_environment(profile: ModelProfile, python: Path) -> None:
    run([python, "-m", "pip", "check"])
    code = (
        "from dllm_bench.registry import build_model_adapter, list_model_variants\n"
        f"config = {profile.model_config!r}\n"
        "for variant in list_model_variants(config):\n"
        "    adapter = build_model_adapter(config, variant=variant)\n"
        "    print(f'{adapter.name}:{adapter.config_name} adapter OK')\n"
    )
    run([python, "-c", code], env=model_environment(profile))
    if profile.model_id == "diffusiongemma":
        run([python, "-c", "from transformers import DiffusionGemmaForBlockDiffusion, Gemma4Processor; print('DiffusionGemma classes OK')"])
    elif profile.model_id == "gemma4_26b_a4b":
        run([python, "-c", "from transformers import AutoModelForMultimodalLM, Gemma4Processor; print('Gemma 4 classes OK')"])


def benchmark_arguments(profile: ModelProfile) -> list[str]:
    data_source = os.environ.get("DATA_SOURCE", "demo")
    if data_source not in {"demo", "real"}:
        raise SystemExit(f"unknown DATA_SOURCE={data_source!r}; use 'demo' or 'real'")
    if profile.model_id == "w1" and not os.environ.get("W1_API_BASE_URL"):
        raise SystemExit("W1_API_BASE_URL must be set before running W1")

    stage = os.environ.get("STAGE", "all")
    arguments = [
        "-m", "dllm_bench.cli", "matrix",
        "--experiment-config", os.environ.get("EXPERIMENT_CONFIG", "configs/experiments/full_matrix.yaml"),
        "--model", profile.model_id,
        "--stage", stage,
        "--demo" if data_source == "demo" else "--no-demo",
        "--output-root", os.environ.get("OUTPUT_ROOT", "output"),
        "--measure-compute" if os.environ.get("MEASURE_COMPUTE", "0") == "1" else "--no-measure-compute",
        "--require-all-metrics" if os.environ.get("REQUIRE_ALL_METRICS", "0") == "1" else "--allow-missing-metrics",
        "--resume" if os.environ.get("RESUME", "1") == "1" else "--no-resume",
    ]
    if stage in {"visualize", "all"}:
        arguments.extend(
            ["--n-representative", os.environ.get("N_REPRESENTATIVE", "3")]
        )
    if os.environ.get("N_SAMPLES"):
        arguments.extend(["--n-samples", os.environ["N_SAMPLES"]])
    if os.environ.get("DATASETS"):
        for dataset_name in os.environ["DATASETS"].split(","):
            if dataset_name.strip():
                arguments.extend(["--dataset", dataset_name.strip()])
    if os.environ.get("MATRIX_VARIANTS"):
        arguments.extend(["--variants", os.environ["MATRIX_VARIANTS"]])
    return arguments


def build_parser(profile: ModelProfile) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Manage the isolated environment for {profile.model_id}."
    )
    parser.add_argument("action", nargs="?", default="run", choices=("setup", "check", "prepare", "run"))
    parser.add_argument("--cuda-index", default=os.environ.get("CUDA_INDEX", "cu124"), choices=CUDA_INDEXES)
    return parser


def main(model_id: str, argv: Sequence[str] | None = None) -> int:
    profile = PROFILES[model_id]
    args = build_parser(profile).parse_args(argv)
    if args.action == "setup":
        setup_environment(profile, args.cuda_index)
        return 0

    if args.action == "prepare":
        python = venv_python(venv_dir(profile))
        if not python.is_file():
            raise SystemExit(
                f"model environment is missing: {venv_dir(profile)}; "
                f"run `python setup_venv.py -m {profile.model_id}` first"
            )
    else:
        python = ensure_environment(profile, args.cuda_index)
    if args.action == "check":
        check_environment(profile, python)
    elif args.action == "prepare":
        command = [python, "prepare_model.py", "--model-config", profile.model_config]
        if os.environ.get("PREPARE_MODEL_VARIANT"):
            command.extend(["--variant", os.environ["PREPARE_MODEL_VARIANT"]])
        if os.environ.get("PREPARE_MODEL_VARIANTS"):
            command.extend(["--variants", os.environ["PREPARE_MODEL_VARIANTS"]])
        run(command, env=model_environment(profile))
    else:
        repair_project_installation(profile, python)
        run([python, *benchmark_arguments(profile)], env=model_environment(profile))
    return 0
