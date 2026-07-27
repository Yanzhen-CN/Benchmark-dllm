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


PROFILES: Mapping[str, ModelProfile] = {
    "qwen3_4b": ModelProfile(
        "qwen3_4b", "qwen3_4b", "configs/models/qwen3_4b.yaml",
        "dev,hf,gpu", "2.6.0", "5.14.1", ("cu118", "cu124", "cu126"),
    ),
    "illada": ModelProfile(
        "illada", "illada", "configs/models/illada.yaml",
        "dev,hf,gpu", "2.6.0", "4.57.1", ("cu118", "cu124", "cu126"),
    ),
    "dreamreasoner": ModelProfile(
        "dreamreasoner", "dreamreasoner", "configs/models/dreamreasoner.yaml",
        "dev,hf,gpu", "2.5.1", "4.46.2", ("cu118", "cu121", "cu124"),
    ),
    "diffusiongemma": ModelProfile(
        "diffusiongemma", "diffusiongemma", "configs/models/diffusiongemma.yaml",
        "dev,diffusiongemma,gpu", "2.6.0", "5.14.1", ("cu118", "cu124", "cu126"),
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
        subprocess.run(
            [str(python), "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"],
            cwd=REPO_ROOT, env=install_env, check=False,
        )
    if profile.transformers_version:
        run(
            [python, "-m", "pip", "install", "--upgrade",
             f"transformers=={profile.transformers_version}", "accelerate==1.14.0",
             "safetensors>=0.8.0", "sentencepiece"],
            env=install_env,
        )
    run([python, "-m", "pip", "install", "-e", f".[{profile.extras}]"], env=install_env)
    run([python, "-m", "pip", "check"], env=install_env)
    run([python, "-c", _IMPORT_CHECK])
    print(f"Environment ready: {profile.model_id}\nPath: {directory}")
    return python


_IMPORT_CHECK = """\
import importlib.util
print('Environment import check')
if importlib.util.find_spec('torch'):
    import torch
    print('torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA runtime:', torch.version.cuda)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
if importlib.util.find_spec('transformers'):
    import transformers
    print('transformers:', transformers.__version__)
if importlib.util.find_spec('requests'):
    import requests
    print('requests:', requests.__version__)
"""


def ensure_environment(profile: ModelProfile, cuda_index: str) -> Path:
    python = venv_python(venv_dir(profile))
    return python if python.is_file() else setup_environment(profile, cuda_index)


def model_environment(profile: ModelProfile) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        DLLM_MODEL=profile.model_id,
        DLLM_MODEL_CONFIG=profile.model_config,
        DLLM_VENV=str(venv_dir(profile)),
    )
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
        run([python, "-c", "from transformers import DiffusionGemmaForBlockDiffusion; print('DiffusionGemma class OK')"])


def benchmark_arguments(profile: ModelProfile) -> list[str]:
    data_source = os.environ.get("DATA_SOURCE", "demo")
    if data_source not in {"demo", "real"}:
        raise SystemExit(f"unknown DATA_SOURCE={data_source!r}; use 'demo' or 'real'")
    if profile.model_id == "w1" and not os.environ.get("W1_API_BASE_URL"):
        raise SystemExit("W1_API_BASE_URL must be set before running W1")

    arguments = [
        "-m", "dllm_bench.cli", "matrix",
        "--experiment-config", os.environ.get("EXPERIMENT_CONFIG", "configs/experiments/full_matrix.yaml"),
        "--model", profile.model_id,
        "--stage", os.environ.get("STAGE", "all"),
        "--demo" if data_source == "demo" else "--no-demo",
        "--output-root", os.environ.get("OUTPUT_ROOT", "output"),
        "--measure-compute" if os.environ.get("MEASURE_COMPUTE", "0") == "1" else "--no-measure-compute",
        "--n-representative", os.environ.get("N_REPRESENTATIVE", "3"),
    ]
    if os.environ.get("N_SAMPLES"):
        arguments.extend(["--n-samples", os.environ["N_SAMPLES"]])
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
        run([python, *benchmark_arguments(profile)], env=model_environment(profile))
    return 0
