"""Deterministic subsampling and the run-metadata record required by section 6:
code commit, model checkpoint, execution path, software/hardware environment,
random seed — all fixed at ``seed = 42``.
"""

from __future__ import annotations

import platform
import random
import subprocess
import sys
import os
from typing import Any, TypeVar

from ..interfaces import ModelAdapter

DEFAULT_SEED = 42

T = TypeVar("T")


def deterministic_sample(items: list[T], n: int, seed: int = DEFAULT_SEED) -> list[T]:
    if n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return result.stdout.decode().strip()
    except Exception:
        return None


def collect_run_metadata(adapter: ModelAdapter, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "seed": DEFAULT_SEED,
        "model": adapter.name,
        "config": adapter.config_name,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_runtime"] = torch.version.cuda
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        metadata["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        metadata["cuda_device_details"] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_memory_bytes = int(properties.total_memory)
            metadata["cuda_device_details"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": total_memory_bytes,
                    "total_memory_gb": total_memory_bytes / (1024 ** 3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
        metadata["cuda_current_device"] = (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        )
    except ImportError:
        metadata["torch_version"] = None
        metadata["cuda_available"] = False

    try:
        import transformers

        metadata["transformers_version"] = transformers.__version__
    except ImportError:
        metadata["transformers_version"] = None

    try:
        from importlib.metadata import version

        metadata["vllm_version"] = version("vllm")
    except Exception:
        metadata["vllm_version"] = None

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            metadata["nvidia_driver_version"] = (
                driver_version.decode()
                if isinstance(driver_version, bytes)
                else str(driver_version)
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        metadata["nvidia_driver_version"] = None

    checkpoint = getattr(adapter, "_model_name", None) or getattr(
        adapter, "_checkpoint", None
    )
    if checkpoint:
        metadata["checkpoint"] = checkpoint
    draft_checkpoint = getattr(adapter, "_draft_model_name", None)
    if draft_checkpoint:
        metadata["draft_checkpoint"] = draft_checkpoint
    loaded_model = getattr(adapter, "_model", None)
    loaded_config = getattr(loaded_model, "config", None)
    checkpoint_revision = getattr(loaded_config, "_commit_hash", None)
    if checkpoint_revision:
        metadata["checkpoint_revision"] = str(checkpoint_revision)
    inference_dtype = getattr(adapter, "_inference_dtype", None)
    if inference_dtype:
        metadata["inference_dtype"] = inference_dtype
    inference_optimizations = getattr(adapter, "inference_optimizations", None)
    if inference_optimizations:
        metadata["inference_optimizations"] = list(inference_optimizations)
    execution_path = getattr(adapter, "execution_path", None)
    if execution_path:
        metadata["execution_path"] = execution_path
    sampling_profile = getattr(adapter, "sampling_profile", None)
    if sampling_profile:
        metadata["sampling_profile"] = sampling_profile
    git_commit = _get_git_commit()
    if git_commit:
        metadata["code_commit"] = git_commit

    if extra:
        metadata.update(extra)
    return metadata
