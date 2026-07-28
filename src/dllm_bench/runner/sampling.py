"""Deterministic subsampling and the run-metadata record required by section 6:
code commit, model checkpoint, execution path, software/hardware environment,
random seed — all fixed at ``seed = 42``.
"""

from __future__ import annotations

import platform
import random
import subprocess
import sys
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
    except ImportError:
        metadata["torch_version"] = None
        metadata["cuda_available"] = False

    checkpoint = getattr(adapter, "_model_name", None)
    if checkpoint:
        metadata["checkpoint"] = checkpoint
    inference_dtype = getattr(adapter, "_inference_dtype", None)
    if inference_dtype:
        metadata["inference_dtype"] = inference_dtype
    # Always emit this one explicitly (not only when true): a `_meta.json`
    # without it would be ambiguous about whether the adapter ever even
    # checked, versus checked and confirmed a clean 100%-GPU run. See
    # `BaseModelAdapter._reload_with_cpu_offload` (models/base.py) — this
    # only ever becomes true reactively, after a real capacity OOM neither a
    # plain retry nor a cache-cleared one could recover from. Whenever true,
    # timing/energy/compute are not comparable to any other (fully-GPU)
    # run's numbers.
    metadata["cpu_offloaded"] = bool(getattr(adapter, "_cpu_offloaded", False))
    offloaded_bytes = getattr(adapter, "_cpu_offloaded_bytes", None)
    if offloaded_bytes is not None:
        metadata["cpu_offloaded_bytes"] = offloaded_bytes

    git_commit = _get_git_commit()
    if git_commit:
        metadata["code_commit"] = git_commit

    if extra:
        metadata.update(extra)
    return metadata
