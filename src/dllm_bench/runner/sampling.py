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
    except ImportError:
        metadata["torch_version"] = None
        metadata["cuda_available"] = False

    git_commit = _get_git_commit()
    if git_commit:
        metadata["code_commit"] = git_commit

    if extra:
        metadata.update(extra)
    return metadata
