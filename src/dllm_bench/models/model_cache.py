"""Process-wide cache of loaded ``(tokenizer, model)`` pairs.

Best/Fast variants share one checkpoint and differ only in generation-time
configuration. The cache keeps exactly one normally loaded model per
``(checkpoint, device)`` for the lifetime of the process. It deliberately
does not reload, migrate, or CPU-offload a model after CUDA OOM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_CACHE: dict[tuple[str, str], Any] = {}


def get_or_load(model_name_or_path: str, device: str, loader: Callable[[], Any]) -> Any:
    key = (model_name_or_path, device)
    if key not in _CACHE:
        _CACHE[key] = loader()
    return _CACHE[key]


def clear() -> None:
    """Drop cached references; used by isolated unit tests."""
    _CACHE.clear()
