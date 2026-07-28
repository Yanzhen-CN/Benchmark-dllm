"""Process-wide cache of loaded ``(tokenizer, model)`` pairs, keyed by
``(model_name_or_path, device)``.

Best/Fast (or standard/jump/gidd) are different *generation-time* configs
of the *same* checkpoint — the whole reason `configs/models/illada.yaml`
nests them under one file (see README) is to test them together without
paying the model-load cost twice. The CLI achieves the "once per process"
part by sweeping every variant in one invocation
(``cli.py``'s `generate`/`score`/`visualize` default to all variants); this
module achieves the "actually shared" part: building a second adapter
instance for the same checkpoint reuses the first one's in-memory weights
instead of calling `from_pretrained` and `.to(device)` again.

Not meaningful across separate process invocations — it's an in-memory
cache, not a disk one (that's `hf_cache.py`'s job).
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

_CACHE: dict[tuple[str, str], Any] = {}


def get_or_load(model_name_or_path: str, device: str, loader: Callable[[], Any]) -> Any:
    key = (model_name_or_path, device)
    if key not in _CACHE:
        _CACHE[key] = loader()
    return _CACHE[key]


def clear() -> None:
    """Mainly for tests — drops every cached (tokenizer, model) pair."""
    _CACHE.clear()


def evict(model_name_or_path: str, device: str) -> None:
    """Drop one cached (tokenizer, model) pair so the next `get_or_load` call
    for this exact key loads fresh — for replacing an already-loaded model
    with a differently-configured one (e.g. a CPU-offload fallback after a
    genuine capacity OOM), unlike a plain retry that reuses whatever's
    already cached."""
    _CACHE.pop((model_name_or_path, device), None)


def reload_with_offload(
    model_name_or_path: str,
    device: str,
    offload_loader: Callable[[], Any],
    *,
    release_current: Callable[[], None] | None = None,
) -> Any:
    """Evict whatever's cached for this (name, device) key and reload via
    `offload_loader` (which must itself request `device_map="auto"` — this
    function has no HF-loading knowledge of its own), replacing the cache
    entry. Every adapter sharing this checkpoint (e.g. `best` and `fast`)
    sees the offloaded model on its own next `get_or_load` call — the whole
    reason this goes through the shared cache rather than each adapter
    instance separately swapping its own `self._model`."""
    evict(model_name_or_path, device)
    # Evicting the cache is not enough: the adapter that hit the OOM still
    # owns its `self._model` reference. Drop that reference before asking
    # Accelerate to inspect available VRAM, otherwise device_map="auto"
    # sees the old full-GPU copy and needlessly offloads too much (or fails
    # to construct the replacement at all).
    if release_current is not None:
        release_current()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return get_or_load(model_name_or_path, device, offload_loader)


def offloaded_parameter_bytes(model: Any) -> int:
    """Sum of parameter+buffer bytes resident on CPU, once
    `accelerate`'s `device_map="auto"` has moved part of a model to CPU —
    a direct, measurable answer to "how much VRAM did this overflow by"
    (design doc: this also incidentally measures each model's real VRAM
    need at its longest tested input, not just whether it fits). Reflects
    `accelerate`'s own placement decision, which already reserves some
    headroom for activations/KV-cache — not a substitute for profiling
    peak memory directly, but a practical, always-available proxy that
    needs no extra instrumentation around the forward pass itself."""
    total = 0
    for tensor in model.parameters():
        if tensor.device.type == "cpu":
            total += tensor.numel() * tensor.element_size()
    for tensor in model.buffers():
        if tensor.device.type == "cpu":
            total += tensor.numel() * tensor.element_size()
    return total
