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

# CPU offload is only reached after a real capacity OOM. Limit model weights
# to half of the selected GPU so long-context KV/activation memory has actual
# headroom; plain `device_map="auto"` only fits weights and can put the whole
# checkpoint back on GPU, immediately reproducing the same runtime OOM.
CPU_OFFLOAD_GPU_MEMORY_FRACTION = 0.50


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


def cpu_offload_max_memory(
    device: str,
    *,
    gpu_fraction: float = CPU_OFFLOAD_GPU_MEMORY_FRACTION,
    detected_max_memory: dict[Any, int] | None = None,
) -> dict[Any, int] | None:
    """Build an Accelerate `max_memory` map with generation headroom.

    Returns ``None`` for non-CUDA execution. Only the selected logical GPU
    and CPU are exposed, preventing an automatic multi-GPU placement from
    silently changing this benchmark's single-device measurement protocol.
    """
    if not 0 < gpu_fraction < 1:
        raise ValueError("gpu_fraction must be between 0 and 1")

    import torch

    if not torch.cuda.is_available():
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    gpu_index = resolved.index
    if gpu_index is None:
        gpu_index = torch.cuda.current_device()

    if detected_max_memory is None:
        from accelerate.utils import get_max_memory

        detected_max_memory = get_max_memory()

    total_bytes = int(torch.cuda.get_device_properties(gpu_index).total_memory)
    gpu_budget = int(total_bytes * gpu_fraction)
    detected_gpu_budget = detected_max_memory.get(gpu_index)
    if detected_gpu_budget is not None:
        gpu_budget = min(gpu_budget, int(detected_gpu_budget))

    limits: dict[Any, int] = {gpu_index: gpu_budget}
    cpu_budget = detected_max_memory.get("cpu")
    if cpu_budget is not None:
        limits["cpu"] = int(cpu_budget)
    return limits


def offloaded_parameter_bytes(model: Any) -> int:
    """Sum parameter+buffer bytes assigned to CPU by Accelerate.

    CPU-offloaded tensors can be physically reported as either ``cpu`` or
    ``meta`` (the latter is backed by Accelerate's CPU weights map and moved
    into the execution device by hooks). Consult ``hf_device_map`` for those
    meta tensors rather than incorrectly recording zero CPU bytes.

    Once `accelerate`'s `device_map="auto"` has moved part of a model to CPU,
    a direct, measurable answer to "how much VRAM did this overflow by"
    (design doc: this also incidentally measures each model's real VRAM
    need at its longest tested input, not just whether it fits). Reflects
    `accelerate`'s own placement decision, which already reserves some
    headroom for activations/KV-cache — not a substitute for profiling
    peak memory directly, but a practical, always-available proxy that
    needs no extra instrumentation around the forward pass itself."""
    device_map = getattr(model, "hf_device_map", {}) or {}

    def assigned_to_cpu(name: str, tensor: Any) -> bool:
        if tensor.device.type == "cpu":
            return True
        if tensor.device.type != "meta":
            return False
        best_prefix = None
        for prefix in device_map:
            if prefix == "" or name == prefix or name.startswith(f"{prefix}."):
                if best_prefix is None or len(prefix) > len(best_prefix):
                    best_prefix = prefix
        return best_prefix is not None and str(device_map[best_prefix]) == "cpu"

    def named_tensors(named_method: str, plain_method: str):
        method = getattr(model, named_method, None)
        if callable(method):
            yield from method()
        else:
            for index, tensor in enumerate(getattr(model, plain_method)()):
                yield str(index), tensor

    total = 0
    for name, tensor in named_tensors("named_parameters", "parameters"):
        if assigned_to_cpu(name, tensor):
            total += tensor.numel() * tensor.element_size()
    for name, tensor in named_tensors("named_buffers", "buffers"):
        if assigned_to_cpu(name, tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def evict_cpu_offloaded_cuda_models() -> int:
    """Drop CPU-offloaded CUDA cache entries at a dataset boundary.

    Best/Fast still share the same placement within one dataset sweep. The
    next dataset must start from its own all-GPU attempt, otherwise a RULER
    OOM would silently force a later, smaller dataset to inherit slower CPU
    placement even if it fits entirely on GPU.
    """
    keys_to_evict: list[tuple[str, str]] = []
    for key, cached in _CACHE.items():
        if not str(key[1]).startswith("cuda"):
            continue
        try:
            model = cached[1]
        except (TypeError, IndexError):
            continue
        if offloaded_parameter_bytes(model) > 0:
            keys_to_evict.append(key)

    for key in keys_to_evict:
        _CACHE.pop(key, None)
    if keys_to_evict:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return len(keys_to_evict)
