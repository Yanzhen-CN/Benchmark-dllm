"""GPU energy measurement (Appendix B).

    synchronize_all_gpus()
    energy_before = read_energy()
    run_one_sample()
    synchronize_all_gpus()
    energy_after = read_energy()
    E_sample = sum(energy_after[g] - energy_before[g] for g in all_gpus)

Uses NVML total-energy-consumption counters (mJ, monotonically increasing)
via ``pynvml`` when available. Without NVML access (no GPU, no permission,
non-NVIDIA hardware) energy cannot be measured locally — callers must treat
``None`` as "unavailable", not as zero, and surface it as such in reports.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class EnergyUnavailableError(RuntimeError):
    """Raised internally when NVML energy counters cannot be read."""


def _read_energy_mj_per_gpu() -> dict[int, float]:
    """Read cumulative energy (mJ) for the benchmark's selected GPU(s)."""
    try:
        import pynvml
    except ImportError as exc:
        raise EnergyUnavailableError("pynvml is not installed") from exc

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        readings: dict[int, float] = {}
        for index in _energy_gpu_indices(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            readings[index] = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
        return readings
    except pynvml.NVMLError as exc:
        raise EnergyUnavailableError(str(exc)) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _energy_gpu_indices(device_count: int) -> list[int]:
    """Resolve physical NVML indices without summing unrelated host GPUs.

    Current adapters place the whole model on CUDA logical device 0. The first
    numeric CUDA_VISIBLE_DEVICES entry is its physical NVML index. Complex
    UUID/MIG mappings can be pinned explicitly with DLLM_NVML_GPU_INDICES.
    """
    if device_count <= 0:
        return []
    configured = os.environ.get("DLLM_NVML_GPU_INDICES")
    if configured:
        try:
            indices = [int(value.strip()) for value in configured.split(",") if value.strip()]
        except ValueError as exc:
            raise EnergyUnavailableError(
                f"invalid DLLM_NVML_GPU_INDICES={configured!r}"
            ) from exc
        if not indices or any(index < 0 or index >= device_count for index in indices):
            raise EnergyUnavailableError(
                f"DLLM_NVML_GPU_INDICES={configured!r} is outside 0..{device_count - 1}"
            )
        return list(dict.fromkeys(indices))

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible and cuda_visible not in {"-1", "none", "None"}:
        first = cuda_visible.split(",")[0].strip()
        if first.isdigit():
            physical = int(first)
            if physical < device_count:
                return [physical]
            # Some containers remap the assigned physical device to NVML 0.
            if device_count == 1:
                return [0]
    return [0]


@dataclass
class EnergyHandle:
    joules: float | None = None
    available: bool = True
    _before_mj: dict[int, float] = field(default_factory=dict, repr=False)


@contextmanager
def measure_energy_joules(*, synchronize: bool = True) -> Iterator[EnergyHandle]:
    """Context manager mirroring :func:`resource.timing.measure_wall_clock`.

    Sets ``handle.available = False`` and ``handle.joules = None`` instead of
    raising when NVML energy counters are not accessible, so a missing local
    energy-counter permission degrades a run's ``Energy per Sample`` field
    rather than crashing the whole benchmark.
    """
    from .timing import synchronize_all_gpus

    handle = EnergyHandle()
    if synchronize:
        synchronize_all_gpus()
    try:
        handle._before_mj = _read_energy_mj_per_gpu()
    except EnergyUnavailableError:
        handle.available = False

    try:
        yield handle
    finally:
        if handle.available:
            if synchronize:
                synchronize_all_gpus()
            try:
                after_mj = _read_energy_mj_per_gpu()
                total_mj = sum(
                    after_mj[gpu] - handle._before_mj[gpu] for gpu in handle._before_mj
                )
                handle.joules = total_mj / 1000.0
            except EnergyUnavailableError:
                handle.available = False
                handle.joules = None


def energy_per_sample(window_total_energy_joules: float, n_samples: int) -> float:
    """EnergyPerSample = WindowTotalEnergy / N (Appendix B batched fallback)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return window_total_energy_joules / n_samples
