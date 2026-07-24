"""Appendix B timing protocol: GPU-synced wall-clock time around one sample.

    synchronize_all_gpus()
    time_before = read_time()
    run_one_sample()
    synchronize_all_gpus()
    time_after = read_time()
    T_sample = time_after - time_before

Model loading, warmup, data loading, tokenization, scoring and file writes
must stay outside the measured window — callers are responsible for putting
only the actual generation call inside :func:`measure_wall_clock`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


def synchronize_all_gpus() -> None:
    """Block until all CUDA devices finish pending work. No-op without torch/CUDA."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


@dataclass
class WallClockHandle:
    """Mutable holder so the caller can read ``seconds`` after the `with` block exits."""

    seconds: float | None = None


@contextmanager
def measure_wall_clock() -> Iterator[WallClockHandle]:
    handle = WallClockHandle()
    synchronize_all_gpus()
    time_before = time.perf_counter()
    try:
        yield handle
    finally:
        synchronize_all_gpus()
        time_after = time.perf_counter()
        handle.seconds = time_after - time_before


def energy_per_sample_from_window(window_total_energy_joules: float, n_samples: int) -> float:
    """EnergyPerSample = WindowTotalEnergy / N, for batch-size-1 serial windows
    used when a single sample's measurement window is too short (Appendix B)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return window_total_energy_joules / n_samples
