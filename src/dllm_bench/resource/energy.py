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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class EnergyUnavailableError(RuntimeError):
    """Raised internally when NVML energy counters cannot be read."""


def _read_energy_mj_per_gpu() -> dict[int, float]:
    """Read the cumulative energy counter (mJ) for every visible GPU."""
    try:
        import pynvml
    except ImportError as exc:
        raise EnergyUnavailableError("pynvml is not installed") from exc

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        readings: dict[int, float] = {}
        for index in range(device_count):
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


@dataclass
class EnergyHandle:
    joules: float | None = None
    available: bool = True
    _before_mj: dict[int, float] = field(default_factory=dict, repr=False)


@contextmanager
def measure_energy_joules() -> Iterator[EnergyHandle]:
    """Context manager mirroring :func:`resource.timing.measure_wall_clock`.

    Sets ``handle.available = False`` and ``handle.joules = None`` instead of
    raising when NVML energy counters are not accessible, so a missing local
    energy-counter permission degrades a run's ``Energy per Sample`` field
    rather than crashing the whole benchmark.
    """
    from .timing import synchronize_all_gpus

    handle = EnergyHandle()
    synchronize_all_gpus()
    try:
        handle._before_mj = _read_energy_mj_per_gpu()
    except EnergyUnavailableError:
        handle.available = False

    try:
        yield handle
    finally:
        if handle.available:
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
