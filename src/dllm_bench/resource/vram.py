"""Peak VRAM measurement: PeakVRAM = max_g PeakVRAM_g (Appendix B)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread


@dataclass
class VramHandle:
    peak_gb: float | None = None
    available: bool = True


@contextmanager
def measure_peak_vram_gb() -> Iterator[VramHandle]:
    handle = VramHandle()
    try:
        import torch
    except ImportError:
        handle.available = False
        yield handle
        return

    if not torch.cuda.is_available():
        handle.available = False
        yield handle
        return

    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)

    try:
        yield handle
    finally:
        peaks = [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ]
        handle.peak_gb = max(peaks) / (1024**3) if peaks else None


@contextmanager
def measure_peak_device_memory_gb(
    *, poll_interval_seconds: float = 0.01
) -> Iterator[VramHandle]:
    """Measure peak total device memory for a separate serving process.

    ``torch.cuda.max_memory_allocated`` only sees allocations owned by the
    current Python process.  A managed vLLM server runs in another process,
    so the DFlash adapter uses NVML device-used memory instead.  Formal runs
    reserve one exclusive GPU; on a shared GPU this value would also include
    unrelated processes and must not be compared.
    """
    handle = VramHandle()
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    nvml_initialized = False
    try:
        import pynvml

        from .energy import _energy_gpu_indices

        pynvml.nvmlInit()
        nvml_initialized = True
        indices = _energy_gpu_indices(pynvml.nvmlDeviceGetCount())
        devices = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in indices]
        if not devices:
            raise RuntimeError("no selected NVML devices")
    except Exception:
        if nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        handle.available = False
        yield handle
        return

    stop = Event()
    peak_bytes = 0

    def sample() -> None:
        nonlocal peak_bytes
        while not stop.is_set():
            try:
                peak_bytes = max(
                    peak_bytes,
                    *(int(pynvml.nvmlDeviceGetMemoryInfo(device).used) for device in devices),
                )
            except Exception:
                handle.available = False
                return
            stop.wait(poll_interval_seconds)

    worker = Thread(target=sample, name="dllm-nvml-vram", daemon=True)
    worker.start()
    try:
        yield handle
    finally:
        stop.set()
        worker.join(timeout=max(1.0, poll_interval_seconds * 4))
        try:
            peak_bytes = max(
                peak_bytes,
                *(int(pynvml.nvmlDeviceGetMemoryInfo(device).used) for device in devices),
            )
        except Exception:
            handle.available = False
        handle.peak_gb = peak_bytes / (1024**3) if handle.available else None
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
