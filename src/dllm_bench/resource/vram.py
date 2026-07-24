"""Peak VRAM measurement: PeakVRAM = max_g PeakVRAM_g (Appendix B)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


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
