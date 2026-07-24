"""Compute-per-sample profiling (Appendix B).

    ComputePerSample = TotalFLOPsOfFullGeneration / 1e12   [TFLOP/sample]

The design doc allows this to be measured via a separate profiling replay on
the same input/config/execution path rather than inline with every timed
run (FLOP counting has overhead and would contaminate the timing window).
``measure_compute_tflops`` follows that: call it as its own pass, not nested
inside :func:`resource.timing.measure_wall_clock`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ComputeHandle:
    tflops: float | None = None
    available: bool = True


@contextmanager
def measure_compute_tflops() -> Iterator[ComputeHandle]:
    handle = ComputeHandle()
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        handle.available = False
        yield handle
        return

    counter = FlopCounterMode(display=False)
    try:
        with counter:
            yield handle
    finally:
        total_flops = counter.get_total_flops()
        handle.tflops = total_flops / 1e12
