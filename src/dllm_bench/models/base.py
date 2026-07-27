"""Common resource-measurement wrapper shared by every local (non-API) adapter.

Concrete adapters (``hf_ar.py``, ``hf_diffusion.py``, ``mock.py``) only need
to implement :meth:`BaseModelAdapter._generate_core`, i.e. "run one sample and
return a :class:`GenerationResult` with ``timing``/``energy_joules``/
``peak_vram_gb``/``compute_tflops`` left unset". :meth:`generate` fills those
fields in using the Appendix B protocol so no adapter has to re-implement the
before/after + GPU-sync boilerplate. API-backed adapters (W1) do not use this
base class since their timing comes from the provider, not from us — see
``models/w1_api.py``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from contextlib import ExitStack

from ..interfaces import GenerationRequest, GenerationResult, RunStatus, TimingResult
from ..resource.compute import ComputeHandle, measure_compute_tflops
from ..resource.energy import measure_energy_joules
from ..resource.timing import measure_wall_clock
from ..resource.vram import measure_peak_vram_gb


class BaseModelAdapter(ABC):
    name: str = "base"
    config_name: str = "default"
    supports_trace: bool = False
    natively_measures_resources: bool = False
    deferred_measurement: bool = False

    def _start_measurement(self) -> None:
        measurement = getattr(self, "_active_measurement", None)
        if measurement is not None:
            measurement.start()

    def _stop_measurement(self) -> None:
        measurement = getattr(self, "_active_measurement", None)
        if measurement is not None:
            measurement.stop()

    @abstractmethod
    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        """Run one sample. Must not itself measure timing/energy/VRAM."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        _seed_everything(request.seed)
        measurement = _SampleMeasurement()
        self._active_measurement = measurement
        try:
            if not self.deferred_measurement:
                measurement.start()
            result = self._generate_core(request)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure -> Run Status
            status = RunStatus.OOM if _looks_like_oom(exc) else RunStatus.FAILED
            return GenerationResult(
                request=request,
                output_text="",
                status=status,
                error_message=str(exc),
            )
        finally:
            measurement.stop()
            self._active_measurement = None

        result.timing = TimingResult(
            wall_clock_seconds=measurement.wall_clock_seconds or 0.0,
            source="measured",
        )
        result.energy_joules = measurement.energy_joules
        result.peak_vram_gb = measurement.peak_vram_gb
        return result

    def profile_compute(self, request: GenerationRequest) -> ComputeHandle:
        """Separate profiling replay for ComputePerSample (Appendix B): run the
        same sample again under FLOP counting instead of inside the timed
        window, since FLOP counting overhead would otherwise contaminate
        Time per Sample."""
        _seed_everything(request.seed)
        with measure_compute_tflops() as handle:
            self._generate_core(request)
        return handle

    def warmup_generation(self, request: GenerationRequest) -> None:
        """Run a short untimed generation to initialize kernels/caches."""
        _seed_everything(request.seed)
        self._active_measurement = None
        self._generate_core(request)

    def warm(self) -> None:
        """Pre-load weights for explicit runtime checks without generating.

        Checkpoint preparation itself uses Hub snapshot download and does not
        call this method. A no-op for adapters with nothing to load.
        """
        ensure_loaded = getattr(self, "_ensure_loaded", None)
        if ensure_loaded is not None:
            ensure_loaded()


class _SampleMeasurement:
    """One aligned energy/VRAM/time window, started by the concrete adapter."""

    def __init__(self) -> None:
        self._stack: ExitStack | None = None
        self._wall = None
        self._energy = None
        self._vram = None

    def start(self) -> None:
        if self._stack is not None:
            return
        stack = ExitStack()
        # Energy and VRAM setup happen before the innermost wall timer. On
        # exit, the wall timer closes first, so NVML reads/reset bookkeeping
        # cannot inflate the reported generation latency.
        self._energy = stack.enter_context(measure_energy_joules())
        self._vram = stack.enter_context(measure_peak_vram_gb())
        self._wall = stack.enter_context(measure_wall_clock())
        self._stack = stack

    def stop(self) -> None:
        if self._stack is None:
            return
        self._stack.close()
        self._stack = None

    @property
    def wall_clock_seconds(self) -> float | None:
        return self._wall.seconds if self._wall is not None else None

    @property
    def energy_joules(self) -> float | None:
        if self._energy is None or not self._energy.available:
            return None
        return self._energy.joules

    @property
    def peak_vram_gb(self) -> float | None:
        if self._vram is None or not self._vram.available:
            return None
        return self._vram.peak_gb


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _looks_like_oom(exc: Exception) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "out of memory" in str(exc).lower()
