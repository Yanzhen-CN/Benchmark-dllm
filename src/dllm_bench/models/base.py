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

from abc import ABC, abstractmethod

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

    @abstractmethod
    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        """Run one sample. Must not itself measure timing/energy/VRAM."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            with measure_wall_clock() as wc, measure_energy_joules() as eh, measure_peak_vram_gb() as vh:
                result = self._generate_core(request)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure -> Run Status
            status = RunStatus.OOM if _looks_like_oom(exc) else RunStatus.FAILED
            return GenerationResult(
                request=request,
                output_text="",
                status=status,
                error_message=str(exc),
            )

        result.timing = TimingResult(wall_clock_seconds=wc.seconds or 0.0, source="measured")
        result.energy_joules = eh.joules if eh.available else None
        result.peak_vram_gb = vh.peak_gb if vh.available else None
        return result

    def profile_compute(self, request: GenerationRequest) -> ComputeHandle:
        """Separate profiling replay for ComputePerSample (Appendix B): run the
        same sample again under FLOP counting instead of inside the timed
        window, since FLOP counting overhead would otherwise contaminate
        Time per Sample."""
        with measure_compute_tflops() as handle:
            self._generate_core(request)
        return handle

    def warm(self) -> None:
        """Pre-load model weights without generating anything (see
        ``prepare_model.py``). A no-op for adapters with nothing to load
        (e.g. the mock backend) rather than an error."""
        ensure_loaded = getattr(self, "_ensure_loaded", None)
        if ensure_loaded is not None:
            ensure_loaded()


def _looks_like_oom(exc: Exception) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "out of memory" in str(exc).lower()
