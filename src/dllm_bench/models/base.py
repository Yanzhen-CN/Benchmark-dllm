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
from contextlib import ExitStack, contextmanager
from time import perf_counter

from ..interfaces import ForwardProfile, GenerationRequest, GenerationResult, RunStatus, TimingResult
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

    def _trace_instrumentation_enabled(self) -> bool:
        return not getattr(self, "_suppress_trace_instrumentation", False)

    @contextmanager
    def _forward_phase(self, phase: str):
        previous = getattr(self, "_active_forward_phase", "denoise")
        self._active_forward_phase = phase
        try:
            yield
        finally:
            self._active_forward_phase = previous

    @staticmethod
    def _synchronize_forward_device() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            return

    @contextmanager
    def _capture_model_forwards(
        self,
        request: GenerationRequest,
        compute_handle: ComputeHandle | None = None,
    ):
        enabled = bool(request.config.get("step_profiling"))
        model = getattr(self, "_model", None)
        if not enabled or model is None or not hasattr(model, "register_forward_pre_hook"):
            yield
            return

        pending: list[tuple[str, float | None, float | None, dict]] = []
        capture_time = compute_handle is None

        def before_forward(_module, args, kwargs):
            phase = str(getattr(self, "_active_forward_phase", "denoise"))
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            attention_mask = kwargs.get("attention_mask")
            past_key_values = kwargs.get("past_key_values")
            kv_cache_tokens = None
            get_seq_length = getattr(past_key_values, "get_seq_length", None)
            if callable(get_seq_length):
                try:
                    kv_cache_tokens = int(get_seq_length())
                except (TypeError, ValueError):
                    kv_cache_tokens = None
            input_tokens = (
                int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else None
            )
            attention_tokens = (
                int(attention_mask.shape[-1])
                if hasattr(attention_mask, "shape")
                else (
                    input_tokens + (kv_cache_tokens or 0)
                    if input_tokens is not None
                    else None
                )
            )
            metadata = {
                "input_tokens": input_tokens,
                "kv_cache_tokens": kv_cache_tokens,
                "attention_tokens": attention_tokens,
                "uses_kv_cache": bool(
                    past_key_values is not None and (kv_cache_tokens or 0) > 0
                ),
                "stores_kv": (
                    bool(kwargs["store_kv"])
                    if "store_kv" in kwargs
                    else (
                        bool(kwargs["use_cache"])
                        if "use_cache" in kwargs
                        else None
                    )
                ),
            }
            if capture_time:
                self._synchronize_forward_device()
            started = perf_counter() if capture_time else None
            flops_before = (
                compute_handle.snapshot_tflops() if compute_handle is not None else None
            )
            pending.append((phase, started, flops_before, metadata))

        def after_forward(_module, _args, _kwargs, _output):
            phase, started, flops_before, metadata = pending.pop()
            elapsed = None
            if started is not None:
                self._synchronize_forward_device()
                elapsed = perf_counter() - started
            flops_after = (
                compute_handle.snapshot_tflops() if compute_handle is not None else None
            )
            step_tflops = (
                flops_after - flops_before
                if flops_after is not None and flops_before is not None
                else None
            )
            self._forward_profiles.append(
                ForwardProfile(
                    forward_index=len(self._forward_profiles),
                    phase=phase,
                    wall_clock_seconds=elapsed,
                    compute_tflops=step_tflops,
                    **metadata,
                )
            )

        pre_handle = model.register_forward_pre_hook(before_forward, with_kwargs=True)
        post_handle = model.register_forward_hook(after_forward, with_kwargs=True)
        try:
            yield
        finally:
            pre_handle.remove()
            post_handle.remove()

    def _annotate_last_forward(
        self,
        *,
        accepted_tokens: int,
        active_tokens: int,
        eligible_tokens: int,
    ) -> None:
        for profile in reversed(getattr(self, "_forward_profiles", [])):
            if profile.phase == "denoise" and profile.accepted_tokens is None:
                profile.accepted_tokens = int(accepted_tokens)
                profile.active_tokens = int(active_tokens)
                profile.eligible_tokens = int(eligible_tokens)
                return

    @contextmanager
    def _exclude_from_measurement(self):
        """Pause every formal resource counter around instrumentation work.

        Iterative adapters call this around trace-only entropy conversion,
        tensor copies and decoding.  All measured generation segments are
        accumulated into one sample result, keeping time and energy on the
        same boundary while allowing a complete trace from the same run.
        """
        measurement = getattr(self, "_active_measurement", None)
        was_active = measurement is not None and measurement.active
        if was_active:
            measurement.stop()
        try:
            yield
        finally:
            if was_active:
                measurement.start()

    @abstractmethod
    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        """Run one sample. Must not itself measure timing/energy/VRAM."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        _seed_everything(request.seed)
        if request.config.get("step_profiling"):
            ensure_loaded = getattr(self, "_ensure_loaded", None)
            if callable(ensure_loaded):
                ensure_loaded()
        self._forward_profiles: list[ForwardProfile] = []
        measurement = _SampleMeasurement()
        self._active_measurement = measurement
        result: GenerationResult
        try:
            if not self.deferred_measurement:
                measurement.start()
            previous_trace_suppression = getattr(
                self, "_suppress_trace_instrumentation", False
            )
            if request.config.get("capture_trace") is False:
                self._suppress_trace_instrumentation = True
            try:
                with self._capture_model_forwards(request):
                    result = self._generate_core(request)
            finally:
                self._suppress_trace_instrumentation = previous_trace_suppression
        except Exception as exc:  # noqa: BLE001 - failure is persisted per sample
            status = RunStatus.OOM if _looks_like_oom(exc) else RunStatus.FAILED
            result = GenerationResult(
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
        result.forward_profiles = list(self._forward_profiles)
        if result.forward_profiles:
            result.extra["profiled_model_forwards"] = len(result.forward_profiles)
        return result

    def profile_compute(self, request: GenerationRequest) -> ComputeHandle:
        """Separate profiling replay for ComputePerSample (Appendix B): run the
        same sample again under FLOP counting instead of inside the timed
        window, since FLOP counting overhead would otherwise contaminate
        Time per Sample."""
        _seed_everything(request.seed)
        previous = getattr(self, "_suppress_trace_instrumentation", False)
        self._suppress_trace_instrumentation = True
        self._forward_profiles = []
        try:
            with measure_compute_tflops() as handle:
                with self._capture_model_forwards(request, compute_handle=handle):
                    self._generate_core(request)
        finally:
            self._suppress_trace_instrumentation = previous
        handle.forward_tflops = [
            float(profile.compute_tflops)
            for profile in self._forward_profiles
            if profile.compute_tflops is not None
        ]
        handle.forward_phases = [profile.phase for profile in self._forward_profiles]
        return handle

    def warmup_generation(self, request: GenerationRequest) -> None:
        """Run a short untimed generation to initialize kernels/caches."""
        _seed_everything(request.seed)
        self._active_measurement = None
        previous = getattr(self, "_suppress_trace_instrumentation", False)
        self._suppress_trace_instrumentation = True
        try:
            self._generate_core(request)
        finally:
            self._suppress_trace_instrumentation = previous

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
        self._wall_total = 0.0
        self._wall_seen = False
        self._energy_total = 0.0
        self._energy_seen = False
        self._energy_available = True
        self._peak_vram: float | None = None
        self._vram_available = True

    @property
    def active(self) -> bool:
        return self._stack is not None

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
        if self._wall is not None and self._wall.seconds is not None:
            self._wall_total += self._wall.seconds
            self._wall_seen = True
        if self._energy is None or not self._energy.available or self._energy.joules is None:
            self._energy_available = False
        else:
            self._energy_total += self._energy.joules
            self._energy_seen = True
        if self._vram is None or not self._vram.available or self._vram.peak_gb is None:
            self._vram_available = False
        else:
            self._peak_vram = (
                self._vram.peak_gb
                if self._peak_vram is None
                else max(self._peak_vram, self._vram.peak_gb)
            )

    @property
    def wall_clock_seconds(self) -> float | None:
        return self._wall_total if self._wall_seen else None

    @property
    def energy_joules(self) -> float | None:
        if not self._energy_available or not self._energy_seen:
            return None
        return self._energy_total

    @property
    def peak_vram_gb(self) -> float | None:
        if not self._vram_available:
            return None
        return self._peak_vram


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
