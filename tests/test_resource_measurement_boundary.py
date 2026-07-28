from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import dllm_bench.models.base as base_module
from dllm_bench.interfaces import GenerationRequest, GenerationResult, RunStatus
from dllm_bench.models.base import BaseModelAdapter


class DeferredAdapter(BaseModelAdapter):
    name = "deferred"
    config_name = "default"
    deferred_measurement = True

    def __init__(self, events):
        self.events = events

    def _generate_core(self, request):
        self.events.append("tokenize")
        self._start_measurement()
        self.events.append("model")
        self._stop_measurement()
        self.events.append("trace-postprocess")
        return GenerationResult(
            request=request,
            output_text="ok",
            status=RunStatus.SUCCESS,
            final_valid_length=1,
        )


class SegmentedAdapter(BaseModelAdapter):
    deferred_measurement = True

    def __init__(self, events):
        self.events = events

    def _generate_core(self, request):
        self._start_measurement()
        self.events.append("model-1")
        with self._exclude_from_measurement():
            self.events.append("trace")
        self.events.append("model-2")
        self._stop_measurement()
        return GenerationResult(
            request=request,
            output_text="ok",
            status=RunStatus.SUCCESS,
            final_valid_length=1,
        )


def _measurement(events, name, handle):
    @contextmanager
    def context():
        events.append(f"enter-{name}")
        try:
            yield handle
        finally:
            events.append(f"exit-{name}")

    return context


def test_deferred_measurement_excludes_tokenization_trace_and_counter_overhead(
    monkeypatch
):
    events = []
    wall = SimpleNamespace(seconds=1.25)
    energy = SimpleNamespace(available=True, joules=4.5)
    vram = SimpleNamespace(available=True, peak_gb=7.0)
    monkeypatch.setattr(
        base_module, "measure_energy_joules", _measurement(events, "energy", energy)
    )
    monkeypatch.setattr(
        base_module, "measure_peak_vram_gb", _measurement(events, "vram", vram)
    )
    monkeypatch.setattr(
        base_module, "measure_wall_clock", _measurement(events, "wall", wall)
    )

    result = DeferredAdapter(events).generate(
        GenerationRequest(prompt="test", max_new_tokens=1, seed=42)
    )

    assert events == [
        "tokenize",
        "enter-energy",
        "enter-vram",
        "enter-wall",
        "model",
        "exit-wall",
        "exit-vram",
        "exit-energy",
        "trace-postprocess",
    ]
    assert result.timing.wall_clock_seconds == 1.25
    assert result.energy_joules == 4.5
    assert result.peak_vram_gb == 7.0


def test_trace_pause_accumulates_aligned_generation_segments(monkeypatch):
    events = []
    wall = SimpleNamespace(seconds=1.25)
    energy = SimpleNamespace(available=True, joules=4.5)
    vram = SimpleNamespace(available=True, peak_gb=7.0)
    monkeypatch.setattr(
        base_module, "measure_energy_joules", _measurement(events, "energy", energy)
    )
    monkeypatch.setattr(
        base_module, "measure_peak_vram_gb", _measurement(events, "vram", vram)
    )
    monkeypatch.setattr(
        base_module, "measure_wall_clock", _measurement(events, "wall", wall)
    )

    result = SegmentedAdapter(events).generate(
        GenerationRequest(prompt="test", max_new_tokens=1, seed=42)
    )

    assert events == [
        "enter-energy", "enter-vram", "enter-wall", "model-1",
        "exit-wall", "exit-vram", "exit-energy", "trace",
        "enter-energy", "enter-vram", "enter-wall", "model-2",
        "exit-wall", "exit-vram", "exit-energy",
    ]
    assert result.timing.wall_clock_seconds == 2.5
    assert result.energy_joules == 9.0
    assert result.peak_vram_gb == 7.0


class _OOMOnceThenSucceedAdapter(BaseModelAdapter):
    """Fails its first `_generate_core` call with an OOM-shaped error, before
    ever starting its own measurement window, then succeeds on the retry.
    `deferred_measurement=True` (like the real HF adapters) means `generate()`
    itself never starts a window either — only `_generate_core`'s own
    `_start_measurement`/`_stop_measurement` calls do, so the failed attempt
    never touches `measure_wall_clock` at all."""

    deferred_measurement = True

    def __init__(self):
        self.generate_core_calls = 0

    def _generate_core(self, request):
        self.generate_core_calls += 1
        if self.generate_core_calls == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        self._start_measurement()
        self._stop_measurement()
        return GenerationResult(
            request=request, output_text="ok", status=RunStatus.SUCCESS, final_valid_length=1
        )


class _AlwaysFailsWithLogicErrorAdapter(BaseModelAdapter):
    def __init__(self):
        self.generate_core_calls = 0

    def _generate_core(self, request):
        self.generate_core_calls += 1
        raise ValueError("not an OOM — a real bug")


def test_generate_retries_once_after_an_oom_and_only_times_the_retry(monkeypatch):
    wall = SimpleNamespace(seconds=1.25)
    monkeypatch.setattr(base_module, "measure_wall_clock", _measurement([], "wall", wall))
    monkeypatch.setattr(
        base_module, "measure_energy_joules",
        _measurement([], "energy", SimpleNamespace(available=False, joules=None)),
    )
    monkeypatch.setattr(
        base_module, "measure_peak_vram_gb",
        _measurement([], "vram", SimpleNamespace(available=False, peak_gb=None)),
    )
    release_calls = []
    monkeypatch.setattr(base_module, "_release_cuda_cache", lambda: release_calls.append(1))

    adapter = _OOMOnceThenSucceedAdapter()
    result = adapter.generate(GenerationRequest(prompt="test", max_new_tokens=1, seed=42))

    assert adapter.generate_core_calls == 2  # first attempt failed, second succeeded
    assert release_calls == [1]  # cache cleared exactly once, between the two attempts
    assert result.status == RunStatus.SUCCESS
    # The failed first attempt never even starts a measurement window (it
    # raises before calling _start_measurement), so this is unambiguously
    # only the successful retry's own timing.
    assert result.timing.wall_clock_seconds == 1.25


def test_generate_does_not_retry_a_non_oom_failure(monkeypatch):
    monkeypatch.setattr(base_module, "_release_cuda_cache", lambda: (_ for _ in ()).throw(
        AssertionError("must not clear the cache for a non-OOM failure")
    ))

    adapter = _AlwaysFailsWithLogicErrorAdapter()
    result = adapter.generate(GenerationRequest(prompt="test", max_new_tokens=1, seed=42))

    assert adapter.generate_core_calls == 1  # no retry attempted
    assert result.status == RunStatus.FAILED


class _OOMTwiceThenSucceedWithOffloadAdapter(BaseModelAdapter):
    """Fails its first two `_generate_core` calls with OOM (simulating
    cache-clearing alone not being enough — a genuine capacity ceiling, not
    fragmentation), then succeeds once `_reload_with_cpu_offload` has
    "reloaded" it (simulated by just flipping `_cpu_offloaded`)."""

    deferred_measurement = True

    def __init__(self, *, supports_offload: bool):
        self.generate_core_calls = 0
        self.reload_calls = 0
        self._supports_offload = supports_offload
        self._cpu_offloaded = False

    def _generate_core(self, request):
        self.generate_core_calls += 1
        if self.generate_core_calls <= 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return GenerationResult(
            request=request, output_text="ok", status=RunStatus.SUCCESS, final_valid_length=1
        )

    def _reload_with_cpu_offload(self):
        self.reload_calls += 1
        if not self._supports_offload:
            return False
        self._cpu_offloaded = True
        self._cpu_offloaded_bytes = 123_456
        return True


class _OOMThreeTimesThenSucceedWithDeeperOffloadAdapter(
    _OOMTwiceThenSucceedWithOffloadAdapter
):
    def __init__(self):
        super().__init__(supports_offload=True)
        self.deeper_reload_calls = 0

    def _generate_core(self, request):
        self.generate_core_calls += 1
        if self.generate_core_calls <= 3:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.16 GiB")
        return GenerationResult(
            request=request,
            output_text="ok",
            status=RunStatus.SUCCESS,
            final_valid_length=1,
        )

    def _reload_with_more_cpu_offload(self):
        self.deeper_reload_calls += 1
        return True


def test_generate_escalates_to_cpu_offload_after_a_second_oom(monkeypatch):
    monkeypatch.setattr(base_module, "_release_cuda_cache", lambda: None)

    adapter = _OOMTwiceThenSucceedWithOffloadAdapter(supports_offload=True)
    result = adapter.generate(GenerationRequest(prompt="test", max_new_tokens=1, seed=42))

    assert adapter.generate_core_calls == 3  # normal, cache-cleared retry, offload retry
    assert adapter.reload_calls == 1  # only escalates to reload once, after the 2nd OOM
    assert result.status == RunStatus.SUCCESS
    assert result.extra["cpu_offloaded"] is True
    assert result.extra["cpu_offloaded_bytes"] == 123_456
    assert result.extra["cpu_offloaded_gib"] == 123_456 / (1024 ** 3)


def test_generate_can_deepen_cpu_offload_after_the_first_offloaded_retry_ooms(
    monkeypatch,
):
    monkeypatch.setattr(base_module, "_release_cuda_cache", lambda: None)
    adapter = _OOMThreeTimesThenSucceedWithDeeperOffloadAdapter()

    result = adapter.generate(
        GenerationRequest(prompt="long context", max_new_tokens=8, seed=42)
    )

    assert result.status == RunStatus.SUCCESS
    assert adapter.generate_core_calls == 4
    assert adapter.reload_calls == 1
    assert adapter.deeper_reload_calls == 1


def test_untimed_warmup_uses_the_same_cpu_offload_recovery(monkeypatch):
    release_calls = []
    monkeypatch.setattr(
        base_module, "_release_cuda_cache", lambda: release_calls.append(1)
    )
    adapter = _OOMTwiceThenSucceedWithOffloadAdapter(supports_offload=True)

    adapter.warmup_generation(
        GenerationRequest(prompt="long context", max_new_tokens=8, seed=42)
    )

    assert adapter.generate_core_calls == 3
    assert adapter.reload_calls == 1
    # One release at warmup start, one after its first failed attempt. The
    # second failure reloads the model through the offload helper instead.
    assert release_calls == [1, 1]
    assert adapter._cpu_offloaded is True


def test_untimed_warmup_can_deepen_cpu_offload(monkeypatch):
    monkeypatch.setattr(base_module, "_release_cuda_cache", lambda: None)
    adapter = _OOMThreeTimesThenSucceedWithDeeperOffloadAdapter()

    adapter.warmup_generation(
        GenerationRequest(prompt="long context", max_new_tokens=8, seed=42)
    )

    assert adapter.generate_core_calls == 4
    assert adapter.reload_calls == 1
    assert adapter.deeper_reload_calls == 1


def test_untimed_warmup_still_raises_when_offload_is_unavailable(monkeypatch):
    monkeypatch.setattr(base_module, "_release_cuda_cache", lambda: None)
    adapter = _OOMTwiceThenSucceedWithOffloadAdapter(supports_offload=False)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        adapter.warmup_generation(
            GenerationRequest(prompt="long context", max_new_tokens=8, seed=42)
        )

    assert adapter.generate_core_calls == 2
    assert adapter.reload_calls == 1


def test_generate_gives_up_as_oom_when_adapter_cannot_reload_with_offload(monkeypatch):
    monkeypatch.setattr(base_module, "_release_cuda_cache", lambda: None)

    adapter = _OOMTwiceThenSucceedWithOffloadAdapter(supports_offload=False)
    result = adapter.generate(GenerationRequest(prompt="test", max_new_tokens=1, seed=42))

    assert adapter.generate_core_calls == 2  # never gets a 3rd attempt
    assert adapter.reload_calls == 1  # asked, but it declined
    assert result.status == RunStatus.OOM
    assert "cpu_offloaded" not in result.extra


def test_generate_never_flags_cpu_offloaded_for_an_adapter_that_never_offloaded():
    result = _OOMOnceThenSucceedAdapter().generate(
        GenerationRequest(prompt="test", max_new_tokens=1, seed=42)
    )
    assert result.status == RunStatus.SUCCESS
    assert "cpu_offloaded" not in result.extra
