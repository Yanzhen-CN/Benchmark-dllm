from contextlib import contextmanager
from types import SimpleNamespace

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
