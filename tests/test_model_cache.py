"""model_cache.get_or_load: the mechanism that lets Best/Fast (same
checkpoint, different generation config) share one in-memory model instead
of loading it twice — plus BaseModelAdapter.warm()'s hook into it."""

from __future__ import annotations

import pytest

from dllm_bench.models import model_cache
from dllm_bench.models.base import BaseModelAdapter
from dllm_bench.models.mock import MockDiffusionAdapter


@pytest.fixture(autouse=True)
def _clear_model_cache():
    model_cache.clear()
    yield
    model_cache.clear()


def test_get_or_load_calls_loader_once_for_repeated_key():
    calls = []

    def loader():
        calls.append(1)
        return object()

    first = model_cache.get_or_load("checkpoint-a", "cpu", loader)
    second = model_cache.get_or_load("checkpoint-a", "cpu", loader)

    assert first is second
    assert len(calls) == 1


def test_get_or_load_distinguishes_by_model_name():
    calls = []

    def loader():
        calls.append(1)
        return object()

    model_cache.get_or_load("checkpoint-a", "cpu", loader)
    model_cache.get_or_load("checkpoint-b", "cpu", loader)

    assert len(calls) == 2


def test_get_or_load_distinguishes_by_device():
    calls = []

    def loader():
        calls.append(1)
        return object()

    model_cache.get_or_load("checkpoint-a", "cpu", loader)
    model_cache.get_or_load("checkpoint-a", "cuda", loader)

    assert len(calls) == 2


def test_clear_forces_a_fresh_load():
    calls = []

    def loader():
        calls.append(1)
        return object()

    model_cache.get_or_load("checkpoint-a", "cpu", loader)
    model_cache.clear()
    model_cache.get_or_load("checkpoint-a", "cpu", loader)

    assert len(calls) == 2


def test_cpu_offload_memory_map_reserves_half_the_gpu_for_generation(monkeypatch):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=24 * gib),
    )

    limits = model_cache.cpu_offload_max_memory(
        "cuda",
        detected_max_memory={0: 20 * gib, 1: 40 * gib, "cpu": 64 * gib},
    )

    assert limits == {0: 12 * gib, "cpu": 64 * gib}
    assert 1 not in limits  # keep the benchmark on its selected logical GPU


def test_dataset_boundary_evicts_only_cpu_offloaded_cuda_models():
    class Tensor:
        def __init__(self, device_type):
            self.device = type("Device", (), {"type": device_type})()

        def numel(self):
            return 1

        def element_size(self):
            return 2

    class Model:
        def __init__(self, device_type):
            self.tensor = Tensor(device_type)

        def parameters(self):
            return iter([self.tensor])

        def buffers(self):
            return iter([])

    gpu_model = (object(), Model("cuda"))
    offloaded_model = (object(), Model("cpu"))
    cpu_run_model = (object(), Model("cpu"))
    model_cache.get_or_load("gpu", "cuda", lambda: gpu_model)
    model_cache.get_or_load("offloaded", "cuda", lambda: offloaded_model)
    model_cache.get_or_load("cpu-run", "cpu", lambda: cpu_run_model)

    assert model_cache.evict_cpu_offloaded_cuda_models() == 1
    assert model_cache.get_or_load("gpu", "cuda", lambda: None) is gpu_model
    assert model_cache.get_or_load("cpu-run", "cpu", lambda: None) is cpu_run_model
    replacement = (object(), Model("cuda"))
    assert (
        model_cache.get_or_load("offloaded", "cuda", lambda: replacement)
        is replacement
    )


class _FakeLoadingAdapter(BaseModelAdapter):
    """Minimal BaseModelAdapter subclass with an `_ensure_loaded` to verify
    `warm()` dispatches to it (mirrors the real HF adapters' shape
    without needing torch/transformers)."""

    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        self.name = "fake"
        self.config_name = "default"
        self._model_name = model_name_or_path
        self._device = device
        self._model = None
        self.load_calls = 0

    def _ensure_loaded(self):
        def loader():
            self.load_calls += 1
            return "tokenizer", "model"

        self._tokenizer, self._model = model_cache.get_or_load(self._model_name, self._device, loader)

    def _generate_core(self, request):
        raise NotImplementedError


def test_warm_triggers_ensure_loaded():
    adapter = _FakeLoadingAdapter("checkpoint-x")
    assert adapter._model is None
    adapter.warm()
    assert adapter._model == "model"


def test_two_variants_of_the_same_checkpoint_share_one_load():
    best = _FakeLoadingAdapter("illada-checkpoint")
    fast = _FakeLoadingAdapter("illada-checkpoint")

    best.warm()
    fast.warm()

    assert best.load_calls == 1
    assert fast.load_calls == 0  # never had to load; the cache already had it
    assert best._model is fast._model


def test_warm_is_a_no_op_for_adapters_without_ensure_loaded():
    adapter = MockDiffusionAdapter()
    adapter.warm()  # must not raise
