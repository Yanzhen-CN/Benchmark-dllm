"""model_cache.get_or_load: the mechanism that lets P1/P2 (same
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
