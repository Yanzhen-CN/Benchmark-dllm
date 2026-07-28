from __future__ import annotations

import sys
from types import SimpleNamespace

import dllm_bench.models.gemma4_ar as gemma4_module
from dllm_bench.models import model_cache
from dllm_bench.models.gemma4_ar import Gemma4ARAdapter


class _FakeModel:
    def __init__(self) -> None:
        self.devices: list[str] = []
        self.eval_calls = 0

    def to(self, device: str):
        self.devices.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())


def test_gemma4_ar_uses_official_multimodal_classes_and_bfloat16(monkeypatch):
    processor_calls: list[str] = []
    model_calls: list[tuple[str, dict]] = []
    processor = object()
    model = _FakeModel()

    class FakeProcessorClass:
        @staticmethod
        def from_pretrained(model_name: str):
            processor_calls.append(model_name)
            return processor

    class FakeModelClass:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            model_calls.append((model_name, kwargs))
            return model

    bfloat16 = object()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16=bfloat16,
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForMultimodalLM=FakeModelClass,
            AutoProcessor=FakeProcessorClass,
        ),
    )
    model_cache.clear()

    first = Gemma4ARAdapter("google/test-gemma4", device="cpu")
    second = Gemma4ARAdapter("google/test-gemma4", device="cpu")
    first._ensure_loaded()
    second._ensure_loaded()

    assert processor_calls == ["google/test-gemma4"]
    assert model_calls == [(
        "google/test-gemma4",
        {"dtype": bfloat16, "low_cpu_mem_usage": True},
    )]
    assert first._tokenizer is second._tokenizer is processor
    assert first._model is second._model is model
    assert model.devices == ["cpu"]
    assert model.eval_calls == 1

    max_memory = {0: 40_000_000_000, "cpu": 100_000_000_000}
    monkeypatch.setattr(
        gemma4_module,
        "cpu_offload_max_memory",
        lambda device: max_memory,
    )
    offload_processor, offload_model = first._load_model_and_tokenizer(
        "cuda:0",
        device_map_auto=True,
    )

    assert offload_processor is processor
    assert offload_model is model
    assert model_calls[-1] == (
        "google/test-gemma4",
        {"dtype": bfloat16, "device_map": "auto", "max_memory": max_memory},
    )
    assert model.devices == ["cpu"]

    model_cache.clear()
