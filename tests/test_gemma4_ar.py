from __future__ import annotations

import sys
from types import SimpleNamespace

from dllm_bench.interfaces import PositionState
from dllm_bench.models import model_cache
from dllm_bench.models.gemma4_ar import Gemma4ARAdapter, _build_ar_trace
from dllm_bench.registry import build_model_adapter


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"[{token_id}]" for token_id in ids)


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


def test_build_gemma4_ar_trace_commits_one_token_per_forward():
    trace = _build_ar_trace([10, 20, 30], _Tokenizer())

    assert len(trace) == 3
    assert trace[0].token_ids == [10, -1, -1]
    assert trace[0].position_states == [
        PositionState.ACCEPTED,
        PositionState.MASKED,
        PositionState.MASKED,
    ]
    assert trace[0].committed_positions == [0]
    assert trace[2].token_ids == [10, 20, 30]
    assert trace[2].committed_positions == [2]


def test_gemma4_model_config_builds_official_non_thinking_adapter():
    adapter = build_model_adapter("configs/models/gemma4_26b_a4b.yaml")

    assert isinstance(adapter, Gemma4ARAdapter)
    assert adapter.name == "gemma4_26b_a4b"
    assert adapter.config_name == "ar-baseline"
    assert adapter._enable_thinking is False


def test_gemma4_ar_loads_once_in_native_bfloat16_without_offload(monkeypatch):
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
    assert first._processor is second._processor is processor
    assert first._model is second._model is model
    assert model.devices == ["cpu"]
    assert model.eval_calls == 1

    model_cache.clear()
