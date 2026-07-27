"""Verifies HF adapters route model loading through the shared cache.
through `models.model_cache` — i.e. that constructing a second adapter for
the same checkpoint (e.g. iLLaDA's `best` and `fast`) does not call
`from_pretrained` a second time. Uses monkeypatched `transformers` entry
points so this doesn't need real weights or a GPU.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dllm_bench.models import model_cache
from dllm_bench.models.diffusiongemma import DiffusionGemmaAdapter
from dllm_bench.models.dreamreasoner import DreamReasonerAdapter
from dllm_bench.models.hf_ar import QwenARAdapter
from dllm_bench.models.hf_diffusion import DiffusionStepConfig
from dllm_bench.models.illada import IlladaAdapter


@pytest.fixture(autouse=True)
def _clear_model_cache():
    model_cache.clear()
    yield
    model_cache.clear()


class _FakeHFModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def _install_counting_fakes(monkeypatch, *auto_classes, tokenizer_class=None):
    tokenizer_calls = {"n": 0}
    model_calls = {"n": 0, "kwargs": []}

    def fake_tokenizer_from_pretrained(name, *args, **kwargs):
        tokenizer_calls["n"] += 1
        return object()

    def fake_model_from_pretrained(name, *args, **kwargs):
        model_calls["n"] += 1
        model_calls["kwargs"].append(kwargs)
        return _FakeHFModel()

    monkeypatch.setattr(
        tokenizer_class or transformers.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained
    )
    for cls in auto_classes:
        monkeypatch.setattr(cls, "from_pretrained", fake_model_from_pretrained)
    return tokenizer_calls, model_calls


def test_illada_best_and_fast_share_one_load(monkeypatch):
    tokenizer_calls, model_calls = _install_counting_fakes(monkeypatch, transformers.AutoModel)

    best = IlladaAdapter(
        "shared-illada-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=32),
        config_name="best",
    )
    fast = IlladaAdapter(
        "shared-illada-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=16),
        config_name="fast",
    )

    best._ensure_loaded()
    fast._ensure_loaded()

    assert model_calls["n"] == 1
    assert tokenizer_calls["n"] == 1
    assert best._model is fast._model
    assert best._tokenizer is fast._tokenizer
    assert model_calls["kwargs"] == [{
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }]
    # each keeps its own step_config despite sharing the loaded model
    assert best._step_config.steps_per_block == 32
    assert fast._step_config.steps_per_block == 16


def test_illada_different_checkpoints_load_independently(monkeypatch):
    tokenizer_calls, model_calls = _install_counting_fakes(monkeypatch, transformers.AutoModel)

    a = IlladaAdapter("checkpoint-a", DiffusionStepConfig(gen_length=64), config_name="best")
    b = IlladaAdapter("checkpoint-b", DiffusionStepConfig(gen_length=64), config_name="best")

    a._ensure_loaded()
    b._ensure_loaded()

    assert model_calls["n"] == 2
    assert a._model is not b._model


def test_dreamreasoner_loads_in_checkpoint_native_bfloat16(monkeypatch):
    _, model_calls = _install_counting_fakes(
        monkeypatch, transformers.AutoModelForCausalLM
    )
    adapter = DreamReasonerAdapter(
        "dreamreasoner-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=32),
        config_name="best",
    )

    adapter._ensure_loaded()

    assert model_calls["kwargs"] == [{
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }]


def test_qwen_ar_adapter_uses_shared_cache(monkeypatch):
    _, model_calls = _install_counting_fakes(monkeypatch, transformers.AutoModelForCausalLM)

    first = QwenARAdapter("shared-qwen-checkpoint")
    second = QwenARAdapter("shared-qwen-checkpoint")

    first._ensure_loaded()
    second._ensure_loaded()

    assert model_calls["n"] == 1
    assert first._model is second._model


def test_diffusiongemma_adapter_uses_shared_cache(monkeypatch):
    from transformers import DiffusionGemmaForBlockDiffusion

    _, model_calls = _install_counting_fakes(
        monkeypatch, DiffusionGemmaForBlockDiffusion, tokenizer_class=transformers.AutoProcessor
    )

    first = DiffusionGemmaAdapter("shared-diffusiongemma-checkpoint")
    second = DiffusionGemmaAdapter("shared-diffusiongemma-checkpoint")

    first._ensure_loaded()
    second._ensure_loaded()

    assert model_calls["n"] == 1
    assert first._model is second._model
    assert first._processor is second._processor


def test_illada_warm_uses_the_same_cache_path(monkeypatch):
    _, model_calls = _install_counting_fakes(monkeypatch, transformers.AutoModel)

    best = IlladaAdapter("warm-checkpoint", DiffusionStepConfig(gen_length=64), config_name="best")
    fast = IlladaAdapter("warm-checkpoint", DiffusionStepConfig(gen_length=64), config_name="fast")

    best.warm()
    fast.warm()

    assert model_calls["n"] == 1
