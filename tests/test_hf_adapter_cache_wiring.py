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


class _FakeOffloadableTensor:
    """Just enough of the `torch.Tensor` interface for
    `model_cache.offloaded_parameter_bytes` to work without a real GPU:
    `.device.type`, `.numel()`, `.element_size()`."""

    def __init__(self, device_type: str, numel: int = 10, element_size: int = 2):
        from types import SimpleNamespace

        self.device = SimpleNamespace(type=device_type)
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class _FakeHFModel:
    def __init__(self):
        self._fake_parameters: list = []
        self._fake_buffers: list = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(self._fake_parameters)

    def buffers(self):
        return iter(self._fake_buffers)


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
    # Default path: never flagged as offloaded, since device_map wasn't even used.
    assert adapter._cpu_offloaded is False


def test_dreamreasoner_reload_with_cpu_offload_switches_to_device_map_auto(monkeypatch):
    """`_reload_with_cpu_offload` only ever runs reactively (from
    `generate()`'s retry escalation, after a real OOM) — never at ordinary
    `_ensure_loaded()` time, which stays 100%-GPU (previous test)."""
    _, model_calls = _install_counting_fakes(
        monkeypatch, transformers.AutoModelForCausalLM
    )
    adapter = DreamReasonerAdapter(
        "dreamreasoner-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=32),
        config_name="best",
    )
    adapter._ensure_loaded()

    assert adapter._reload_with_cpu_offload() is True
    assert model_calls["kwargs"] == [
        {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True},
        {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "auto"},
    ]


def test_dreamreasoner_releases_old_model_before_auto_device_placement(monkeypatch):
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )
    load_count = 0
    adapter = None

    def fake_model_from_pretrained(name, *args, **kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            assert adapter._model is None
        return _FakeHFModel()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    adapter = DreamReasonerAdapter(
        "dreamreasoner-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=32),
        config_name="best",
    )

    adapter._ensure_loaded()
    adapter._reload_with_cpu_offload()

    assert load_count == 2


def test_dreamreasoner_reload_flags_cpu_offloaded_when_real_bytes_land_on_cpu(monkeypatch):
    def fake_model_from_pretrained(name, *args, **kwargs):
        model = _FakeHFModel()
        model._fake_parameters = [
            _FakeOffloadableTensor("cuda", numel=100, element_size=2),
            _FakeOffloadableTensor("cpu", numel=50, element_size=2),
        ]
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    adapter = DreamReasonerAdapter(
        "dreamreasoner-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=32),
        config_name="best",
    )
    adapter._ensure_loaded()
    adapter._reload_with_cpu_offload()

    assert adapter._cpu_offloaded is True
    assert adapter._cpu_offloaded_bytes == 50 * 2  # only the cpu-resident tensor's bytes


def test_offload_measurement_does_not_count_meta_or_disk_placeholders():
    model = _FakeHFModel()
    model._fake_parameters = [
        _FakeOffloadableTensor("cuda", numel=100, element_size=2),
        _FakeOffloadableTensor("cpu", numel=50, element_size=2),
        _FakeOffloadableTensor("meta", numel=500, element_size=2),
    ]

    assert model_cache.offloaded_parameter_bytes(model) == 50 * 2


def test_dreamreasoner_reload_not_flagged_when_everything_still_lands_on_gpu(monkeypatch):
    """Evicting the old (100%-GPU) copy before reloading frees its memory
    first — it's possible `device_map="auto"` then finds enough room to
    keep everything on the GPU after all. That's a real, valid outcome (the
    reload still succeeded and the caller should still retry), just not one
    that should be flagged as offloaded."""
    def fake_model_from_pretrained(name, *args, **kwargs):
        model = _FakeHFModel()
        model._fake_parameters = [_FakeOffloadableTensor("cuda", numel=100, element_size=2)]
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    adapter = DreamReasonerAdapter(
        "dreamreasoner-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=32),
        config_name="best",
    )
    adapter._ensure_loaded()

    assert adapter._reload_with_cpu_offload() is True
    assert adapter._cpu_offloaded is False


def test_dreamreasoner_fast_inherits_bests_offload_without_hitting_its_own_oom(monkeypatch):
    """Matches the real CLI flow: `best` runs its entire sample sweep (and
    any OOM-recovery reload within it) before `fast` is even constructed —
    `fast`'s *first* `_ensure_loaded()` call should see the already
    -offloaded shared cache entry directly, no separate OOM/reload of its
    own required."""
    def fake_model_from_pretrained(name, *args, **kwargs):
        model = _FakeHFModel()
        model._fake_parameters = [_FakeOffloadableTensor("cpu", numel=10, element_size=2)]
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    best = DreamReasonerAdapter(
        "shared-dreamreasoner-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=32),
        config_name="best",
    )
    best._ensure_loaded()
    assert best._reload_with_cpu_offload() is True  # simulates best's own OOM+recovery
    assert best._cpu_offloaded is True

    # Only now does `fast` get constructed and load for the first time —
    # same order the CLI's variant-sweep loop actually runs in.
    fast = DreamReasonerAdapter(
        "shared-dreamreasoner-checkpoint",
        DiffusionStepConfig(gen_length=64, block_length=32, steps_per_block=16),
        config_name="fast",
    )
    fast._ensure_loaded()

    assert fast._model is best._model
    assert fast._cpu_offloaded is True


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


def test_qwen_ar_reload_with_cpu_offload_switches_to_device_map_auto(monkeypatch):
    """The CPU-offload recovery path (see `models/base.py`'s
    `_reload_with_cpu_offload`) is wired uniformly into every model in this
    benchmark, not special-cased to just the diffusion models that happened
    to need it first."""
    _, model_calls = _install_counting_fakes(monkeypatch, transformers.AutoModelForCausalLM)
    adapter = QwenARAdapter("qwen-checkpoint")
    adapter._ensure_loaded()

    assert adapter._reload_with_cpu_offload() is True
    assert model_calls["kwargs"] == [{}, {"device_map": "auto"}]


def test_diffusiongemma_reload_with_cpu_offload_switches_to_device_map_auto(monkeypatch):
    from transformers import DiffusionGemmaForBlockDiffusion

    _, model_calls = _install_counting_fakes(
        monkeypatch, DiffusionGemmaForBlockDiffusion, tokenizer_class=transformers.AutoProcessor
    )
    adapter = DiffusionGemmaAdapter("diffusiongemma-checkpoint")
    adapter._ensure_loaded()

    assert adapter._reload_with_cpu_offload() is True
    assert model_calls["kwargs"] == [{}, {"device_map": "auto"}]


def test_illada_warm_uses_the_same_cache_path(monkeypatch):
    _, model_calls = _install_counting_fakes(monkeypatch, transformers.AutoModel)

    best = IlladaAdapter("warm-checkpoint", DiffusionStepConfig(gen_length=64), config_name="best")
    fast = IlladaAdapter("warm-checkpoint", DiffusionStepConfig(gen_length=64), config_name="fast")

    best.warm()
    fast.warm()

    assert model_calls["n"] == 1
