"""Wiring tests for the Dream adapter (models/dream.py). Unlike
illada.py/dg.py, Dream's own algorithm isn't locally verified (no reference
implementation was available — see the module docstring), so these only
check the *data flow*: `diffusion_generate` gets called with the expected
arguments, `output.history`/`output.sequences` get correctly sliced into the
generation region and handed to the shared snapshot-diffing utility, and a
missing `mask_token_id` fails clearly instead of silently misbehaving.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dllm_bench.models.dream import DreamAdapter
from dllm_bench.models.hf_diffusion import DiffusionStepConfig

MASK = 99


class _FakeEncoded(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    mask_token_id = MASK

    def __call__(self, text, return_tensors="pt"):
        return _FakeEncoded(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{i}>" for i in ids)


class _FakeDreamModel:
    def __init__(self):
        self.last_call_kwargs = None
        from types import SimpleNamespace

        self.config = SimpleNamespace()

    def diffusion_generate(self, input_ids, **kwargs):
        self.last_call_kwargs = kwargs
        from types import SimpleNamespace

        prompt_len = input_ids.shape[1]
        # 2-step history: step 0 fully masked, step 1 fully resolved.
        history = [
            torch.cat([input_ids, torch.tensor([[MASK, MASK]])], dim=1),
            torch.cat([input_ids, torch.tensor([[10, 11]])], dim=1),
        ]
        sequences = history[-1]
        return SimpleNamespace(sequences=sequences, history=history)


def _make_adapter(model_mask_token_id=None) -> DreamAdapter:
    step_config = DiffusionStepConfig(gen_length=2, steps=2, extra={"alg": "entropy"})
    adapter = DreamAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeDreamModel()
    adapter._tokenizer = _FakeTokenizer()
    adapter._device = "cpu"
    if model_mask_token_id is not None:
        adapter._model.config.mask_token_id = model_mask_token_id
    return adapter


def test_dream_passes_step_config_extras_into_diffusion_generate():
    adapter = _make_adapter()
    adapter._run_denoising("prompt", adapter._step_config)

    kwargs = adapter._model.last_call_kwargs
    assert kwargs["steps"] == 2
    assert kwargs["alg"] == "entropy"
    assert kwargs["max_new_tokens"] == 2
    assert kwargs["output_history"] is True
    assert kwargs["return_dict_in_generate"] is True


def test_dream_builds_trace_from_history_via_shared_snapshot_diff():
    adapter = _make_adapter()
    output_text, trace, final_valid_length = adapter._run_denoising("prompt", adapter._step_config)

    assert final_valid_length == 2
    assert len(trace) == 2
    assert trace[0].committed_positions == []  # both still masked
    assert trace[1].committed_positions == [0, 1]  # both resolved
    assert output_text == "<10><11>"


def test_dream_uses_tokenizer_mask_token_id_when_present():
    adapter = _make_adapter()
    assert adapter._tokenizer.mask_token_id == MASK
    # doesn't raise, confirms the mask id is actually being picked up
    adapter._run_denoising("prompt", adapter._step_config)


class _FakeTokenizerNoMaskId:
    """Callable/decodable like a real tokenizer, but deliberately has no
    `mask_token_id` attribute at all (unlike `_FakeTokenizer` above)."""

    def __call__(self, text, return_tensors="pt"):
        return _FakeEncoded(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{i}>" for i in ids)


def test_dream_falls_back_to_explicit_extra_mask_token_id():
    step_config = DiffusionStepConfig(gen_length=2, steps=2, extra={"mask_token_id": MASK})
    adapter = DreamAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeDreamModel()
    adapter._tokenizer = _FakeTokenizerNoMaskId()
    adapter._device = "cpu"

    _, trace, _ = adapter._run_denoising("prompt", step_config)
    assert len(trace) == 2


def test_dream_raises_a_clear_error_when_mask_token_id_is_unknowable():
    step_config = DiffusionStepConfig(gen_length=2, steps=2)
    adapter = DreamAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeDreamModel()
    adapter._tokenizer = _FakeTokenizerNoMaskId()
    adapter._device = "cpu"

    with pytest.raises(ValueError, match="mask_token_id"):
        adapter._run_denoising("prompt", step_config)
