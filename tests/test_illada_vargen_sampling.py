"""Algorithm tests for the official iLLaDA ``var_generate`` canvas path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dllm_bench.interfaces import PositionState
from dllm_bench.models.hf_diffusion import DiffusionStepConfig
from dllm_bench.models.illada_vargen import IlladaVarGenAdapter


VOCAB_SIZE = 10


class _LengthAwareModel:
    """Return deterministic logits at the sequence length actually received."""

    def __init__(self) -> None:
        self.call_lengths: list[int] = []

    def __call__(self, x, attention_mask=None):
        assert attention_mask is None
        self.call_lengths.append(int(x.shape[1]))
        logits = torch.zeros(1, x.shape[1], VOCAB_SIZE)
        for position in range(2, x.shape[1]):
            logits[0, position, 3] = float(position)

        class _Output:
            pass

        output = _Output()
        output.logits = logits
        return output


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{token_id}>" for token_id in ids)


def _adapter(*, gen_length=4, block_length=2, steps_per_block=2):
    config = DiffusionStepConfig(
        gen_length=gen_length,
        block_length=block_length,
        steps_per_block=steps_per_block,
    )
    adapter = IlladaVarGenAdapter("unused-checkpoint", config, config_name="test")
    model = _LengthAwareModel()
    adapter._model = model
    adapter._tokenizer = _FakeTokenizer()
    adapter._device = "cpu"
    return adapter, model


def test_vargen_does_not_allocate_or_forward_future_blocks():
    adapter, model = _adapter()

    _, trace, final_valid_length = adapter._run_denoising(
        "prompt", adapter._step_config
    )

    # prompt=2. During block 1 only prompt+2 tokens exist; block 2 is
    # appended only after block 1 has completed.
    assert model.call_lengths == [4, 4, 6, 6]
    assert [len(step.token_ids) for step in trace] == [2, 2, 4, 4]
    assert [len(step.position_states) for step in trace] == [2, 2, 4, 4]
    assert final_valid_length == 4


def test_vargen_commits_each_block_before_appending_the_next():
    adapter, _ = _adapter()

    _, trace, _ = adapter._run_denoising("prompt", adapter._step_config)

    assert sorted(
        trace[0].committed_positions + trace[1].committed_positions
    ) == [0, 1]
    assert sorted(
        trace[2].committed_positions + trace[3].committed_positions
    ) == [2, 3]
    assert trace[1].position_states == [PositionState.ACCEPTED] * 2
    assert trace[-1].position_states == [PositionState.ACCEPTED] * 4
    assert all(
        max(step.entropy_by_position, default=-1) < len(step.token_ids)
        for step in trace
    )


def test_vargen_matches_official_block_divisibility_requirement():
    adapter, _ = _adapter(gen_length=5)

    with pytest.raises(ValueError, match="divisible by block_length"):
        adapter._run_denoising("prompt", adapter._step_config)


def test_vargen_declares_one_complete_block_for_warmup():
    adapter, _ = _adapter(gen_length=4)

    assert adapter.warmup_new_tokens == 32


def test_vargen_rejects_remasking_modes_not_supported_by_official_sampler():
    adapter, _ = _adapter(gen_length=2)
    adapter._step_config.extra["remasking"] = "unknown"

    with pytest.raises(NotImplementedError, match="unknown"):
        adapter._run_denoising("prompt", adapter._step_config)
