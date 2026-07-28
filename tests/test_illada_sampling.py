"""Algorithm-level tests for the real iLLaDA sampler port (models/illada.py),
using a fake model with controlled logits — no real weights/GPU needed.
Verifies the ported algorithm itself (block/step loop, confidence-based
top-k selection, gumbel-noise no-op at temperature=0), not just that the
class wires together.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from dllm_bench.interfaces import PositionState
from dllm_bench.models.hf_diffusion import DiffusionStepConfig
from dllm_bench.models.illada import (
    MASK_ID,
    IlladaAdapter,
    _add_gumbel_noise,
    _selected_token_probabilities,
    _transfer_schedule,
)

VOCAB_SIZE = 10


def test_transfer_schedule_spreads_remainder_across_first_steps():
    assert _transfer_schedule(10, 3) == [4, 3, 3]
    assert _transfer_schedule(9, 3) == [3, 3, 3]
    assert _transfer_schedule(4, 4) == [1, 1, 1, 1]


def test_transfer_schedule_zero_steps_is_empty():
    assert _transfer_schedule(5, 0) == []


def test_gumbel_noise_is_a_no_op_at_temperature_zero():
    logits = torch.randn(1, 3, VOCAB_SIZE)
    assert torch.equal(_add_gumbel_noise(logits, 0.0), logits)


def test_gumbel_noise_perturbs_at_nonzero_temperature():
    logits = torch.zeros(1, 3, VOCAB_SIZE)
    perturbed = _add_gumbel_noise(logits, 1.0)
    assert not torch.equal(perturbed, logits)


def test_selected_token_probabilities_match_full_softmax():
    logits = torch.randn(2, 5, VOCAB_SIZE)
    token_ids = torch.randint(0, VOCAB_SIZE, (2, 5))

    expected = torch.softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(_selected_token_probabilities(logits, token_ids), expected)


class _FakeLogitsModel:
    """Always returns the same fixed logits — fine here because the real
    algorithm excludes already-accepted positions via `mask_index`, so a
    constant confidence ranking still produces a deterministic, checkable
    commit order across steps."""

    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, x, attention_mask=None):
        class _Output:
            pass

        out = _Output()
        out.logits = self._logits
        return out


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{i}>" for i in ids)


def _make_adapter_with_fixed_confidence_ranking() -> IlladaAdapter:
    """4 gen positions (block_length=4), confidence strictly ranked
    local-position 2 > 0 > 3 > 1 by construction (higher peak logit ->
    higher softmax max-probability, all else held at 0)."""
    prompt_len = 2
    gen_length = 4
    seq_len = prompt_len + gen_length
    logits = torch.zeros(1, seq_len, VOCAB_SIZE)
    peak_by_local_position = {2: 20.0, 0: 10.0, 3: 5.0, 1: 1.0}
    for local_pos, peak in peak_by_local_position.items():
        logits[0, prompt_len + local_pos, 3] = peak

    step_config = DiffusionStepConfig(gen_length=gen_length, block_length=4, steps_per_block=4)
    adapter = IlladaAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeLogitsModel(logits)
    adapter._tokenizer = _FakeTokenizer()
    adapter._device = "cpu"
    return adapter


def test_illada_selects_positions_in_descending_confidence_order():
    adapter = _make_adapter_with_fixed_confidence_ranking()
    _, trace, final_valid_length = adapter._run_denoising("prompt", adapter._step_config)

    assert final_valid_length == 4
    assert len(trace) == 4  # steps_per_block=4, block_length=4 -> 1 commit/step

    committed_order = [step.committed_positions for step in trace]
    assert committed_order == [[2], [0], [3], [1]]


def test_illada_never_recommits_an_already_accepted_position():
    adapter = _make_adapter_with_fixed_confidence_ranking()
    _, trace, _ = adapter._run_denoising("prompt", adapter._step_config)

    seen = set()
    for step in trace:
        for position in step.committed_positions:
            assert position not in seen, f"position {position} committed twice"
            seen.add(position)
    assert seen == {0, 1, 2, 3}


def test_illada_position_states_track_accept_progress_monotonically():
    adapter = _make_adapter_with_fixed_confidence_ranking()
    _, trace, _ = adapter._run_denoising("prompt", adapter._step_config)

    accepted_counts = [
        sum(1 for s in step.position_states if s == PositionState.ACCEPTED) for step in trace
    ]
    assert accepted_counts == [1, 2, 3, 4]
    # never re-masks: once ACCEPTED, always ACCEPTED in later steps
    for earlier, later in zip(trace, trace[1:]):
        for i, state in enumerate(earlier.position_states):
            if state == PositionState.ACCEPTED:
                assert later.position_states[i] == PositionState.ACCEPTED


def test_illada_entropy_and_confidence_only_reported_for_remaining_positions():
    adapter = _make_adapter_with_fixed_confidence_ranking()
    _, trace, _ = adapter._run_denoising("prompt", adapter._step_config)

    first_step = trace[0]
    assert set(first_step.entropy_by_position) == {0, 1, 3}  # position 2 just got committed
    assert set(first_step.top1_confidence_by_position) == {0, 1, 3}
    for entropy in first_step.entropy_by_position.values():
        assert 0.0 <= entropy <= 1.0  # normalized by log(vocab_size)

    last_step = trace[-1]
    assert last_step.entropy_by_position == {}


def test_illada_two_blocks_run_in_order_and_reset_confidence_scope():
    """block_length=2 with gen_length=4 -> two blocks; a block's own top-k
    selection must never consider positions from a later block."""
    prompt_len = 2
    gen_length = 4
    block_length = 2
    seq_len = prompt_len + gen_length
    logits = torch.zeros(1, seq_len, VOCAB_SIZE)
    # Block 0 = local positions [0,1]; block 1 = local positions [2,3].
    # Give position 3 (in block 1) a huge peak — it must NOT be pickable
    # while block 0 is still active.
    logits[0, prompt_len + 3, 3] = 50.0
    logits[0, prompt_len + 0, 3] = 5.0
    logits[0, prompt_len + 1, 3] = 1.0
    logits[0, prompt_len + 2, 3] = 2.0

    step_config = DiffusionStepConfig(gen_length=gen_length, block_length=block_length, steps_per_block=2)
    adapter = IlladaAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeLogitsModel(logits)
    adapter._tokenizer = _FakeTokenizer()
    adapter._device = "cpu"

    _, trace, _ = adapter._run_denoising("prompt", step_config)

    assert len(trace) == 4  # 2 blocks * 2 steps_per_block
    # first two steps (block 0) may only ever commit positions 0/1
    assert trace[0].committed_positions[0] in (0, 1)
    assert trace[1].committed_positions[0] in (0, 1)
    all_block0_commits = trace[0].committed_positions + trace[1].committed_positions
    assert sorted(all_block0_commits) == [0, 1]
    # block 1 (positions 2/3) only starts after block 0 fully committed
    all_block1_commits = trace[2].committed_positions + trace[3].committed_positions
    assert sorted(all_block1_commits) == [2, 3]


def test_illada_mask_id_matches_reference_override():
    # iLLaDA overrides LLaDA's default mask_id=126336 with 5 (verified
    # against the reference project's own README/config).
    assert MASK_ID == 5
