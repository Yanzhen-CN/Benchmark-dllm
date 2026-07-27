"""Algorithm-level tests for the real DreamReasoner sampler port
(models/dreamreasoner.py), using a fake model with controlled logits — no
real weights/GPU needed. Verifies the ported algorithm itself (per-block
denoising loop, KV-cache store_kv flags, `_select_transfer_index`'s several
real remasking strategies, force_accept on a block's last step,
mask_token_id resolution), not just that the class wires together — same
spirit as `test_illada_sampling.py`/`test_diffusiongemma_sampling.py`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dllm_bench.interfaces import PositionState
from dllm_bench.models.dreamreasoner import (
    DreamReasonerAdapter,
    _get_num_transfer_tokens,
    _select_transfer_index,
)
from dllm_bench.models.hf_diffusion import DiffusionStepConfig

VOCAB_SIZE = 10


def test_get_num_transfer_tokens_spreads_remainder_across_first_steps():
    assert _get_num_transfer_tokens(10, 3).tolist() == [4, 3, 3]
    assert _get_num_transfer_tokens(9, 3).tolist() == [3, 3, 3]
    assert _get_num_transfer_tokens(4, 4).tolist() == [1, 1, 1, 1]


class _FakeLogitsModel:
    """Always returns the same fixed logits for whatever block-local slice
    it's called with — fine because the real algorithm excludes
    already-committed positions via `mask_index`, so a constant confidence
    ranking still produces a deterministic, checkable commit order across
    steps (same trick as `test_illada_sampling.py`). Records every call's
    `store_kv` flag so the KV-cache call pattern (prefill store, draft
    no-store, finalize store) can be asserted on directly.
    """

    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        self.calls: list[dict] = []

    def __call__(self, x, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, store_kv=None):
        self.calls.append({"store_kv": store_kv, "shape": tuple(x.shape)})

        class _Output:
            pass

        out = _Output()
        out.logits = self._logits
        return out


class _FakeTokenizer:
    def __init__(self, prompt_len: int):
        self._prompt_len = prompt_len

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.arange(1, self._prompt_len + 1).unsqueeze(0)}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{i}>" for i in ids)


def _make_adapter_with_fixed_confidence_ranking(
    *, remasking_strategy: str = "low_confidence_static", extra: dict | None = None
) -> tuple[DreamReasonerAdapter, _FakeLogitsModel]:
    """block_length=4, one block's worth of gen (prompt_len is block-aligned
    so prefill covers the whole prompt in one go and the single processed
    block maps 1:1 onto gen-local positions 0..3). Confidence strictly
    ranked local-position 2 > 0 > 3 > 1 by construction."""
    prompt_len = 4
    gen_length = 4
    logits = torch.zeros(1, 4, VOCAB_SIZE)
    peak_by_local_position = {2: 20.0, 0: 10.0, 3: 5.0, 1: 1.0}
    for local_pos, peak in peak_by_local_position.items():
        logits[0, local_pos, 3] = peak

    step_extra = {"remasking_strategy": remasking_strategy, **(extra or {}), "mask_token_id": 99}
    step_config = DiffusionStepConfig(
        gen_length=gen_length, block_length=4, steps_per_block=4, extra=step_extra
    )
    adapter = DreamReasonerAdapter("unused-checkpoint", step_config, config_name="test")
    fake_model = _FakeLogitsModel(logits)
    adapter._model = fake_model
    adapter._tokenizer = _FakeTokenizer(prompt_len)
    adapter._device = "cpu"
    return adapter, fake_model


def test_dreamreasoner_selects_positions_in_descending_confidence_order():
    adapter, _ = _make_adapter_with_fixed_confidence_ranking()
    _, trace, final_valid_length = adapter._run_denoising("prompt", adapter._step_config)

    assert final_valid_length == 4
    assert len(trace) == 4  # steps_per_block=4, block_length=4 -> 1 commit/step

    committed_order = [step.committed_positions for step in trace]
    assert committed_order == [[2], [0], [3], [1]]


def test_dreamreasoner_never_recommits_an_already_committed_position():
    adapter, _ = _make_adapter_with_fixed_confidence_ranking()
    _, trace, _ = adapter._run_denoising("prompt", adapter._step_config)

    seen = set()
    for step in trace:
        for position in step.committed_positions:
            assert position not in seen, f"position {position} committed twice"
            seen.add(position)
    assert seen == {0, 1, 2, 3}


def test_dreamreasoner_position_states_track_commit_progress_monotonically():
    adapter, _ = _make_adapter_with_fixed_confidence_ranking()
    _, trace, _ = adapter._run_denoising("prompt", adapter._step_config)

    accepted_counts = [
        sum(1 for s in step.position_states if s == PositionState.ACCEPTED) for step in trace
    ]
    assert accepted_counts == [1, 2, 3, 4]
    for earlier, later in zip(trace, trace[1:]):
        for i, state in enumerate(earlier.position_states):
            if state == PositionState.ACCEPTED:
                assert later.position_states[i] == PositionState.ACCEPTED


def test_dreamreasoner_entropy_and_confidence_only_reported_for_active_block():
    adapter, _ = _make_adapter_with_fixed_confidence_ranking()
    _, trace, _ = adapter._run_denoising("prompt", adapter._step_config)

    first_step = trace[0]
    assert set(first_step.entropy_by_position) == {0, 1, 3}  # position 2 just got committed
    assert set(first_step.top1_confidence_by_position) == {0, 1, 3}
    for entropy in first_step.entropy_by_position.values():
        assert 0.0 <= entropy <= 1.0  # normalized by log(vocab_size)

    last_step = trace[-1]
    assert last_step.entropy_by_position == {}


def test_dreamreasoner_kv_cache_call_pattern_matches_real_source():
    """Confirmed from `Dream-org/DreamReasoner-8B/generation_utils.py`: a
    prefill forward with `store_kv=True`, then `denoising_steps` draft
    forwards with `store_kv=False` (drafts aren't final yet), then one more
    `store_kv=True` forward once the block is fully committed (to push its
    now-final tokens into the cache) — that last call produces no
    TraceStep since no positions change."""
    adapter, fake_model = _make_adapter_with_fixed_confidence_ranking()
    adapter._run_denoising("prompt", adapter._step_config)

    store_kv_sequence = [c["store_kv"] for c in fake_model.calls]
    assert store_kv_sequence == [True, False, False, False, False, True]


def test_dreamreasoner_force_accept_commits_everything_remaining_on_last_step():
    """Even without a confidence-based strategy driving it, the last step in
    a block always commits every remaining masked position — ported
    directly from the real `force_accept = step == denoising_steps - 1`."""
    prompt_len = 4
    gen_length = 4
    logits = torch.zeros(1, 4, VOCAB_SIZE)  # perfectly flat: no confidence signal at all
    step_config = DiffusionStepConfig(
        gen_length=gen_length,
        block_length=4,
        steps_per_block=2,  # force_accept fires at step index 1 (of range(0,2))
        extra={"remasking_strategy": "low_confidence_static", "mask_token_id": 99},
    )
    adapter = DreamReasonerAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeLogitsModel(logits)
    adapter._tokenizer = _FakeTokenizer(prompt_len)
    adapter._device = "cpu"

    _, trace, final_valid_length = adapter._run_denoising("prompt", step_config)

    assert final_valid_length == 4
    assert len(trace) == 2
    # step 0 commits num_transfer_tokens[0] = 2 positions (block_length=4 / steps=2)
    assert len(trace[0].committed_positions) == 2
    # step 1 (force_accept) commits everything still remaining
    assert len(trace[1].committed_positions) == 2
    assert sorted(trace[0].committed_positions + trace[1].committed_positions) == [0, 1, 2, 3]


def test_dreamreasoner_two_blocks_run_strictly_in_order():
    """block_length=4 with gen_length=8 -> two blocks processed by the
    denoising loop; block 1's positions structurally cannot be touched
    before block 0's loop finishes (the model is only ever called with that
    block's own slice — unlike iLLaDA, which recomputes the whole sequence
    every step, there is no logits value that could leak a future block's
    position into this step's `transfer_index`)."""
    prompt_len = 4
    gen_length = 8
    logits = torch.zeros(1, 4, VOCAB_SIZE)
    logits[0, 0, 3] = 10.0  # arbitrary distinct confidence ranking within a block
    logits[0, 1, 3] = 1.0
    logits[0, 2, 3] = 5.0
    logits[0, 3, 3] = 2.0

    step_config = DiffusionStepConfig(
        gen_length=gen_length,
        block_length=4,
        steps_per_block=4,
        extra={"remasking_strategy": "low_confidence_static", "mask_token_id": 99},
    )
    adapter = DreamReasonerAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeLogitsModel(logits)
    adapter._tokenizer = _FakeTokenizer(prompt_len)
    adapter._device = "cpu"

    _, trace, final_valid_length = adapter._run_denoising("prompt", step_config)

    assert final_valid_length == 8
    assert len(trace) == 8  # 2 blocks * 4 steps_per_block
    all_block0_commits = sorted(sum((s.committed_positions for s in trace[:4]), []))
    all_block1_commits = sorted(sum((s.committed_positions for s in trace[4:]), []))
    assert all_block0_commits == [0, 1, 2, 3]
    assert all_block1_commits == [4, 5, 6, 7]


def test_select_transfer_index_low_confidence_dynamic_takes_all_above_threshold():
    """The library's own default strategy (kept as-is, no override — see
    models/dreamreasoner.py's module docstring): unlike a strict top-k, if
    more than k positions already exceed `confidence_threshold`, ALL of them
    get committed this step, not just the top k."""
    mask_index = torch.tensor([[True, True, True, False]])
    x0 = torch.tensor([[1, 2, 3, 4]])
    # Two positions far above the 0.9 threshold, one just below it.
    x0_p = torch.tensor([[0.99, 0.95, 0.5, 0.0]])
    num_transfer_tokens = torch.tensor([1])

    transfer_index = _select_transfer_index(
        "low_confidence_dynamic", mask_index, x0, x0_p, num_transfer_tokens, step=0,
        confidence_threshold=0.9, eb_threshold=0.35,
    )
    assert transfer_index.tolist() == [[True, True, False, False]]


def test_select_transfer_index_low_confidence_static_is_strict_top_k():
    mask_index = torch.tensor([[True, True, True, False]])
    x0 = torch.tensor([[1, 2, 3, 4]])
    x0_p = torch.tensor([[0.99, 0.95, 0.5, 0.0]])
    num_transfer_tokens = torch.tensor([1])

    transfer_index = _select_transfer_index(
        "low_confidence_static", mask_index, x0, x0_p, num_transfer_tokens, step=0,
        confidence_threshold=0.9, eb_threshold=0.35,
    )
    # only the single highest-confidence masked position, regardless of how
    # many others also clear the (unused-by-this-strategy) threshold
    assert transfer_index.tolist() == [[True, False, False, False]]


def test_select_transfer_index_force_accept_ignores_strategy_entirely():
    mask_index = torch.tensor([[True, False, True, False]])
    x0 = torch.tensor([[1, 2, 3, 4]])
    x0_p = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    num_transfer_tokens = torch.tensor([1])

    transfer_index = _select_transfer_index(
        "low_confidence_static", mask_index, x0, x0_p, num_transfer_tokens, step=0,
        confidence_threshold=0.9, eb_threshold=0.35, force_accept=True,
    )
    assert torch.equal(transfer_index, mask_index)


def test_dreamreasoner_mask_token_id_prefers_explicit_extra_override():
    step_config = DiffusionStepConfig(gen_length=4, extra={"mask_token_id": 42})
    adapter = DreamReasonerAdapter("unused-checkpoint", step_config, config_name="test")
    from types import SimpleNamespace

    adapter._model = SimpleNamespace(config=SimpleNamespace(mask_token_id=7))
    adapter._tokenizer = SimpleNamespace(mask_token_id=8)
    assert adapter._resolve_mask_token_id(step_config) == 42


def test_dreamreasoner_mask_token_id_falls_back_to_the_checkpoints_own_config():
    """Real checkpoint ships `config.json`'s `mask_token_id: 151669` directly
    — this is the primary source, checked ahead of the tokenizer."""
    step_config = DiffusionStepConfig(gen_length=4)
    adapter = DreamReasonerAdapter("unused-checkpoint", step_config, config_name="test")
    from types import SimpleNamespace

    adapter._model = SimpleNamespace(config=SimpleNamespace(mask_token_id=151669))
    adapter._tokenizer = SimpleNamespace(mask_token_id=8)
    assert adapter._resolve_mask_token_id(step_config) == 151669


def test_dreamreasoner_raises_a_clear_error_when_mask_token_id_is_unknowable():
    step_config = DiffusionStepConfig(gen_length=4)
    adapter = DreamReasonerAdapter("unused-checkpoint", step_config, config_name="test")
    from types import SimpleNamespace

    adapter._model = SimpleNamespace(config=SimpleNamespace())
    adapter._tokenizer = SimpleNamespace()
    with pytest.raises(ValueError, match="mask_token_id"):
        adapter._resolve_mask_token_id(step_config)


def test_dreamreasoner_default_denoising_steps_falls_back_to_block_length():
    """Real library default (confirmed from source): `denoising_steps =
    block_length` whenever unset and `remasking_strategy` isn't
    `low_confidence_static` — this benchmark's Best config just keeps that
    default explicit (see configs/models/dreamreasoner.yaml)."""
    prompt_len = 4
    gen_length = 4
    logits = torch.zeros(1, 4, VOCAB_SIZE)
    step_config = DiffusionStepConfig(
        gen_length=gen_length,
        block_length=4,
        steps_per_block=None,  # deliberately unset
        extra={"mask_token_id": 99},
    )
    adapter = DreamReasonerAdapter("unused-checkpoint", step_config, config_name="test")
    adapter._model = _FakeLogitsModel(logits)
    adapter._tokenizer = _FakeTokenizer(prompt_len)
    adapter._device = "cpu"

    _, trace, _ = adapter._run_denoising("prompt", step_config)
    assert len(trace) == 4  # denoising_steps defaulted to block_length=4
