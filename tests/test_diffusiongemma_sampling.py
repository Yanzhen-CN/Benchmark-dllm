"""Algorithm-level tests for the real DiffusionGemma adapter.
`_assign_canvas_indices`/`_build_trace_from_captured_steps` trace-conversion
logic, and the `_prepare_sampler` patch-and-restore mechanism, all against a
fake model — no real weights/GPU needed.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from dllm_bench.interfaces import GenerationRequest, PositionState
from dllm_bench.models.diffusiongemma import (
    DiffusionGemmaAdapter,
    _assign_canvas_indices,
    _build_trace_from_captured_steps,
)

VOCAB_SIZE = 8


def test_assign_canvas_indices_single_canvas_counts_down():
    assert _assign_canvas_indices([{"cur_step": s} for s in [4, 3, 2, 1]]) == [0, 0, 0, 0]


def test_assign_canvas_indices_detects_new_canvas_on_increase():
    steps = [{"cur_step": s} for s in [2, 1, 2, 1]]
    assert _assign_canvas_indices(steps) == [0, 0, 1, 1]


def test_assign_canvas_indices_three_canvases():
    steps = [{"cur_step": s} for s in [3, 2, 1, 3, 2, 1, 3, 2, 1]]
    assert _assign_canvas_indices(steps) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


class _RecordingTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return f"[{ids[0]}]"


def _uniform_entropy_step(cur_step, accepted_canvas, accepted_mask):
    """Uniform logits -> maximal entropy (normalized to 1.0) at every
    position, so expected entropy values are trivial to assert."""
    n = len(accepted_canvas)
    return {
        "cur_step": cur_step,
        "accepted_canvas": torch.tensor([accepted_canvas]),
        "accepted_token_mask": torch.tensor([accepted_mask]),
        "entropy": torch.full((1, n), math.log(VOCAB_SIZE)),
        "vocab_size": VOCAB_SIZE,
    }


def test_build_trace_single_canvas_marks_non_accepted_as_visible():
    captured = [
        _uniform_entropy_step(2, [10, 9, 12, 9], [True, False, True, False]),
        _uniform_entropy_step(1, [20, 21, 22, 23], [True, True, True, True]),
    ]
    trace = _build_trace_from_captured_steps(captured, canvas_length=4, tokenizer=_RecordingTokenizer())

    assert len(trace) == 2
    step0 = trace[0]
    assert step0.token_ids == [10, 9, 12, 9]
    assert step0.position_states == [
        PositionState.ACCEPTED, PositionState.VISIBLE, PositionState.ACCEPTED, PositionState.VISIBLE,
    ]
    assert step0.committed_positions == [0, 2]
    assert set(step0.entropy_by_position) == {1, 3}
    assert step0.entropy_by_position[1] == pytest.approx(1.0)

    step1 = trace[1]
    assert step1.token_ids == [20, 21, 22, 23]
    assert step1.position_states == [PositionState.ACCEPTED] * 4
    assert step1.committed_positions == [0, 1, 2, 3]
    assert step1.entropy_by_position is None  # every position accepted -> nothing "remaining"


def test_build_trace_allows_revision_of_a_previously_accepted_position():
    """Position 0 is accepted with token 10 at step0, then genuinely
    revised to a different token (20) at step1 — this is real DiffusionGemma behavior
    (accepted_token_mask recomputed from scratch every step), not a bug."""
    captured = [
        _uniform_entropy_step(2, [10, 9], [True, False]),
        _uniform_entropy_step(1, [20, 21], [True, True]),
    ]
    trace = _build_trace_from_captured_steps(captured, canvas_length=2, tokenizer=_RecordingTokenizer())
    assert trace[0].token_ids[0] == 10
    assert trace[1].token_ids[0] == 20  # revised
    assert 0 in trace[1].committed_positions  # re-committed, not skipped


def test_build_trace_does_not_repeat_unchanged_cumulative_accepts():
    captured = [
        _uniform_entropy_step(2, [10, 9], [True, False]),
        _uniform_entropy_step(1, [10, 21], [True, True]),
    ]
    trace = _build_trace_from_captured_steps(
        captured, canvas_length=2, tokenizer=_RecordingTokenizer()
    )
    assert trace[0].committed_positions == [0]
    assert trace[1].committed_positions == [1]


def test_build_trace_multi_canvas_offsets_positions_globally():
    # canvas 0 counts fully down (2, 1) before canvas 1's first step (2)
    # registers as a new canvas — a same-or-lower cur_step never does (the
    # heuristic, ported from the reference trace tooling, only fires on a
    # strict increase versus the immediately preceding row).
    captured = [
        _uniform_entropy_step(2, [1, 9], [True, False]),
        _uniform_entropy_step(1, [1, 2], [True, True]),  # canvas 0 finishes here
        _uniform_entropy_step(2, [3, 9], [False, True]),  # canvas 1 starts (2 > previous row's 1)
        _uniform_entropy_step(1, [3, 5], [True, True]),
    ]
    trace = _build_trace_from_captured_steps(captured, canvas_length=2, tokenizer=_RecordingTokenizer())

    assert len(trace) == 4
    # canvas 1's steps must carry canvas 0's finalized tokens ([1, 2]) as a prefix
    assert trace[2].token_ids == [1, 2, 3, 9]
    assert trace[3].token_ids == [1, 2, 3, 5]
    # committed positions in canvas 1 are offset by canvas_length=2
    assert trace[2].committed_positions == [3]  # only local position 1 accepted this step
    assert trace[3].committed_positions == [2, 3]


def test_build_trace_empty_input_is_empty_output():
    assert _build_trace_from_captured_steps([], canvas_length=4, tokenizer=_RecordingTokenizer()) == []


# ---------------------------------------------------------------------------
# _generate_core integration: fake model + processor, no real weights.
# ---------------------------------------------------------------------------

class _FakeEncoded(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _RecordingTokenizer()
        self.messages = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return _FakeEncoded(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))


class _FakeSampler:
    def __init__(self, masks):
        self._masks = list(masks)
        self._call_index = 0
        self.accepted_token_mask = None

    def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
        mask = self._masks[self._call_index]
        self._call_index += 1
        self.accepted_token_mask = mask
        return torch.where(mask, denoiser_canvas, current_canvas)


class _FakeDGModel:
    """Faithfully mirrors the real model's calling convention just enough to
    test our patch: `generate()` looks up `self._prepare_sampler` through
    normal attribute access (so an instance-level override on `self` — what
    our adapter installs — takes effect exactly like it would for real)."""

    def __init__(self, raise_after_first_step: bool = False):
        from types import SimpleNamespace

        self.config = SimpleNamespace(canvas_length=4)
        self.generation_config = SimpleNamespace(pad_token_id=0)
        self._raise_after_first_step = raise_after_first_step
        self.last_generate_kwargs = None

    def _prepare_sampler(self, generation_config):
        return _FakeSampler(masks=[
            torch.tensor([[True, False, True, False]]),
            torch.tensor([[True, True, True, True]]),
        ])

    def generate(self, **kwargs):
        from types import SimpleNamespace

        self.last_generate_kwargs = kwargs
        sampler = self._prepare_sampler(None)
        current = torch.tensor([[9, 9, 9, 9]])
        script = [
            (2, torch.tensor([[10, 11, 12, 13]]), torch.zeros(1, 4, VOCAB_SIZE)),
            (1, torch.tensor([[20, 21, 22, 23]]), torch.zeros(1, 4, VOCAB_SIZE)),
        ]
        for index, (cur_step, denoiser_canvas, logits) in enumerate(script):
            current = sampler.accept_canvas(current, denoiser_canvas, logits, cur_step)
            if self._raise_after_first_step and index == 0:
                raise RuntimeError("simulated failure mid-generation")
        final_sequence = torch.cat([kwargs["input_ids"], current], dim=1)
        return SimpleNamespace(
            sequences=final_sequence,
            tokens_per_forward=torch.tensor([2.0]),
        )


def _make_diffusiongemma_adapter(raise_after_first_step: bool = False) -> DiffusionGemmaAdapter:
    adapter = DiffusionGemmaAdapter("unused-checkpoint")
    adapter._model = _FakeDGModel(raise_after_first_step=raise_after_first_step)
    adapter._processor = _FakeProcessor()
    adapter._device = "cpu"
    return adapter


def test_diffusiongemma_generate_core_builds_correct_trace_and_output():
    adapter = _make_diffusiongemma_adapter()
    request = GenerationRequest(
        prompt="hi",
        max_new_tokens=4,
    )

    result = adapter._generate_core(request)

    assert result.output_text == "[20]"  # _RecordingTokenizer.decode returns f"[{ids[0]}]"
    assert result.final_valid_length == 4
    assert result.num_forward_passes == 2
    assert result.extra["input_tokens"] == 2
    assert result.extra["official_tokens_per_forward"] == 2.0
    assert "disable_compile" not in adapter._model.last_generate_kwargs
    assert len(result.trace) == 2
    assert result.trace[0].committed_positions == [0, 2]
    assert result.trace[1].committed_positions == [0, 1, 2, 3]
    assert adapter._processor.messages == [{"role": "user", "content": "hi"}]
    assert adapter._processor.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }


def test_diffusiongemma_prepare_sampler_is_restored_after_successful_generate():
    adapter = _make_diffusiongemma_adapter()
    original = adapter._model._prepare_sampler
    adapter._generate_core(GenerationRequest(prompt="hi", max_new_tokens=4))
    assert adapter._model._prepare_sampler == original


def test_diffusiongemma_prepare_sampler_is_restored_even_if_generate_raises():
    adapter = _make_diffusiongemma_adapter(raise_after_first_step=True)
    original = adapter._model._prepare_sampler

    with pytest.raises(RuntimeError, match="simulated failure"):
        adapter._generate_core(GenerationRequest(prompt="hi", max_new_tokens=4))

    assert adapter._model._prepare_sampler == original


def test_diffusiongemma_generate_core_failure_surfaces_as_failed_status_via_base_adapter():
    """Through the public `.generate()` entry point (not `_generate_core`
    directly), a mid-generation exception must degrade to RunStatus.FAILED,
    not crash the whole benchmark run — and the sampler patch must still be
    restored (checked above) even on that path."""
    from dllm_bench.interfaces import RunStatus

    adapter = _make_diffusiongemma_adapter(raise_after_first_step=True)
    result = adapter.generate(GenerationRequest(prompt="hi", max_new_tokens=4))
    assert result.status == RunStatus.FAILED
    assert "simulated failure" in result.error_message
