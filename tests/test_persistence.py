"""Round-trip tests for the model_output/score_output per-sample artifacts —
this is the contract that lets generation run on one machine and
scoring/visualization run on another."""

from __future__ import annotations

import pytest

from dllm_bench.datasets.base import ScoreResult
from dllm_bench.interfaces import (
    GenerationRequest,
    GenerationResult,
    ForwardProfile,
    PositionState,
    RunStatus,
    TraceStep,
)
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.runner.persistence import (
    generation_result_from_dict,
    generation_result_to_dict,
    load_generation_result,
    load_meta,
    load_score_result,
    save_generation_result,
    save_meta,
    save_score_result,
    score_result_from_dict,
    score_result_to_dict,
)


def _sample_generation():
    adapter = MockDiffusionAdapter(steps=4)
    request = GenerationRequest(prompt="what is 2+2", max_new_tokens=10, seed=42, sample_id="s1")
    return adapter.generate(request)


def test_generation_result_round_trips_through_dict():
    original = _sample_generation()
    original.forward_profiles = [
        ForwardProfile(
            forward_index=0,
            phase="denoise",
            wall_clock_seconds=0.01,
            compute_tflops=0.5,
            accepted_tokens=2,
        )
    ]
    restored = generation_result_from_dict(generation_result_to_dict(original))

    assert restored.output_text == original.output_text
    assert restored.status == original.status
    assert restored.num_forward_passes == original.num_forward_passes
    assert restored.final_valid_length == original.final_valid_length
    assert restored.timing.wall_clock_seconds == pytest.approx(original.timing.wall_clock_seconds)
    assert restored.timing.source == original.timing.source
    assert restored.request.prompt == original.request.prompt
    assert restored.request.sample_id == original.request.sample_id
    assert len(restored.trace) == len(original.trace)
    assert restored.forward_profiles == original.forward_profiles
    for restored_step, original_step in zip(restored.trace, original.trace):
        assert restored_step.forward_index == original_step.forward_index
        assert restored_step.token_ids == original_step.token_ids
        assert restored_step.position_states == original_step.position_states
        assert restored_step.committed_positions == original_step.committed_positions
        assert restored_step.decoded_text == original_step.decoded_text
        assert restored_step.entropy_by_position == original_step.entropy_by_position


def test_generation_result_round_trips_through_file(tmp_path):
    original = _sample_generation()
    path = tmp_path / "s1.json"
    save_generation_result(original, path)
    restored = load_generation_result(path)
    assert restored.output_text == original.output_text
    assert len(restored.trace) == len(original.trace)


def test_generation_result_with_no_trace_round_trips():
    adapter = MockDiffusionAdapter(steps=4)
    request = GenerationRequest(prompt="x", max_new_tokens=5, seed=1)
    generation = adapter.generate(request)
    generation.trace = []
    restored = generation_result_from_dict(generation_result_to_dict(generation))
    assert restored.trace == []


def test_legacy_generation_recovers_first_eos_from_trace(tmp_path):
    original = GenerationResult(
        request=GenerationRequest(prompt="p", max_new_tokens=3),
        output_text="answerjunk",
        status=RunStatus.SUCCESS,
        final_valid_length=3,
        trace=[
            TraceStep(
                forward_index=0,
                token_ids=[10, 11, 12],
                position_states=[
                    PositionState.ACCEPTED,
                    PositionState.ACCEPTED,
                    PositionState.ACCEPTED,
                ],
                committed_positions=[0, 1, 2],
                decoded_text="answer<[EOS]>junk",
                entropy_by_position={"0": 0.1, "2": 0.3},
                top1_confidence_by_position={"0": 0.9, "2": 0.7},
                token_texts=["answer", "<[EOS]>", "junk"],
            )
        ],
    )
    path = tmp_path / "legacy-eos.json"
    save_generation_result(original, path)

    restored = load_generation_result(path)

    assert restored.output_text == "answer"
    assert restored.final_valid_length == 1
    assert restored.trace[-1].token_ids == [10]
    assert restored.trace[-1].entropy_by_position == {"0": 0.1}
    assert restored.trace[-1].top1_confidence_by_position == {"0": 0.9}
    assert restored.extra["stop_reason"] == "eos"
    assert restored.extra["eos_boundary_recovered_from_trace"] is True


def test_score_result_round_trips_through_dict():
    original = ScoreResult(primary_score=0.75, aux={"valid_rate": 1.0}, valid=True, complete=False)
    restored = score_result_from_dict(score_result_to_dict(original))
    assert restored.primary_score == pytest.approx(0.75)
    assert restored.aux == {"valid_rate": 1.0}
    assert restored.valid is True
    assert restored.complete is False


def test_score_result_round_trips_through_file(tmp_path):
    original = ScoreResult(primary_score=1.0, aux={}, valid=True, complete=True)
    path = tmp_path / "s1.json"
    save_score_result(original, path)
    restored = load_score_result(path)
    assert restored.primary_score == pytest.approx(1.0)


def test_meta_round_trips_through_file(tmp_path):
    meta = {"model_name": "mock", "config_name": "default", "run_metadata": {"seed": 42}}
    path = tmp_path / "_meta.json"
    save_meta(meta, path)
    assert load_meta(path) == meta
