from __future__ import annotations

from pathlib import Path

import pytest

from dllm_bench.datasets.base import Sample
from dllm_bench.datasets.sudoku9 import SudokuReference
from dllm_bench.interfaces import (
    ForwardProfile,
    GenerationRequest,
    GenerationResult,
    PositionState,
    RunStatus,
    TimingResult,
    TraceStep,
)
from dllm_bench.visual.public.dataset_trace_report import (
    build_dataset_trace_summary,
    render_dataset_trace_report,
)
from dllm_bench.visual.public.plots import plot_sudoku_revision_diagnostics


def _result(trace: list[TraceStep], length: int) -> GenerationResult:
    previous_accepted: set[int] = set()
    profiles = []
    for index, step in enumerate(trace):
        current_accepted = {
            position
            for position, state in enumerate(step.position_states)
            if state == PositionState.ACCEPTED
        }
        profiles.append(
            ForwardProfile(
                index,
                "denoise_step",
                accepted_tokens=len(current_accepted - previous_accepted),
            )
        )
        previous_accepted = current_accepted
    return GenerationResult(
        request=GenerationRequest(prompt="p", max_new_tokens=length),
        output_text="1" * length,
        status=RunStatus.SUCCESS,
        trace=trace,
        forward_profiles=profiles,
        num_forward_passes=len(trace),
        final_valid_length=length,
        timing=TimingResult(2.0),
    )


def _step(
    index: int,
    token_ids: list[int],
    states: list[PositionState],
    *,
    entropy: dict[int, float] | None = None,
    top1: dict[int, float] | None = None,
    token_texts: list[str] | None = None,
) -> TraceStep:
    return TraceStep(
        forward_index=index,
        token_ids=token_ids,
        position_states=states,
        committed_positions=[],
        decoded_text="",
        entropy_by_position=entropy,
        top1_confidence_by_position=top1,
        token_texts=token_texts,
    )


def test_ar_trace_without_logits_reports_certainty_as_unavailable():
    trace = [
        _step(0, [1, -1], [PositionState.ACCEPTED, PositionState.MASKED]),
        _step(1, [1, 2], [PositionState.ACCEPTED, PositionState.ACCEPTED]),
    ]
    summary, curves = build_dataset_trace_summary(
        "gsm8k", [(Sample("s", "p", "r"), _result(trace, 2))]
    )
    assert "certainty" not in curves
    assert "top1" not in curves
    assert summary["certainty_observation"]["scope"] == "unavailable"
    assert summary["certainty_observation"]["top1_scope"] == "unavailable"
    assert summary["certainty_observation"]["curve_sample_rate"] == 0.0
    assert summary["parallelism_signature"]["peak_to_mean_tpf"]["mean"] >= 1.0
    assert 0.0 <= summary["final_stable_progress"]["p90"]["mean"] <= 1.0
    assert summary["visible_draft_correction"]["observation_status"] == (
        "commitment_only_trace"
    )
    assert summary["update_geometry"]["mean_finalization_run_length"]["mean"] > 0
    assert "finalization_cdf" not in curves


def test_certainty_curve_uses_accepted_ratio_and_records_observation_scope():
    trace = [
        _step(
            0,
            [-1, -1],
            [PositionState.MASKED, PositionState.MASKED],
            entropy={0: 0.8, 1: 0.6},
            top1={0: 0.2, 1: 0.3},
        ),
        _step(
            1,
            [1, -1],
            [PositionState.ACCEPTED, PositionState.MASKED],
            entropy={1: 0.2},
            top1={1: 0.8},
        ),
        _step(2, [1, 2], [PositionState.ACCEPTED, PositionState.ACCEPTED]),
    ]
    summary, curves = build_dataset_trace_summary(
        "gsm8k", [(Sample("s", "p", "r"), _result(trace, 2))]
    )
    assert curves["certainty"]
    assert curves["top1"]
    assert summary["certainty_observation"]["scope"] == "full_remaining"
    assert summary["certainty_observation"]["top1_scope"] == "full_remaining"
    assert summary["certainty_observation"]["curve_sample_rate"] == 1.0
    assert summary["confidence_dynamics"]["backslide_step_rate"] is not None


def test_visible_draft_correction_separates_helpful_changes_from_n_a():
    trace = [
        _step(
            0,
            [9, 2],
            [PositionState.VISIBLE, PositionState.VISIBLE],
        ),
        _step(
            1,
            [1, 2],
            [PositionState.VISIBLE, PositionState.VISIBLE],
        ),
        _step(2, [1, 2], [PositionState.ACCEPTED, PositionState.ACCEPTED]),
    ]
    summary, _ = build_dataset_trace_summary(
        "gsm8k", [(Sample("s", "p", "r"), _result(trace, 2))]
    )
    correction = summary["visible_draft_correction"]
    assert correction["observation_status"] == "observable"
    assert correction["first_visible_final_match_rate"]["mean"] == pytest.approx(0.5)
    assert correction["helpful_revision_share"]["mean"] == pytest.approx(1.0)
    assert correction["harmful_revision_share"]["mean"] == pytest.approx(0.0)


def test_sudoku_revision_is_easy_hard_and_mapping_coverage_gated(tmp_path: Path):
    puzzle = [[0] * 9 for _ in range(9)]
    solution = [[1] * 9 for _ in range(9)]
    sample = Sample(
        "sudoku-easy",
        "p",
        SudokuReference(puzzle=puzzle, solution=solution, difficulty="easy"),
    )
    first = ["1"] * 81
    first[0] = "2"
    trace = [
        _step(
            0,
            [int(value) for value in first],
            [PositionState.VISIBLE] * 81,
            token_texts=first,
        ),
        _step(
            1,
            [1] * 81,
            [PositionState.VISIBLE] * 81,
            token_texts=["1"] * 81,
        ),
        _step(
            2,
            [1] * 81,
            [PositionState.ACCEPTED] * 81,
            token_texts=["1"] * 81,
        ),
    ]
    records = [(sample, _result(trace, 81))]
    summary, _ = build_dataset_trace_summary("sudoku9", records)
    easy = summary["sudoku_revision"]["by_difficulty"]["easy"]
    assert easy["mapping_eligible_ratio"] == pytest.approx(1.0)
    assert easy["interpretation_status"] == "interpretable"
    assert easy["revision_count_by_stage"]["middle"]["mean"] == pytest.approx(1.0)
    assert easy["correction_success_rate"]["mean"] == pytest.approx(1.0)
    assert easy["pooled_correction_success_rate"] == pytest.approx(1.0)

    written = render_dataset_trace_report("sudoku9", records, tmp_path)
    assert Path(written["summary"]).exists()
    assert Path(written["acceptance_throughput"]).exists()
    assert set(written) == {
        "summary",
        "acceptance_throughput",
        "auxiliary_performance",
    }

    comparison_path = tmp_path / "sudoku_comparison.png"
    plot_sudoku_revision_diagnostics(
        [{"Model": "mock", "Config": "default", "Trace Summary": summary}],
        str(comparison_path),
    )
    assert comparison_path.exists()
