from pathlib import Path

import pytest

from dllm_bench.interfaces import GenerationRequest
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.report.plots import (
    plot_best_vs_fast,
    plot_quality_vs_resource,
    plot_scenario_ranking,
    plot_score_per_unit,
)
from dllm_bench.report.tables import (
    compute_converted_row,
    raw_results_row,
    render_converted_results_table,
    render_raw_results_table,
)
from dllm_bench.report.trace_report import render_sample_report


def _summary(model, config, q, tps=1.0, eps=None):
    return {
        "dataset_name": "gsm8k",
        "model_name": model,
        "config_name": config,
        "q": q,
        "tps": tps,
        "sps": 0.5,
        "eps": eps,
        "cps": None,
        "time_per_sample": 1.0,
        "energy_per_sample": None,
        "compute_per_sample": None,
        "peak_vram_gb": None,
        "score_per_energy": None,
        "score_per_compute": None,
        "status_counts": {"success": 5},
        "timing_source": "measured",
    }


def test_raw_results_row_shape():
    row = raw_results_row(_summary("mock", "default", 0.8))
    assert row["Dataset"] == "gsm8k"
    assert row["Model"] == "mock"
    assert row["q"] == 0.8
    assert row["SPS"] == pytest.approx(0.5)
    assert row["Status"] == "success"


def test_raw_results_row_flags_mixed_status():
    summary = _summary("mock", "default", 0.8)
    summary["status_counts"] = {"success": 3, "failed": 2}
    row = raw_results_row(summary)
    assert row["Status"].endswith("*")


def test_render_raw_results_table_contains_header_and_values():
    rows = [raw_results_row(_summary("mock", "default", 0.8))]
    table = render_raw_results_table(rows)
    assert "Dataset" in table
    assert "mock" in table
    assert "0.8" in table


def test_render_raw_results_table_handles_no_rows():
    assert render_raw_results_table([]) == "(no rows)"


def test_compute_converted_row_faster_model_gets_higher_q_speed():
    baseline = _summary("qwen3_4b", "ar-baseline", 0.5, tps=10.0)
    fast_model = _summary("illada", "fast", 0.5, tps=50.0)
    row = compute_converted_row(fast_model, baseline)
    assert row["r_speed"] == pytest.approx(5.0)
    assert row["Q_speed"] > 0.5


def test_compute_converted_row_missing_energy_leaves_energy_fields_none():
    baseline = _summary("qwen3_4b", "ar-baseline", 0.5, tps=10.0)
    model = _summary("illada", "best", 0.6, tps=8.0)
    row = compute_converted_row(model, baseline)
    assert row["r_energy"] is None
    assert row["Q_energy"] is None
    assert row["Energy-priority"] is None


def test_render_converted_results_table_smoke():
    baseline = _summary("qwen3_4b", "ar-baseline", 0.5, tps=10.0)
    model = _summary("illada", "best", 0.6, tps=8.0)
    row = compute_converted_row(model, baseline)
    table = render_converted_results_table([row])
    assert "r_speed" in table


# ---------------------------------------------------------------------------
# Plots: smoke-test that files actually get written (content correctness is
# out of scope for a unit test — matplotlib rendering itself is trusted).
# ---------------------------------------------------------------------------

def test_plot_quality_vs_resource_writes_file(tmp_path):
    rows = [raw_results_row(_summary("mock", "default", 0.8, tps=1.5))]
    out = tmp_path / "quality_time.png"
    plot_quality_vs_resource(rows, "TPS", str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_score_per_unit_skips_when_no_data(tmp_path):
    rows = [raw_results_row(_summary("mock", "default", 0.8))]
    out = tmp_path / "score_per_j.png"
    plot_score_per_unit(rows, "Score/J", str(out))
    assert not out.exists()  # no Score/J values -> nothing to plot


def test_plot_best_vs_fast_writes_file_when_both_configs_present(tmp_path):
    rows = [
        raw_results_row(_summary("illada", "best", 0.7, tps=5.0)),
        raw_results_row(_summary("illada", "fast", 0.6, tps=10.0)),
    ]
    out = tmp_path / "best_vs_fast.png"
    plot_best_vs_fast(rows, "q", str(out))
    assert out.exists()


def test_plot_scenario_ranking_writes_file(tmp_path):
    baseline = _summary("qwen3_4b", "ar-baseline", 0.5, tps=10.0)
    model = _summary("illada", "best", 0.6, tps=8.0)
    row = compute_converted_row(model, baseline)
    out = tmp_path / "ranking.png"
    plot_scenario_ranking([row], "Speed-priority", str(out))
    assert not out.exists()  # only one row and Speed-priority is None (no energy) -> nothing to plot


# ---------------------------------------------------------------------------
# Trace report (4.6)
# ---------------------------------------------------------------------------

def test_render_sample_report_writes_expected_files(tmp_path):
    adapter = MockDiffusionAdapter(steps=4)
    request = GenerationRequest(prompt="what is 2+2", max_new_tokens=10, seed=42)
    result = adapter.generate(request)

    written = render_sample_report(
        sample_id="demo-1",
        trace=result.trace,
        final_valid_length=result.final_valid_length,
        out_dir=str(tmp_path),
        final_output_text=result.output_text,
        final_score=1.0,
    )
    # design doc 4.1's exact 3 items (heatmap, certainty, result) plus the
    # DGtest-style single-sample extras (GIF/final-PNG/position-vs-commit/
    # speed) — NOT "parallelism"/"strategy": those are 4.2 dataset-level
    # aggregates now, see report/dataset_trace_report.py, never shown
    # redundantly for one sample here.
    for key in (
        "heatmap",
        "token_grid_gif",
        "token_grid_final",
        "position_vs_commit",
        "speed",
        "certainty",
        "result",
    ):
        assert key in written
        assert Path(written[key]).exists()
    assert "parallelism" not in written
    assert "strategy" not in written


def test_render_sample_report_skips_sudoku_gif_for_non_sudoku_dataset(tmp_path):
    adapter = MockDiffusionAdapter(steps=4)
    request = GenerationRequest(prompt="what is 2+2", max_new_tokens=10, seed=42)
    result = adapter.generate(request)

    written = render_sample_report(
        sample_id="demo-1",
        trace=result.trace,
        final_valid_length=result.final_valid_length,
        out_dir=str(tmp_path),
        dataset_name="gsm8k",
    )
    assert "sudoku_gif" not in written


def test_render_sample_report_renders_sudoku_gif_for_81_position_trace(tmp_path):
    from dllm_bench.datasets.base import Sample
    from dllm_bench.datasets.sudoku import SudokuReference
    from dllm_bench.interfaces import PositionState, TraceStep

    solution = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]
    puzzle = [row[:] for row in solution]
    puzzle[0][0] = 0

    digits = [str(solution[r][c]) for r in range(9) for c in range(9)]
    trace = [
        TraceStep(
            forward_index=0,
            token_ids=list(range(81)),
            position_states=[PositionState.ACCEPTED] * 81,
            committed_positions=list(range(81)),
            decoded_text="",
            token_texts=digits,
        )
    ]
    sample = Sample(sample_id="sudoku-demo-0", prompt="solve", reference=SudokuReference(puzzle=puzzle, solution=solution))

    written = render_sample_report(
        sample_id=sample.sample_id,
        trace=trace,
        final_valid_length=81,
        out_dir=str(tmp_path),
        dataset_name="sudoku",
        sample=sample,
    )
    assert "sudoku_gif" in written
    assert Path(written["sudoku_gif"]).exists()


def test_render_sample_report_renders_instructed_sudoku_from_decoded_canvas(tmp_path):
    from dllm_bench.datasets.base import Sample
    from dllm_bench.datasets.sudoku import SudokuReference
    from dllm_bench.interfaces import PositionState, TraceStep

    solution = [[(row * 3 + row // 3 + col) % 9 + 1 for col in range(9)] for row in range(9)]
    puzzle = [row[:] for row in solution]
    puzzle[0][0] = 0
    digits = "".join(str(value) for row in solution for value in row)
    trace = [
        TraceStep(
            forward_index=0,
            token_ids=list(range(256)),
            position_states=[PositionState.VISIBLE] * 256,
            committed_positions=[],
            decoded_text=digits,
        )
    ]
    sample = Sample("sudoku-test-0000", "solve", SudokuReference(puzzle, solution))

    written = render_sample_report(
        sample_id=sample.sample_id,
        trace=trace,
        final_valid_length=81,
        out_dir=str(tmp_path),
        dataset_name="sudoku_trace",
        sample=sample,
    )

    assert Path(written["sudoku_gif"]).exists()
