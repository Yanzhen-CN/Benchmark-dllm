from pathlib import Path

import pytest

from dllm_bench.interfaces import GenerationRequest
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.visual.public.plots import (
    plot_answer_region_diagnostics,
    plot_p1_vs_p2,
    plot_quality_vs_resource,
    plot_score_per_unit,
    plot_speculative_acceptance,
    plot_task4_forward_yield,
)
from dllm_bench.visual.public.pairwise import (
    PairwiseCompatibilityError,
    compute_pairwise_row,
    render_pairwise_table,
    write_pairwise_outputs,
)
from dllm_bench.visual.public.tables import raw_results_row, render_raw_results_table
from dllm_bench.visual.public.trace_report import render_sample_report


def _summary(model, config, q, tps=1.0, eps=None):
    return {
        "dataset_name": "gsm8k",
        "model_name": model,
        "config_name": config,
        "q": q,
        "tps": tps,
        "accepted_tps": tps,
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
        "n_samples": 5,
        "aux": {
            "answer_start_char_ratio": 0.4,
            "answer_region_detected_rate": 1.0,
        },
        "run_metadata": {
            "measurement_protocol": "gpu-synced-v4",
            "cuda_devices": ["test GPU"],
        },
        "scoring_metadata": {
            "sample_set_hash": "same-samples",
            "dataset_revision": "same-dataset",
            "prompt_protocol_revision": "same-prompts",
            "generation_protocol_revision": "same-budgets",
            "expected_sample_count": 5,
            "primary_metric": "accuracy",
        },
    }


def test_dflash_acceptance_and_forward_yield_plots(tmp_path):
    row = raw_results_row(_summary("gemma_dflash", "dflash", 0.8, tps=40.0))
    row["Aux"].update(
        {
            "speculative_draft_acceptance_rate": 0.4,
            "speculative_mean_acceptance_length": 7.0,
        }
    )
    acceptance = tmp_path / "acceptance.png"
    forward_yield = tmp_path / "yield.png"
    plot_speculative_acceptance([row], str(acceptance))
    plot_task4_forward_yield([row], str(forward_yield))
    assert acceptance.exists()
    assert forward_yield.exists()


def test_raw_results_row_shape():
    row = raw_results_row(_summary("mock", "default", 0.8))
    assert row["Dataset"] == "gsm8k"
    assert row["Model"] == "mock"
    assert row["q"] == 0.8
    assert row["Seconds/Sample"] == pytest.approx(1.0)
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


def test_compute_pairwise_row_uses_seconds_per_sample_not_tps():
    baseline = _summary("qwen3_8b", "ar-baseline", 0.5, tps=100.0)
    fast_model = _summary("illada", "p2", 0.5, tps=1.0)
    baseline["time_per_sample"] = 10.0
    fast_model["time_per_sample"] = 2.0
    row, metadata = compute_pairwise_row(fast_model, baseline, beta=100, gamma=50)
    assert row["r_speed"] == pytest.approx(5.0)
    assert row["Q_speed"] > 0.5
    assert metadata["direction"] == "illada/p2 relative to qwen3_8b/ar-baseline"


def test_compute_pairwise_row_missing_energy_leaves_energy_fields_none():
    baseline = _summary("qwen3_4b", "ar-baseline", 0.5, tps=10.0)
    model = _summary("illada", "p1", 0.6, tps=8.0)
    row, _ = compute_pairwise_row(model, baseline, beta=50, gamma=75)
    assert row["r_energy"] is None
    assert row["Q_energy"] is None
    assert row["Q_beta_gamma"] is None


def test_pairwise_requires_matching_sample_set():
    baseline = _summary("qwen3_8b", "ar-baseline", 0.5)
    model = _summary("illada", "p2", 0.6)
    model["scoring_metadata"]["sample_set_hash"] = "different"
    with pytest.raises(PairwiseCompatibilityError, match="sample set hash differs"):
        compute_pairwise_row(model, baseline, beta=50, gamma=50)


def test_self_reported_timing_is_not_compared_with_measured_baseline():
    baseline = _summary("qwen3_4b", "ar-baseline", 0.5, tps=10.0)
    w1 = _summary("w1", "standard", 0.6, tps=50.0)
    w1["timing_source"] = "self_reported"

    with pytest.raises(PairwiseCompatibilityError, match="timing sources"):
        compute_pairwise_row(w1, baseline, beta=50, gamma=50)


def test_render_and_write_pairwise_outputs_smoke(tmp_path):
    baseline = _summary("qwen3_4b", "ar-baseline", 0.5, tps=10.0)
    model = _summary("illada", "p1", 0.6, tps=8.0)
    baseline["energy_per_sample"] = 20.0
    model["energy_per_sample"] = 10.0
    row, metadata = compute_pairwise_row(model, baseline, beta=60, gamma=30)
    table = render_pairwise_table(row, metadata)
    assert "r_speed" in table
    assert "sample_set_hash=same-samples" in table
    written = write_pairwise_outputs(row, metadata, tmp_path)
    assert len(written) == 4
    assert all(path.exists() for path in written)


# ---------------------------------------------------------------------------
# Plots: smoke-test that files actually get written (content correctness is
# out of scope for a unit test — matplotlib rendering itself is trusted).
# ---------------------------------------------------------------------------

def test_plot_quality_vs_resource_writes_file(tmp_path):
    rows = [raw_results_row(_summary("mock", "default", 0.8, tps=1.5))]
    out = tmp_path / "quality_time.png"
    plot_quality_vs_resource(rows, "Accepted TPS", str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_score_per_unit_skips_when_no_data(tmp_path):
    rows = [raw_results_row(_summary("mock", "default", 0.8))]
    out = tmp_path / "score_per_j.png"
    plot_score_per_unit(rows, "Score per Unit Energy", str(out))
    assert not out.exists()


def test_plot_p1_vs_p2_writes_file_when_both_configs_present(tmp_path):
    rows = [
        raw_results_row(_summary("illada", "p1", 0.7, tps=5.0)),
        raw_results_row(_summary("illada", "p2", 0.6, tps=10.0)),
    ]
    out = tmp_path / "p1_vs_p2.png"
    plot_p1_vs_p2(rows, "q", str(out))
    assert out.exists()


def test_plot_answer_region_diagnostics_writes_file(tmp_path):
    row = raw_results_row(_summary("illada", "p1", 0.6))
    out = tmp_path / "answer_region.png"
    plot_answer_region_diagnostics([row], str(out))
    assert out.exists()


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
    # Curated sample evidence keeps one DGtest-style accept/revision trace.
    for key in (
        "accept_trace",
        "result",
    ):
        assert key in written
        assert Path(written[key]).exists()
    assert "entropy" not in written
    assert "token_grid_gif" not in written
    assert "parallelism" not in written
    assert "strategy" not in written
    assert "token_grid_final" not in written
    assert "position_vs_commit" not in written
    assert "speed" not in written


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
    from dllm_bench.datasets.sudoku9 import SudokuReference
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
        dataset_name="sudoku9",
        sample=sample,
    )
    assert "sudoku_gif" in written
    assert Path(written["sudoku_gif"]).exists()
    assert "token_trace_gif" in written
    assert Path(written["token_trace_gif"]).exists()


def test_render_sample_report_renders_instructed_sudoku_from_decoded_canvas(tmp_path):
    from dllm_bench.datasets.base import Sample
    from dllm_bench.datasets.sudoku9 import SudokuReference
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
