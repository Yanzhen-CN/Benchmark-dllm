"""Full pipeline smoke test using the mock adapter: generate -> score ->
measure resources -> aggregate -> persist -> render report/trace output.
Exercises every layer (interfaces, resource measurement, metrics, datasets,
orchestrator, report) without needing a GPU, real weights, or network access.
"""

from __future__ import annotations

import re

import pytest

from dllm_bench.datasets.gsm8k import GSM8KDataset
from dllm_bench.interfaces import GenerationRequest, RunStatus
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.report.tables import compute_converted_row, raw_results_row, render_raw_results_table
from dllm_bench.report.trace_report import render_sample_report
from dllm_bench.runner.demo_samples import build_demo_samples
from dllm_bench.runner.orchestrator import run_experiment
from dllm_bench.runner.persistence import load_run_summary_dict, save_run_summary


def _correct_gsm8k_response(request: GenerationRequest) -> str:
    numbers = [int(n) for n in re.findall(r"\d+", request.prompt)]
    return f"Adding them up gives #### {sum(numbers)}"


def test_mock_adapter_solves_demo_gsm8k_samples_end_to_end():
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=5)

    summary = run_experiment(
        adapter=adapter,
        dataset=dataset,
        samples=samples,
        max_new_tokens=32,
        measure_compute=True,
    )

    assert summary.q == pytest.approx(1.0)
    assert summary.status_counts == {"success": 5}
    assert summary.n_samples == 5
    assert summary.time_per_sample is not None and summary.time_per_sample >= 0
    assert summary.timing_source == "measured"
    assert len(summary.records) == 5
    assert all(r.generation.status == RunStatus.SUCCESS for r in summary.records)
    assert all(r.generation.has_trace for r in summary.records)


def test_mock_adapter_wrong_answers_give_zero_quality():
    adapter = MockDiffusionAdapter(steps=4)  # default response_fn never matches gsm8k arithmetic
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=3)

    summary = run_experiment(adapter=adapter, dataset=dataset, samples=samples, max_new_tokens=16)
    assert summary.q == 0.0


def test_run_summary_round_trips_through_persistence_and_report(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=4)
    summary = run_experiment(adapter=adapter, dataset=dataset, samples=samples, max_new_tokens=16)

    out_path = tmp_path / "summary.json"
    save_run_summary(summary, out_path)
    assert out_path.exists()

    loaded = load_run_summary_dict(out_path)
    assert loaded["dataset_name"] == "gsm8k"
    assert loaded["model_name"] == "mock"
    assert loaded["q"] == pytest.approx(1.0)
    assert "records" not in loaded  # summary.json is aggregate-only by design

    row = raw_results_row(loaded)
    table = render_raw_results_table([row])
    assert "gsm8k" in table
    assert "mock" in table


def test_two_runs_feed_the_converted_results_table(tmp_path):
    baseline_adapter = MockDiffusionAdapter(name="qwen3-4b", config_name="ar-baseline", response_fn=_correct_gsm8k_response, steps=2)
    fast_adapter = MockDiffusionAdapter(name="illada", config_name="fast", response_fn=_correct_gsm8k_response, steps=8)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=3)

    baseline_summary = run_experiment(baseline_adapter, dataset, samples, max_new_tokens=16)
    fast_summary = run_experiment(fast_adapter, dataset, samples, max_new_tokens=16)

    baseline_dict = load_run_summary_dict(_dump(tmp_path / "baseline.json", baseline_summary))
    fast_dict = load_run_summary_dict(_dump(tmp_path / "fast.json", fast_summary))

    row = compute_converted_row(fast_dict, baseline_dict)
    assert row["Model"] == "illada"
    # both q's are 1.0 (both solved every sample correctly) so Q_time/Q_energy
    # collapse to 1.0 regardless of the resource ratio - just check the shape.
    assert row["r_time"] is not None


def _dump(path, summary) -> str:
    save_run_summary(summary, path)
    return str(path)


def test_render_sample_report_from_a_real_run_record(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=1)
    summary = run_experiment(adapter=adapter, dataset=dataset, samples=samples, max_new_tokens=16)

    record = summary.records[0]
    written = render_sample_report(
        sample_id=record.sample.sample_id,
        trace=record.generation.trace,
        final_valid_length=record.generation.final_valid_length,
        out_dir=str(tmp_path),
        final_output_text=record.generation.output_text,
        final_score=record.score.primary_score,
        dataset_name=dataset.name,
        sample=record.sample,
    )
    assert set(written) >= {"token_grid_gif", "parallelism", "certainty", "result"}
