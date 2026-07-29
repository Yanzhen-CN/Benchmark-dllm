"""Full pipeline smoke test using the mock adapter: generate -> score ->
measure resources -> aggregate -> persist -> render report/trace output.
Exercises every layer (interfaces, resource measurement, metrics, datasets,
orchestrator, report) without needing a GPU, real weights, or network access.
"""

from __future__ import annotations

import re

import pytest

from dllm_bench.datasets.base import ScoreResult
from dllm_bench.datasets.gsm8k import GSM8KDataset
from dllm_bench.interfaces import (
    GenerationRequest,
    GenerationResult,
    RunStatus,
    TimingResult,
)
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.report.tables import compute_converted_row, raw_results_row, render_raw_results_table
from dllm_bench.report.trace_report import render_sample_report
from dllm_bench.runner.demo_samples import build_demo_samples
from dllm_bench.runner.orchestrator import SampleRecord, run_experiment, summarize_records
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
    assert summary.tps is not None and summary.tps > 0
    assert summary.sps is not None and summary.sps > 0
    assert summary.sps == pytest.approx(1.0 / summary.time_per_sample)
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


def test_resource_rates_include_timed_failed_samples_in_window_denominator():
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=2)
    records = [
        SampleRecord(
            sample=samples[0],
            generation=GenerationResult(
                request=GenerationRequest("one", 16),
                output_text="ok",
                status=RunStatus.SUCCESS,
                final_valid_length=10,
                timing=TimingResult(2.0),
                energy_joules=4.0,
                peak_vram_gb=3.0,
            ),
            score=ScoreResult(1.0),
        ),
        SampleRecord(
            sample=samples[1],
            generation=GenerationResult(
                request=GenerationRequest("two", 16),
                output_text="",
                status=RunStatus.FAILED,
                final_valid_length=0,
                timing=TimingResult(3.0),
                energy_joules=6.0,
                peak_vram_gb=4.0,
            ),
            score=ScoreResult(0.0, valid=False, complete=False),
        ),
    ]

    summary = summarize_records("model", "config", dataset, records)

    assert summary.time_per_sample == pytest.approx(2.5)
    assert summary.energy_per_sample == pytest.approx(5.0)
    assert summary.tps == pytest.approx(10 / 5)
    assert summary.sps == pytest.approx(2 / 5)
    assert summary.eps == pytest.approx(10 / 5)
    assert summary.peak_vram_gb == pytest.approx(4.0)


def test_resource_rates_are_unavailable_when_any_selected_sample_lacks_timing():
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=2)
    records = [
        SampleRecord(
            sample=samples[0],
            generation=GenerationResult(
                request=GenerationRequest("one", 16),
                output_text="ok",
                status=RunStatus.SUCCESS,
                final_valid_length=10,
                timing=TimingResult(2.0),
            ),
            score=ScoreResult(1.0),
        ),
        SampleRecord(
            sample=samples[1],
            generation=GenerationResult(
                request=GenerationRequest("two", 16),
                output_text="",
                status=RunStatus.FAILED,
            ),
            score=ScoreResult(0.0, valid=False, complete=False),
        ),
    ]

    summary = summarize_records("model", "config", dataset, records)

    assert summary.time_per_sample is None
    assert summary.tps is None
    assert summary.sps is None
    assert summary.eps is None


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
    assert loaded["sps"] == pytest.approx(summary.sps)
    assert "records" not in loaded  # summary.json is aggregate-only by design

    row = raw_results_row(loaded)
    table = render_raw_results_table([row])
    assert "gsm8k" in table
    assert "mock" in table


def test_two_runs_feed_the_converted_results_table(tmp_path):
    baseline_adapter = MockDiffusionAdapter(name="qwen3_4b", config_name="ar-baseline", response_fn=_correct_gsm8k_response, steps=2)
    fast_adapter = MockDiffusionAdapter(name="illada", config_name="fast", response_fn=_correct_gsm8k_response, steps=8)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=3)

    baseline_summary = run_experiment(baseline_adapter, dataset, samples, max_new_tokens=16)
    fast_summary = run_experiment(fast_adapter, dataset, samples, max_new_tokens=16)

    baseline_dict = load_run_summary_dict(_dump(tmp_path / "baseline.json", baseline_summary))
    fast_dict = load_run_summary_dict(_dump(tmp_path / "fast.json", fast_summary))

    row = compute_converted_row(fast_dict, baseline_dict)
    assert row["Model"] == "illada"
    assert row["r_speed"] is not None


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
    assert set(written) >= {"heatmap", "token_grid_gif", "certainty", "result"}
