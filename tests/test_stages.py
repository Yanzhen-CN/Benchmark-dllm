"""generate_stage/score_stage: per-sample resume behavior and the
generate-on-one-machine / score-on-another split."""

from __future__ import annotations

import re

import pytest

from dllm_bench.datasets.gsm8k import GSM8KDataset
from dllm_bench.interfaces import GenerationRequest
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.runner.demo_samples import build_demo_samples
from dllm_bench.runner.generate_stage import run_generation
from dllm_bench.runner.persistence import load_generation_result
from dllm_bench.runner.score_stage import run_scoring


def _correct_gsm8k_response(request: GenerationRequest) -> str:
    numbers = [int(n) for n in re.findall(r"\d+", request.prompt)]
    return f"#### {sum(numbers)}"


def test_run_generation_writes_one_file_per_sample(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=3)
    out_dir = tmp_path / "model_output"

    summary = run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir)

    assert summary.generated == 3
    assert summary.skipped == 0
    for sample in samples:
        assert (out_dir / f"{sample.sample_id}.json").exists()


def test_run_generation_reports_per_sample_start_and_finish(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    samples = build_demo_samples("gsm8k", n=2)
    events = []

    run_generation(
        adapter,
        "gsm8k",
        samples,
        max_new_tokens=16,
        out_dir=tmp_path / "model_output",
        progress=lambda event, index, total, sample, generation: events.append(
            (event, index, total, sample.sample_id, generation is not None)
        ),
    )

    assert events == [
        ("start", 1, 2, samples[0].sample_id, False),
        ("finish", 1, 2, samples[0].sample_id, True),
        ("start", 2, 2, samples[1].sample_id, False),
        ("finish", 2, 2, samples[1].sample_id, True),
    ]


def test_run_generation_uses_per_sample_max_new_tokens(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=2)
    samples[0].meta["max_new_tokens"] = 7
    samples[1].meta["max_new_tokens"] = 13
    out_dir = tmp_path / "model_output"

    run_generation(adapter, "gsm8k", samples, max_new_tokens=256, out_dir=out_dir)

    first = load_generation_result(out_dir / f"{samples[0].sample_id}.json")
    second = load_generation_result(out_dir / f"{samples[1].sample_id}.json")
    assert first.request.max_new_tokens == 7
    assert second.request.max_new_tokens == 13
    assert (out_dir / "_meta.json").exists()


def test_run_generation_resumes_and_skips_existing_samples(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=3)
    out_dir = tmp_path / "model_output"

    run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir)

    call_count = {"n": 0}
    real_generate = adapter.generate

    def counting_generate(request):
        call_count["n"] += 1
        return real_generate(request)

    adapter.generate = counting_generate
    second = run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir)

    assert second.skipped == 3
    assert second.generated == 0
    assert call_count["n"] == 0  # nothing re-generated


def test_run_generation_no_resume_regenerates_everything(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=2)
    out_dir = tmp_path / "model_output"

    run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir)
    second = run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir, resume=False)

    assert second.generated == 2
    assert second.skipped == 0


def test_run_generation_partial_then_resume_only_fills_the_gap(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=5)
    out_dir = tmp_path / "model_output"

    run_generation(adapter, "gsm8k", samples[:2], max_new_tokens=16, out_dir=out_dir)
    second = run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir)

    assert second.skipped == 2
    assert second.generated == 3


def test_run_scoring_requires_generation_first(tmp_path):
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=2)
    with pytest.raises(FileNotFoundError):
        run_scoring(dataset, samples, tmp_path / "model_output", tmp_path / "score_output")


def test_run_scoring_end_to_end_after_generation(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=3)
    model_out = tmp_path / "model_output"
    score_out = tmp_path / "score_output"

    run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=model_out)
    result = run_scoring(dataset, samples, model_out, score_out)

    assert result.summary.q == pytest.approx(1.0)
    assert result.scored == 3
    assert result.skipped == 0
    assert not result.missing_sample_ids
    assert (score_out / "summary.json").exists()
    for sample in samples:
        assert (score_out / f"{sample.sample_id}.json").exists()


def test_run_scoring_reports_missing_samples_without_crashing(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=3)
    model_out = tmp_path / "model_output"
    score_out = tmp_path / "score_output"

    run_generation(adapter, "gsm8k", samples[:2], max_new_tokens=16, out_dir=model_out)
    result = run_scoring(dataset, samples, model_out, score_out)

    assert result.missing_sample_ids == [samples[2].sample_id]
    assert result.summary.n_samples == 2


def test_run_scoring_resume_skips_rescoring(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=2)
    model_out = tmp_path / "model_output"
    score_out = tmp_path / "score_output"

    run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=model_out)
    run_scoring(dataset, samples, model_out, score_out)
    second = run_scoring(dataset, samples, model_out, score_out)

    assert second.scored == 0
    assert second.skipped == 2


def test_generate_then_score_on_separate_dataset_objects_still_works(tmp_path):
    """Simulates generate-on-server / score-locally: a fresh Dataset instance
    (no shared in-memory state with the one used for generation) still scores
    correctly purely from persisted model_output."""
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=2)
    model_out = tmp_path / "model_output"
    score_out = tmp_path / "score_output"

    run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=model_out)
    fresh_dataset = GSM8KDataset()
    result = run_scoring(fresh_dataset, samples, model_out, score_out)
    assert result.summary.q == pytest.approx(1.0)
