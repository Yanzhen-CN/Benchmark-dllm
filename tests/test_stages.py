"""generate_stage/score_stage: per-sample resume behavior and the
generate-on-one-machine / score-on-another split."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from dllm_bench.datasets.gsm8k import GSM8KDataset
from dllm_bench.interfaces import GenerationRequest, GenerationResult, RunStatus
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.runner.demo_samples import build_demo_samples
from dllm_bench.runner.generate_stage import OOMInvalidTestError, run_generation
from dllm_bench.runner.persistence import (
    load_generation_result,
    load_score_result,
    save_generation_result,
)
from dllm_bench.runner.score_stage import (
    IncompleteTestError,
    InvalidTestError,
    run_scoring,
)


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
    meta = json.loads((out_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["test_valid"] is True
    assert meta["test_complete"] is True
    assert meta["selected_samples"] == 3
    assert meta["completed_samples"] == 3


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


def test_compute_replays_only_after_all_formal_generations(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    samples = build_demo_samples("gsm8k", n=3)
    events = []
    real_generate = adapter.generate

    def generate(request):
        events.append(("generate", request.sample_id))
        return real_generate(request)

    def profile_compute(request):
        events.append(("compute", request.sample_id))
        return SimpleNamespace(available=True, tflops=1.25)

    adapter.generate = generate
    adapter.profile_compute = profile_compute
    out_dir = tmp_path / "model_output"
    run_generation(
        adapter,
        "gsm8k",
        samples,
        max_new_tokens=16,
        out_dir=out_dir,
        measure_compute=True,
    )

    ids = [sample.sample_id for sample in samples]
    assert events == [*(('generate', sample_id) for sample_id in ids),
                      *(('compute', sample_id) for sample_id in ids)]
    assert all(
        load_generation_result(out_dir / f"{sample_id}.json").compute_tflops == 1.25
        for sample_id in ids
    )


def test_resume_fills_missing_compute_without_regenerating(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    samples = build_demo_samples("gsm8k", n=2)
    adapter.profile_compute = lambda request: SimpleNamespace(available=True, tflops=2.0)
    out_dir = tmp_path / "model_output"
    run_generation(
        adapter,
        "gsm8k",
        samples,
        max_new_tokens=16,
        out_dir=out_dir,
        measure_compute=True,
    )

    first_path = out_dir / f"{samples[0].sample_id}.json"
    first = load_generation_result(first_path)
    first.compute_tflops = None
    save_generation_result(first, first_path)

    adapter.generate = lambda request: (_ for _ in ()).throw(
        AssertionError("resume regenerated a completed formal sample")
    )
    replayed = []
    adapter.profile_compute = lambda request: (
        replayed.append(request.sample_id)
        or SimpleNamespace(available=True, tflops=3.0)
    )
    summary = run_generation(
        adapter,
        "gsm8k",
        samples,
        max_new_tokens=16,
        out_dir=out_dir,
        measure_compute=True,
    )

    assert summary.generated == 0
    assert summary.skipped == 2
    assert replayed == [samples[0].sample_id]
    assert load_generation_result(first_path).compute_tflops == 3.0


def test_compute_can_be_added_to_a_no_compute_run_without_regeneration(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    samples = build_demo_samples("gsm8k", n=2)
    out_dir = tmp_path / "model_output"
    run_generation(
        adapter,
        "gsm8k",
        samples,
        max_new_tokens=16,
        out_dir=out_dir,
        measure_compute=False,
        require_all_metrics=False,
    )
    # The real formal entry point already requires these directly measured
    # fields. Populate them here because the mock adapter has no NVML backend.
    for sample in samples:
        sample_path = out_dir / f"{sample.sample_id}.json"
        generation = load_generation_result(sample_path)
        generation.energy_joules = 1.0
        generation.peak_vram_gb = 2.0
        save_generation_result(generation, sample_path)

    adapter.generate = lambda request: (_ for _ in ()).throw(
        AssertionError("compute supplementation regenerated a formal sample")
    )
    replayed = []
    adapter.profile_compute = lambda request: (
        replayed.append(request.sample_id)
        or SimpleNamespace(available=True, tflops=4.0)
    )
    summary = run_generation(
        adapter,
        "gsm8k",
        samples,
        max_new_tokens=16,
        out_dir=out_dir,
        measure_compute=True,
        require_all_metrics=True,
    )

    assert summary.generated == 0
    assert summary.skipped == 2
    assert replayed == [sample.sample_id for sample in samples]
    assert all(
        load_generation_result(out_dir / f"{sample.sample_id}.json").compute_tflops
        == 4.0
        for sample in samples
    )


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


def test_run_generation_can_force_one_temporary_output_length(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=2)
    samples[0].meta["max_new_tokens"] = 7
    samples[1].meta["max_new_tokens"] = 13
    out_dir = tmp_path / "model_output"

    run_generation(
        adapter,
        "gsm8k",
        samples,
        max_new_tokens=512,
        out_dir=out_dir,
        force_max_new_tokens=True,
    )

    for sample in samples:
        result = load_generation_result(out_dir / f"{sample.sample_id}.json")
        assert result.request.max_new_tokens == 512


def test_run_generation_passes_ruler_input_budget_to_adapter(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    sample = build_demo_samples("gsm8k", n=1)[0]
    sample.meta["target_input_tokens"] = 4096
    out_dir = tmp_path / "model_output"

    run_generation(adapter, "ruler", [sample], max_new_tokens=64, out_dir=out_dir)

    result = load_generation_result(out_dir / f"{sample.sample_id}.json")
    assert result.request.config["target_input_tokens"] == 4096


@pytest.mark.parametrize("dataset_name", ["ruler", "hellobench"])
def test_long_task_stops_after_first_oom_and_persists_boundary(
    tmp_path, dataset_name
):
    samples = build_demo_samples("gsm8k", n=3)
    if dataset_name == "hellobench":
        samples[0].meta.update(target_output_words=2000, max_new_tokens=3072)

    class OOMAdapter:
        name = "oom-model"
        config_name = "default"
        supports_trace = False
        natively_measures_resources = True

        def __init__(self):
            self.calls = []

        def generate(self, request):
            self.calls.append(request.sample_id)
            return GenerationResult(
                request=request,
                output_text="",
                status=RunStatus.OOM,
                error_message="CUDA out of memory",
            )

    adapter = OOMAdapter()
    out_dir = tmp_path / "model_output"
    with pytest.raises(OOMInvalidTestError, match="OOM invalidated"):
        run_generation(
            adapter, dataset_name, samples, max_new_tokens=64, out_dir=out_dir
        )

    assert adapter.calls == [samples[0].sample_id]
    assert not (out_dir / f"{samples[1].sample_id}.json").exists()

    first = load_generation_result(out_dir / f"{samples[0].sample_id}.json")
    assert first.extra["oom_sample_ordinal"] == 1
    assert first.extra["selected_samples"] == 3
    if dataset_name == "hellobench":
        assert first.extra["oom_sample_context"] == {
            "target_output_words": 2000,
            "max_new_tokens": 3072,
        }
    meta = json.loads((out_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["test_valid"] is False
    assert meta["test_complete"] is False
    assert meta["invalid_reason"] == "oom"
    assert meta["early_stop"] == {
        "reason": "oom",
        "sample_ordinal": 1,
        "sample_id": samples[0].sample_id,
        "attempted_samples": 1,
        "selected_samples": 3,
        "remaining_samples_not_attempted": 2,
        "sample_context": (
            {"target_output_words": 2000, "max_new_tokens": 3072}
            if dataset_name == "hellobench"
            else {}
        ),
    }
    oom_info = json.loads((out_dir / "oom_info.json").read_text(encoding="utf-8"))
    assert oom_info["test_valid"] is False
    assert oom_info["scope"] == "model_x_variant_x_dataset"
    assert oom_info["model_name"] == "oom-model"
    assert oom_info["config_name"] == "default"
    assert oom_info["dataset_name"] == dataset_name
    assert oom_info["sample_ordinal"] == 1
    assert oom_info["sample_id"] == samples[0].sample_id
    assert oom_info["error_type"] == "cuda_out_of_memory"
    assert oom_info["error_message"] == "CUDA out of memory"
    assert "gpu" in oom_info
    assert "run_metadata" in oom_info

    with pytest.raises(OOMInvalidTestError, match="OOM invalidated"):
        run_generation(
            adapter, dataset_name, samples, max_new_tokens=64, out_dir=out_dir
        )
    assert adapter.calls == [samples[0].sample_id]


def test_any_dataset_stops_and_becomes_invalid_after_oom(tmp_path):
    samples = build_demo_samples("gsm8k", n=2)

    class OOMAdapter:
        name = "oom-model"
        config_name = "default"
        supports_trace = False
        natively_measures_resources = True

        def __init__(self):
            self.calls = []

        def generate(self, request):
            self.calls.append(request.sample_id)
            return GenerationResult(
                request=request,
                output_text="",
                status=RunStatus.OOM,
                error_message="CUDA out of memory",
            )

    adapter = OOMAdapter()
    with pytest.raises(OOMInvalidTestError, match="OOM invalidated"):
        run_generation(
            adapter, "gsm8k", samples, max_new_tokens=16, out_dir=tmp_path / "out"
        )

    assert adapter.calls == [samples[0].sample_id]


def test_long_task_migrates_existing_oom_to_early_stop_marker(tmp_path):
    samples = build_demo_samples("gsm8k", n=2)
    out_dir = tmp_path / "model_output"
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    run_generation(
        adapter, "gsm8k", samples[:1], max_new_tokens=16, out_dir=out_dir
    )
    first_path = out_dir / f"{samples[0].sample_id}.json"
    existing = load_generation_result(first_path)
    existing.status = RunStatus.OOM
    existing.error_message = "CUDA out of memory"
    save_generation_result(existing, first_path)

    adapter.generate = lambda request: (_ for _ in ()).throw(
        AssertionError("sample after an existing long-task OOM was attempted")
    )
    with pytest.raises(OOMInvalidTestError, match="OOM invalidated"):
        run_generation(
            adapter, "ruler", samples, max_new_tokens=64, out_dir=out_dir
        )

    meta = json.loads((out_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["early_stop"]["sample_ordinal"] == 1


def test_run_generation_can_disable_trace_without_losing_forward_count(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    sample = build_demo_samples("gsm8k", n=1)[0]
    out_dir = tmp_path / "model_output"

    run_generation(
        adapter,
        "hellobench",
        [sample],
        max_new_tokens=16,
        out_dir=out_dir,
        capture_trace=False,
    )

    generation = load_generation_result(out_dir / f"{sample.sample_id}.json")
    assert generation.trace == []
    assert generation.num_forward_passes == 2
    assert generation.request.config["capture_trace"] is False
    meta = json.loads((out_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["run_metadata"]["trace_scope"] == "none"


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


def test_run_generation_refuses_to_resume_with_changed_output_budget(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    samples = build_demo_samples("gsm8k", n=1)
    out_dir = tmp_path / "model_output"

    run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir)

    with pytest.raises(RuntimeError, match="max_new_tokens=16.*requires 32.*--no-resume"):
        run_generation(
            adapter,
            "gsm8k",
            samples,
            max_new_tokens=32,
            out_dir=out_dir,
            resume=True,
        )


def test_run_generation_refuses_to_mix_legacy_measurement_protocol(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    samples = build_demo_samples("gsm8k", n=1)
    out_dir = tmp_path / "model_output"
    out_dir.mkdir()
    (out_dir / "_meta.json").write_text(
        '{"run_metadata": {"measurement_protocol": "legacy"}}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="--no-resume"):
        run_generation(
            adapter, "gsm8k", samples, max_new_tokens=16, out_dir=out_dir
        )


def test_run_generation_require_all_metrics_fails_before_persisting_missing_metrics(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=2)
    samples = build_demo_samples("gsm8k", n=1)
    out_dir = tmp_path / "model_output"

    with pytest.raises(RuntimeError, match="energy_joules, peak_vram_gb"):
        run_generation(
            adapter,
            "gsm8k",
            samples,
            max_new_tokens=16,
            out_dir=out_dir,
            require_all_metrics=True,
        )

    assert not (out_dir / f"{samples[0].sample_id}.json").exists()


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


def test_run_scoring_refuses_an_oom_invalidated_test(tmp_path):
    samples = build_demo_samples("gsm8k", n=2)

    class OOMAdapter:
        name = "oom-model"
        config_name = "default"
        supports_trace = False
        natively_measures_resources = True

        def generate(self, request):
            return GenerationResult(
                request=request,
                output_text="",
                status=RunStatus.OOM,
                error_message="CUDA out of memory",
            )

    model_out = tmp_path / "model_output"
    score_out = tmp_path / "score_output"
    with pytest.raises(OOMInvalidTestError):
        run_generation(
            OOMAdapter(), "gsm8k", samples, max_new_tokens=16, out_dir=model_out
        )

    with pytest.raises(InvalidTestError, match="test is invalid because of OOM"):
        run_scoring(GSM8KDataset(), samples, model_out, score_out)
    assert not score_out.exists()


def test_variant_dataset_oom_does_not_invalidate_completed_sibling_variant(tmp_path):
    samples = build_demo_samples("gsm8k", n=1)
    model_root = tmp_path / "model_output"
    fast_out = model_root / "mock_fast" / "gsm8k"
    best_out = model_root / "mock_best" / "gsm8k"

    fast_adapter = MockDiffusionAdapter(
        config_name="fast", response_fn=_correct_gsm8k_response, steps=2
    )
    run_generation(
        fast_adapter, "gsm8k", samples, max_new_tokens=16, out_dir=fast_out
    )

    class OOMBestAdapter:
        name = "mock"
        config_name = "best"
        supports_trace = False
        natively_measures_resources = True

        def generate(self, request):
            return GenerationResult(
                request=request,
                output_text="",
                status=RunStatus.OOM,
                error_message="CUDA out of memory",
            )

    with pytest.raises(OOMInvalidTestError):
        run_generation(
            OOMBestAdapter(), "gsm8k", samples, max_new_tokens=16, out_dir=best_out
        )

    result = run_scoring(
        GSM8KDataset(), samples, fast_out, tmp_path / "score_output"
    )
    assert result.summary.q == pytest.approx(1.0)


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


def test_run_scoring_persists_dataset_trace_aux_metrics(tmp_path):
    class TraceAwareGSM8KDataset(GSM8KDataset):
        def trace_aux_metrics(self, sample, trace):
            return {"test_trace_step_count": float(len(trace))}

    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = TraceAwareGSM8KDataset()
    samples = build_demo_samples("gsm8k", n=1)
    model_out = tmp_path / "model_output"
    score_out = tmp_path / "score_output"
    run_generation(adapter, "gsm8k", samples, max_new_tokens=16, out_dir=model_out)

    run_scoring(dataset, samples, model_out, score_out)

    score = load_score_result(score_out / f"{samples[0].sample_id}.json")
    assert score.aux["test_trace_step_count"] > 0


def test_run_scoring_keeps_sample_scores_but_refuses_partial_summary(tmp_path):
    adapter = MockDiffusionAdapter(response_fn=_correct_gsm8k_response, steps=4)
    dataset = GSM8KDataset()
    samples = build_demo_samples("gsm8k", n=3)
    model_out = tmp_path / "model_output"
    score_out = tmp_path / "score_output"

    run_generation(adapter, "gsm8k", samples[:2], max_new_tokens=16, out_dir=model_out)
    with pytest.raises(IncompleteTestError, match="1 of 3 selected generation"):
        run_scoring(dataset, samples, model_out, score_out)

    assert (score_out / f"{samples[0].sample_id}.json").exists()
    assert (score_out / f"{samples[1].sample_id}.json").exists()
    assert not (score_out / "summary.json").exists()


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
