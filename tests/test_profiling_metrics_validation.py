from types import SimpleNamespace

import pytest

from dllm_bench.interfaces import (
    ForwardProfile,
    GenerationRequest,
    GenerationResult,
    RunStatus,
    TimingResult,
)
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.runner.demo_samples import build_demo_samples
from dllm_bench.runner.generate_stage import (
    _apply_compute_handle,
    _validate_required_metrics,
    run_generation,
)
from dllm_bench.runner.persistence import load_generation_result


class _Adapter:
    natively_measures_resources = False
    supports_trace = False


def _result() -> GenerationResult:
    return GenerationResult(
        request=GenerationRequest("prompt", 8, sample_id="sample"),
        output_text="answer",
        status=RunStatus.SUCCESS,
        timing=TimingResult(1.0),
        energy_joules=10.0,
        peak_vram_gb=2.0,
        compute_tflops=3.0,
        forward_profiles=[
            ForwardProfile(
                forward_index=0,
                phase="decode_cached",
                wall_clock_seconds=0.1,
                accepted_tokens=2,
            )
        ],
        extra={
            "stage_profiles": [
                {
                    "stage": "denoise_step",
                    "wall_clock_seconds": 0.1,
                    "compute_flops": None,
                    "compute_tflops": None,
                }
            ],
            "clean_replay_validation": {"status": "matched"},
        },
    )


def test_compute_merge_preserves_timed_step_and_stage_profiles():
    result = _result()
    handle = SimpleNamespace(
        available=True,
        flops=2_000_000_000_000,
        tflops=2.0,
        forward_tflops=[2.0],
        forward_flops=[2_000_000_000_000],
        forward_phases=["decode_cached"],
        stage_profiles=[
            {
                "stage": "denoise_step",
                "wall_clock_seconds": None,
                "compute_flops": 2_000_000_000_000,
                "compute_tflops": 2.0,
            }
        ],
        torch_profile={"status": "complete"},
    )

    _apply_compute_handle(result, handle)

    assert result.forward_profiles[0].wall_clock_seconds == 0.1
    assert result.forward_profiles[0].compute_tflops == 2.0
    assert result.extra["stage_profiles"][0]["wall_clock_seconds"] == 0.1
    assert result.extra["stage_profiles"][0]["compute_tflops"] == 2.0
    assert result.extra["step_compute_status"] == "complete"
    assert result.extra["stage_compute_status"] == "complete"


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("time", "time_per_step"),
        ("accepted", "accepted_tokens_per_step"),
        ("compute", "flops_per_step"),
    ],
)
def test_required_profiling_metrics_reject_incomplete_steps(field, expected):
    result = _result()
    result.extra["step_compute_status"] = "complete"
    result.extra["stage_compute_status"] = "complete"
    result.extra["stage_profiles"][0]["compute_flops"] = 1
    result.forward_profiles[0].compute_flops = 1
    result.forward_profiles[0].compute_tflops = 1e-12
    if field == "time":
        result.forward_profiles[0].wall_clock_seconds = None
    elif field == "accepted":
        result.forward_profiles[0].accepted_tokens = None
    else:
        result.forward_profiles[0].compute_flops = None

    with pytest.raises(RuntimeError, match=expected):
        _validate_required_metrics(
            _Adapter(),
            result,
            require_compute=True,
            require_trace=False,
        )


def test_profiling_pipeline_keeps_timing_acceptance_compute_and_clean_totals(
    tmp_path,
):
    adapter = MockDiffusionAdapter(response_fn=lambda request: "answer", steps=2)
    real_generate = adapter.generate

    def generate(request):
        result = real_generate(request)
        result.energy_joules = 3.0
        result.peak_vram_gb = 4.0
        if request.config.get("step_profiling"):
            result.forward_profiles = [
                ForwardProfile(
                    forward_index=index,
                    phase="denoise",
                    wall_clock_seconds=0.1 + index * 0.01,
                    accepted_tokens=1,
                )
                for index in range(2)
            ]
            result.extra["stage_profiles"] = [
                {
                    "stage": "denoise_step",
                    "wall_clock_seconds": 0.1 + index * 0.01,
                    "compute_flops": None,
                    "compute_tflops": None,
                }
                for index in range(2)
            ]
        else:
            result.timing = TimingResult(0.5)
        return result

    adapter.generate = generate
    adapter.profile_compute = lambda request: SimpleNamespace(
        available=True,
        flops=3_000_000_000_000,
        tflops=3.0,
        forward_tflops=[1.0, 2.0],
        forward_flops=[1_000_000_000_000, 2_000_000_000_000],
        forward_phases=["denoise", "denoise"],
        stage_profiles=[
            {
                "stage": "denoise_step",
                "compute_flops": 1_000_000_000_000,
                "compute_tflops": 1.0,
            },
            {
                "stage": "denoise_step",
                "compute_flops": 2_000_000_000_000,
                "compute_tflops": 2.0,
            },
        ],
        torch_profile={"status": "complete"},
    )
    sample = build_demo_samples("gsm8k", n=1)[0]
    out_dir = tmp_path / "model_profiling"

    run_generation(
        adapter,
        "gsm8k",
        [sample],
        max_new_tokens=16,
        out_dir=out_dir,
        measure_compute=True,
        require_all_metrics=True,
        capture_trace=False,
    )

    result = load_generation_result(out_dir / f"{sample.sample_id}.json")
    assert result.timing.wall_clock_seconds == 0.5
    assert result.timing.source == "clean_replay"
    assert [profile.wall_clock_seconds for profile in result.forward_profiles] == [
        0.1,
        0.11,
    ]
    assert [profile.accepted_tokens for profile in result.forward_profiles] == [1, 1]
    assert [profile.compute_tflops for profile in result.forward_profiles] == [
        1.0,
        2.0,
    ]
    assert result.extra["clean_replay_validation"]["status"] == "matched"
    assert result.extra["step_compute_status"] == "complete"
    assert result.extra["stage_compute_status"] == "complete"
