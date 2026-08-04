from dllm_bench.datasets.base import Sample
from dllm_bench.interfaces import (
    ForwardProfile,
    GenerationRequest,
    GenerationResult,
    RunStatus,
)
from dllm_bench.visual.public.profiling_report import (
    build_dataset_profiling_summary,
    render_dataset_profiling_report,
)


def _profiled_result() -> GenerationResult:
    return GenerationResult(
        request=GenerationRequest(
            prompt="profile",
            max_new_tokens=16,
            sample_id="sample-1",
        ),
        output_text="answer",
        status=RunStatus.SUCCESS,
        forward_profiles=[
            ForwardProfile(
                forward_index=0,
                phase="prefill",
                wall_clock_seconds=0.1,
                compute_tflops=1.0,
                input_tokens=8,
                kv_cache_tokens=0,
                attention_tokens=8,
                uses_kv_cache=False,
                stores_kv=True,
            ),
            ForwardProfile(
                forward_index=1,
                phase="denoise",
                wall_clock_seconds=0.2,
                compute_tflops=2.0,
                accepted_tokens=4,
                input_tokens=16,
                kv_cache_tokens=8,
                attention_tokens=24,
                uses_kv_cache=True,
                stores_kv=False,
            ),
        ],
        extra={
            "stage_profiles": [
                {
                    "stage_index": 0,
                    "stage": "prefill",
                    "wall_clock_seconds": 0.1,
                    "compute_flops": 1_000_000_000_000,
                    "compute_tflops": 1.0,
                },
                {
                    "stage_index": 1,
                    "stage": "token_selection",
                    "wall_clock_seconds": 0.05,
                    "compute_flops": 10_000_000,
                    "compute_tflops": 0.00001,
                },
            ]
        },
    )


def test_profiling_summary_uses_forward_profiles_without_trace():
    sample = Sample("sample-1", "profile", "answer")
    summary, rows = build_dataset_profiling_summary(
        "gsm8k",
        [(sample, _profiled_result())],
        model_name="dreamreasoner",
        config_name="p2",
    )

    assert summary["step_count"] == 2
    assert summary["compute_tflops"] == 3.0
    assert summary["accepted_tokens"] == 4
    assert summary["phase_contribution"]["prefill"]["steps"] == 1
    assert summary["stage_contribution"]["prefill"]["calls"] == 1
    assert rows[-1].cumulative_compute_tflops == 3.0


def test_profiling_visualization_writes_only_png(tmp_path):
    sample = Sample("sample-1", "profile", "answer")
    written = render_dataset_profiling_report(
        "gsm8k",
        [(sample, _profiled_result())],
        tmp_path,
        model_name="dreamreasoner",
        config_name="p2",
    )

    assert set(written) == {"step_profiling_plot", "stage_profiling_plot"}
    assert (tmp_path / "dataset_step_profiling.png").is_file()
    assert (tmp_path / "dataset_stage_profiling.png").is_file()
