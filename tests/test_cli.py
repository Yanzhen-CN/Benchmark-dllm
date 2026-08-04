"""End-to-end CLI test: generate -> score -> visualize -> report through the
mock adapter, using Click's test runner (no subprocess needed)."""

from __future__ import annotations

import io
import json
import random
import re
import time
from pathlib import Path

from click.testing import CliRunner

import dllm_bench.cli as cli_module
from dllm_bench.cli import main
from dllm_bench.datasets.base import Sample
from dllm_bench.datasets.gsm8k import GSM8KDataset
from dllm_bench.datasets.mbpp import MBPPDataset, MbppSample
from dllm_bench.interfaces import GenerationResult, RunStatus
from dllm_bench.models.mock import MockDiffusionAdapter

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _run(runner, args):
    result = runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def test_generate_score_visualize_report_pipeline(tmp_path, monkeypatch):
    runner = CliRunner()
    output_root = tmp_path / "output"
    model_config = str(CONFIGS_DIR / "models" / "mock.yaml")
    dataset_config = str(CONFIGS_DIR / "datasets" / "gsm8k.yaml")

    generate_result = _run(runner, [
        "generate",
        "--model-config", model_config, "--variant", "default",
        "--dataset-config", dataset_config,
        "--demo", "--n-samples", "3", "--max-new-tokens", "16",
        "--output-root", str(output_root),
    ])
    assert "generated=3 skipped=0" in generate_result.output
    assert "loading model into runtime device (outside sample timing)" in generate_result.output
    assert "[default] [1/3] gsm8k-demo-0: generating" in generate_result.output

    model_out = output_root / "model_output" / "mock_default" / "gsm8k"
    assert (model_out / "_meta.json").exists()
    assert len(list(model_out.glob("gsm8k-demo-*.json"))) == 3

    # re-running generate should skip everything (resume).
    resume_result = _run(runner, [
        "generate",
        "--model-config", model_config, "--variant", "default",
        "--dataset-config", dataset_config,
        "--demo", "--n-samples", "3", "--max-new-tokens", "16",
        "--output-root", str(output_root),
    ])
    assert "generated=0 skipped=3" in resume_result.output
    assert "all sample outputs already exist; model load skipped" in resume_result.output

    # Local stages derive output names from YAML and must never construct a
    # model adapter or touch model dependencies/weights.
    monkeypatch.setattr(
        cli_module,
        "build_model_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local stage constructed a model adapter")
        ),
    )

    score_result = _run(runner, [
        "score",
        "--model-config", model_config, "--variant", "default",
        "--dataset-config", dataset_config,
        "--demo", "--n-samples", "3",
        "--output-root", str(output_root),
    ])
    assert "scored=3" in score_result.output

    score_out = output_root / "score_output" / "mock_default" / "gsm8k"
    assert (score_out / "summary.json").exists()

    visualize_result = _run(runner, [
        "visualize",
        "--model-config", model_config, "--variant", "default",
        "--dataset-config", dataset_config,
        "--demo", "--n-samples", "3",
        "--output-root", str(output_root),
        "--n-representative", "2",
    ])
    assert "rendered 2 sample(s)" in visualize_result.output

    viz_out = output_root / "visualization_output" / "mock_default" / "gsm8k"
    assert len(list(viz_out.glob("*_all_updates.png"))) == 2
    assert not list(viz_out.glob("*_trace.gif"))
    trace_summary = json.loads(
        (viz_out / "dataset_trace_summary.json").read_text(encoding="utf-8")
    )
    assert trace_summary["trace_samples"] == 3
    assert trace_summary["model"] == "mock"
    assert trace_summary["config"] == "default"
    assert not (viz_out / "dataset_finalization_map.png").exists()
    assert not (viz_out / "dataset_commit_order_tau.png").exists()
    assert (viz_out / "dataset_tpf_tps.txt").exists()

    report_result = _run(runner, [
        "report", "--output-root", str(output_root), "--dataset", "gsm8k",
    ])
    assert "gsm8k" in report_result.output
    assert "mock" in report_result.output
    assert not (output_root / "report" / "gsm8k" / "quality_tps.png").exists()
    assert not (output_root / "report" / "gsm8k" / "quality_seconds_per_sample.png").exists()
    assert not (output_root / "report" / "gsm8k" / "quality_energy_per_sample.png").exists()
    assert (output_root / "report" / "raw_results.csv").exists()
    assert (output_root / "report" / "gsm8k" / "trace_metrics.csv").exists()
    assert not (output_root / "report" / "gsm8k" / "task4_tpf_vs_tps.png").exists()
    assert not (
        output_root / "report" / "gsm8k" / "task4_parallelism_signature.png"
    ).exists()


def test_generate_reports_durable_sample_counts_without_terminal_controls(tmp_path):
    runner = CliRunner()

    result = _run(runner, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "2", "--max-new-tokens", "16",
        "--output-root", str(tmp_path / "output"),
    ])

    assert "\r" not in result.output
    assert "[default] [1/2] gsm8k-demo-0: generating" in result.output
    assert "[default] [1/2] gsm8k-demo-0: success" in result.output
    assert "[default] [2/2] gsm8k-demo-1: generating" in result.output
    assert "[default] [2/2] gsm8k-demo-1: success" in result.output


def test_long_task_warmup_uses_short_prompt_without_formal_context_budget(
    tmp_path, monkeypatch
):
    captured = []
    real_warmup = MockDiffusionAdapter.warmup_generation

    def capture_warmup(self, request):
        captured.append(request)
        return real_warmup(self, request)

    monkeypatch.setattr(
        MockDiffusionAdapter, "warmup_generation", capture_warmup
    )
    runner = CliRunner()
    _run(runner, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "ruler.yaml"),
        "--demo", "--n-samples", "1", "--max-new-tokens", "64",
        "--output-root", str(tmp_path / "output"),
    ])

    assert len(captured) == 1
    assert captured[0].prompt == "Warm up."
    assert captured[0].max_new_tokens == 8
    assert captured[0].config == {}


def test_warmup_honors_adapter_minimum_valid_generation_length(
    tmp_path, monkeypatch
):
    captured = []
    real_warmup = MockDiffusionAdapter.warmup_generation

    def capture_warmup(self, request):
        captured.append(request)
        return real_warmup(self, request)

    monkeypatch.setattr(MockDiffusionAdapter, "warmup_new_tokens", 32, raising=False)
    monkeypatch.setattr(MockDiffusionAdapter, "warmup_generation", capture_warmup)
    runner = CliRunner()
    _run(runner, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "1", "--max-new-tokens", "64",
        "--output-root", str(tmp_path / "output"),
    ])

    assert len(captured) == 1
    assert captured[0].max_new_tokens == 32


def test_dataset_oom_stops_later_samples_but_other_variant_is_still_attempted(
    tmp_path, monkeypatch
):
    calls = []

    def oom_generate(self, request):
        calls.append((self.config_name, request.sample_id))
        return GenerationResult(
            request=request,
            output_text="",
            status=RunStatus.OOM,
            error_message="CUDA out of memory",
        )

    monkeypatch.setattr(MockDiffusionAdapter, "generate", oom_generate)
    runner = CliRunner()
    output_root = tmp_path / "output"
    result = runner.invoke(main, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "ruler.yaml"),
        "--demo", "--n-samples", "3", "--max-new-tokens", "64",
        "--output-root", str(output_root),
    ], catch_exceptions=True)

    assert result.exit_code != 0
    assert isinstance(result.exception, cli_module.OOMInvalidTestError)
    assert calls == [
        ("default", "ruler-demo-0"),
        ("fast", "ruler-demo-0"),
    ]
    assert (
        output_root / "model_output" / "mock_default" / "ruler" / "_meta.json"
    ).exists()
    assert (
        output_root
        / "model_output"
        / "mock_default"
        / "ruler"
        / "oom_info.json"
    ).exists()
    assert (
        output_root / "model_output" / "mock_fast" / "ruler" / "oom_info.json"
    ).exists()


def test_warmup_oom_writes_invalid_info_before_raising(tmp_path, monkeypatch):
    def oom_warmup(self, request):
        raise RuntimeError("CUDA out of memory during warmup")

    monkeypatch.setattr(MockDiffusionAdapter, "warmup_generation", oom_warmup)
    runner = CliRunner()
    output_root = tmp_path / "output"
    result = runner.invoke(main, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "2", "--max-new-tokens", "16",
        "--output-root", str(output_root),
    ], catch_exceptions=True)

    assert result.exit_code != 0
    assert isinstance(result.exception, cli_module.OOMInvalidTestError)
    out_dir = output_root / "model_output" / "mock_default" / "gsm8k"
    oom_info = json.loads((out_dir / "oom_info.json").read_text(encoding="utf-8"))
    assert oom_info["test_valid"] is False
    assert oom_info["failure_stage"] == "warmup"
    assert oom_info["attempted_samples"] == 0
    assert oom_info["selected_samples"] == 2
    assert oom_info["error_message"] == "CUDA out of memory during warmup"
    assert "devices" in oom_info["gpu"]


def test_generate_sweeps_every_variant_by_default(tmp_path):
    """mock.yaml declares `default` and `fast` — omitting --variant/--variants
    must run both, in one invocation, without reloading anything (the atomic
    unit of testing is the model, not model+variant)."""
    runner = CliRunner()
    output_root = tmp_path / "output"
    model_config = str(CONFIGS_DIR / "models" / "mock.yaml")
    dataset_config = str(CONFIGS_DIR / "datasets" / "gsm8k.yaml")

    result = _run(runner, [
        "generate",
        "--model-config", model_config,
        "--dataset-config", dataset_config,
        "--demo", "--n-samples", "2", "--max-new-tokens", "16",
        "--output-root", str(output_root),
    ])
    assert "['default', 'fast']" in result.output
    assert (output_root / "model_output" / "mock_default" / "gsm8k" / "_meta.json").exists()
    assert (output_root / "model_output" / "mock_fast" / "gsm8k" / "_meta.json").exists()


def test_generate_variants_option_selects_an_explicit_subset(tmp_path):
    runner = CliRunner()
    output_root = tmp_path / "output"
    model_config = str(CONFIGS_DIR / "models" / "mock.yaml")
    dataset_config = str(CONFIGS_DIR / "datasets" / "gsm8k.yaml")

    _run(runner, [
        "generate",
        "--model-config", model_config, "--variants", "fast",
        "--dataset-config", dataset_config,
        "--demo", "--n-samples", "2", "--max-new-tokens", "16",
        "--output-root", str(output_root),
    ])
    assert (output_root / "model_output" / "mock_fast" / "gsm8k" / "_meta.json").exists()
    assert not (output_root / "model_output" / "mock_default").exists()


def test_variant_and_variants_together_is_a_usage_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--variant", "default", "--variants", "default,fast",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "2", "--max-new-tokens", "16",
        "--output-root", str(tmp_path / "output"),
    ])
    assert result.exit_code != 0


def test_generate_and_score_real_samples_use_the_same_seeded_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / ".data"))
    available = [
        Sample(sample_id=f"official-{i}", prompt=f"What is {i}+1?", reference=float(i + 1))
        for i in range(6)
    ]
    monkeypatch.setattr(GSM8KDataset, "load_samples", lambda self, n=None: list(available))

    runner = CliRunner()
    output_root = tmp_path / "output"
    common = [
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"), "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--no-demo", "--n-samples", "3", "--seed", "42",
        "--output-root", str(output_root),
    ]

    generate_result = _run(runner, ["generate", *common, "--max-new-tokens", "16"])
    assert "generated=3" in generate_result.output

    expected = list(available)
    random.Random(42).shuffle(expected)
    model_out = output_root / "model_output" / "mock_default" / "gsm8k"
    assert {path.stem for path in model_out.glob("official-*.json")} == {
        sample.sample_id for sample in expected[:3]
    }

    score_result = _run(runner, ["score", *common])
    assert "scored=3" in score_result.output
    assert "WARNING" not in score_result.output


def test_no_demo_auto_prepares_missing_real_dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / ".data"))
    monkeypatch.setattr(
        MBPPDataset,
        "load_samples",
        lambda self, n=None: [
            Sample(
                sample_id="mbpp-official-11",
                prompt="Write a function add(a, b).",
                reference=MbppSample(["assert add(1, 2) == 3"]),
            )
        ],
    )
    runner = CliRunner()
    result = runner.invoke(main, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"), "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "mbpp.yaml"),
        "--no-demo", "--n-samples", "1", "--max-new-tokens", "16",
        "--output-root", str(tmp_path / "output"),
    ])
    assert result.exit_code == 0, result.output
    assert "Prepared 1 mbpp samples" in result.output


def test_matrix_reports_oom_invalid_job_and_continues_later_jobs(tmp_path, monkeypatch):
    """One invalid model×dataset row must not terminate the remaining matrix."""
    runner = CliRunner()
    output_root = tmp_path / "output"
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    gsm8k_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    mbpp_config = CONFIGS_DIR / "datasets" / "mbpp.yaml"

    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n    variants: [default]\n"
        "datasets:\n"
        f"  - config: {gsm8k_config}\n    max_new_tokens: 16\n"
        f"  - config: {mbpp_config}\n    max_new_tokens: 16\n"
    )

    calls = []
    real_generate = cli_module.generate.callback

    def fake_generate(**kwargs):
        calls.append(kwargs["dataset_config"])
        if kwargs["dataset_config"] == str(gsm8k_config):
            raise cli_module.OOMInvalidTestError(
                "OOM invalidated the complete model×dataset test"
            )
        return real_generate(**kwargs)

    monkeypatch.setattr(cli_module, "generate", fake_generate)

    result = runner.invoke(
        main,
        [
            "matrix",
            "--experiment-config", str(experiment_config),
            "--model", "mock",
            "--stage", "generate",
            "--demo", "--n-samples", "3",
            "--output-root", str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "skipping this test and continuing" in result.output
    assert calls == [str(gsm8k_config), str(mbpp_config)]


def test_matrix_reports_missing_score_job_and_continues_later_jobs(
    tmp_path, monkeypatch
):
    runner = CliRunner()
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    gsm8k_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    mbpp_config = CONFIGS_DIR / "datasets" / "mbpp.yaml"
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n    variants: [default]\n"
        "datasets:\n"
        f"  - config: {gsm8k_config}\n    max_new_tokens: 16\n"
        f"  - config: {mbpp_config}\n    max_new_tokens: 16\n"
    )
    calls = []

    def fake_score(**kwargs):
        calls.append(kwargs["dataset_config"])
        if kwargs["dataset_config"] == str(gsm8k_config):
            raise FileNotFoundError("generation output is missing")

    monkeypatch.setattr(cli_module, "score", fake_score)

    result = runner.invoke(
        main,
        [
            "matrix",
            "--experiment-config", str(experiment_config),
            "--model", "mock",
            "--stage", "score",
            "--demo",
            "--output-root", str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "skipping this test and continuing" in result.output
    assert "0 invalid test(s), 1 incomplete test(s) excluded" in result.output
    assert calls == [str(gsm8k_config), str(mbpp_config)]


def test_matrix_reuses_one_adapter_across_dataset_jobs(tmp_path, monkeypatch):
    runner = CliRunner()
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    gsm8k_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    mbpp_config = CONFIGS_DIR / "datasets" / "mbpp.yaml"
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n    variants: [default]\n"
        "datasets:\n"
        f"  - config: {gsm8k_config}\n    max_new_tokens: 4\n"
        f"  - config: {mbpp_config}\n    max_new_tokens: 4\n"
    )
    real_build = cli_module.build_model_adapter
    built = []

    def tracked_build(*args, **kwargs):
        adapter = real_build(*args, **kwargs)
        built.append(adapter)
        return adapter

    monkeypatch.setattr(cli_module, "build_model_adapter", tracked_build)
    result = runner.invoke(
        main,
        [
            "matrix", "--experiment-config", str(experiment_config),
            "--model", "mock", "--stage", "generate", "--demo",
            "--n-samples", "1", "--output-root", str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(built) == 1


def test_matrix_variants_option_can_select_model_variant_outside_matrix_defaults(tmp_path, monkeypatch):
    runner = CliRunner()
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    dataset_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n"
        "    variants: [default]\n"
        "datasets:\n"
        f"  - config: {dataset_config}\n    max_new_tokens: 16\n"
    )
    captured = []

    def fake_generate(**kwargs):
        captured.append(kwargs["variants"])

    monkeypatch.setattr(cli_module, "generate", fake_generate)
    result = runner.invoke(
        main,
        [
            "matrix", "--experiment-config", str(experiment_config),
            "--model", "mock", "--variants", "fast",
            "--stage", "generate", "--demo",
            "--output-root", str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == ["fast"]


def test_matrix_output_length_override_has_highest_priority(tmp_path, monkeypatch):
    runner = CliRunner()
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    dataset_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n"
        "    variants: [default]\n"
        "datasets:\n"
        f"  - config: {dataset_config}\n    max_new_tokens: 16\n"
    )
    captured = []

    def fake_generate(**kwargs):
        captured.append(
            (kwargs["max_new_tokens"], kwargs["force_max_new_tokens"])
        )

    monkeypatch.setattr(cli_module, "generate", fake_generate)
    result = runner.invoke(
        main,
        [
            "matrix", "--experiment-config", str(experiment_config),
            "--model", "mock", "--stage", "generate", "--demo",
            "--max-new-tokens", "512",
            "--output-root", str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [(512, True)]


def test_matrix_multiple_output_lengths_share_process_and_split_output(
    tmp_path, monkeypatch
):
    runner = CliRunner()
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    dataset_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n"
        "    variants: [default]\n"
        "datasets:\n"
        f"  - config: {dataset_config}\n    max_new_tokens: 16\n"
    )
    captured = []

    def fake_generate(**kwargs):
        captured.append(
            (
                kwargs["max_new_tokens"],
                kwargs["force_max_new_tokens"],
                kwargs["output_root"],
                id(kwargs["adapter_cache"]),
            )
        )

    monkeypatch.setattr(cli_module, "generate", fake_generate)
    output_root = tmp_path / "output"
    result = runner.invoke(
        main,
        [
            "matrix", "--experiment-config", str(experiment_config),
            "--model", "mock", "--stage", "generate", "--demo",
            "--max-new-tokens", "1024",
            "--max-new-tokens", "2048",
            "--output-root", str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [(row[0], row[1]) for row in captured] == [
        (1024, True),
        (2048, True),
    ]
    assert [Path(row[2]) for row in captured] == [
        output_root / "len1024",
        output_root / "len2048",
    ]
    assert len({row[3] for row in captured}) == 1


def test_matrix_still_aborts_on_a_non_oom_failure(tmp_path, monkeypatch):
    """Only an OOM-shaped failure is swallowed per-job — anything else is
    a real bug likely to affect every job, so it must still abort loudly."""
    runner = CliRunner()
    output_root = tmp_path / "output"
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    gsm8k_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    mbpp_config = CONFIGS_DIR / "datasets" / "mbpp.yaml"

    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n    variants: [default]\n"
        "datasets:\n"
        f"  - config: {gsm8k_config}\n    max_new_tokens: 16\n"
        f"  - config: {mbpp_config}\n    max_new_tokens: 16\n"
    )

    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs["dataset_config"])
        raise KeyError("some unrelated config bug")

    monkeypatch.setattr(cli_module, "generate", fake_generate)

    result = runner.invoke(
        main,
        [
            "matrix",
            "--experiment-config", str(experiment_config),
            "--model", "mock",
            "--stage", "generate",
            "--demo", "--n-samples", "3",
            "--output-root", str(output_root),
        ],
        catch_exceptions=True,
    )

    assert result.exit_code != 0
    assert calls == [str(gsm8k_config)]  # aborted before the second job


def test_score_before_generate_gives_clear_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "score",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"), "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "2",
        "--output-root", str(tmp_path / "output"),
    ], catch_exceptions=True)
    assert result.exit_code != 0


def test_visualize_before_generate_gives_clear_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "visualize",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"), "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "2",
        "--output-root", str(tmp_path / "output"),
    ])
    assert result.exit_code != 0
    assert "_meta.json" in result.output


def test_report_with_no_matches_errors_clearly(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--output-root", str(tmp_path / "output")])
    assert result.exit_code != 0


def test_visualize_and_report_exclude_oom_invalid_test_even_with_stale_scores(
    tmp_path
):
    runner = CliRunner()
    output_root = tmp_path / "output"
    common = [
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "2",
        "--output-root", str(output_root),
    ]
    _run(runner, ["generate", *common, "--max-new-tokens", "16"])
    _run(runner, ["score", *common])

    model_out = output_root / "model_output" / "mock_default" / "gsm8k"
    meta_path = model_out / "_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(test_valid=False, invalid_reason="oom")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (model_out / "oom_info.json").write_text(
        json.dumps({
            "test_valid": False,
            "invalid_reason": "oom",
            "failure_stage": "generation",
            "sample_ordinal": 1,
            "sample_id": "gsm8k-demo-0",
        }),
        encoding="utf-8",
    )

    visualize_result = runner.invoke(main, ["visualize", *common])
    assert visualize_result.exit_code != 0
    assert isinstance(visualize_result.exception, cli_module.InvalidTestError)

    report_result = runner.invoke(
        main, ["report", "--output-root", str(output_root)]
    )
    assert report_result.exit_code != 0
    assert "excluding OOM-invalid test" in report_result.output
    assert "no valid summary.json files remain" in report_result.output


def test_matrix_yaml_uses_concrete_stage_directory_without_duplication(
    tmp_path, monkeypatch
):
    runner = CliRunner()
    captured = []

    def capture_generate(**kwargs):
        captured.append((kwargs["output_root"], kwargs["profiling_output"]))

    monkeypatch.setattr(cli_module, "generate", capture_generate)
    model_config = (CONFIGS_DIR / "models" / "mock.yaml").as_posix()
    dataset_config = (CONFIGS_DIR / "datasets" / "gsm8k.yaml").as_posix()

    for stage_name, profiling in (
        ("model_output", False),
        ("model_profiling", True),
    ):
        experiment = tmp_path / f"{stage_name}.yaml"
        concrete_output = tmp_path / "output" / stage_name
        experiment.write_text(
            f"base_dir: .\n"
            f"output_root: {concrete_output.as_posix()}\n"
            f"output_stage: {stage_name}\n"
            f"models:\n"
            f"  - config: {model_config}\n"
            f"    variants: [default]\n"
            f"datasets:\n"
            f"  - config: {dataset_config}\n"
            f"    max_new_tokens: 16\n"
            f"seed: 42\n"
            f"profiling_output: {str(profiling).lower()}\n",
            encoding="utf-8",
        )
        arguments = [
            "matrix", "--experiment-config", str(experiment),
            "--model", "mock", "--stage", "generate", "--demo",
            "--n-samples", "1",
        ]
        if profiling:
            arguments.append("--measure-compute")
        result = runner.invoke(main, arguments)
        assert result.exit_code == 0, result.output

    expected_parent = str(tmp_path / "output")
    assert captured == [
        (expected_parent, False),
        (expected_parent, True),
    ]
