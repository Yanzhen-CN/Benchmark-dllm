"""End-to-end CLI test: generate -> score -> visualize -> report through the
mock adapter, using Click's test runner (no subprocess needed)."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

from click.testing import CliRunner

import dllm_bench.cli as cli_module
from dllm_bench.cli import main
from dllm_bench.datasets.base import Sample
from dllm_bench.datasets.gsm8k import GSM8KDataset
from dllm_bench.datasets.mbpp import MBPPDataset, MbppSample

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
    assert len(list(viz_out.glob("*_trace.gif"))) == 2
    trace_summary = json.loads(
        (viz_out / "dataset_trace_summary.json").read_text(encoding="utf-8")
    )
    assert trace_summary["trace_samples"] == 3

    report_result = _run(runner, [
        "report", "--output-root", str(output_root), "--dataset", "gsm8k",
    ])
    assert "gsm8k" in report_result.output
    assert "mock" in report_result.output
    assert (output_root / "report" / "gsm8k" / "quality_tps.png").exists()
    assert (output_root / "report" / "gsm8k" / "quality_sps.png").exists()


def test_generate_uses_sample_progress_bar_on_interactive_terminal(
    tmp_path, monkeypatch
):
    class InteractiveStream(io.StringIO):
        def isatty(self):
            return True

    terminal = InteractiveStream()
    monkeypatch.setattr(
        cli_module.click,
        "get_text_stream",
        lambda name: terminal if name == "stdout" else io.StringIO(),
    )
    runner = CliRunner()

    _run(runner, [
        "generate",
        "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
        "--variant", "default",
        "--dataset-config", str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
        "--demo", "--n-samples", "2", "--max-new-tokens", "16",
        "--output-root", str(tmp_path / "output"),
    ])

    rendered = terminal.getvalue()
    assert "[default] gsm8k" in rendered
    assert "2/2" in rendered
    assert "gsm8k-demo-1" in rendered


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


def test_matrix_propagates_warmup_oom_and_stops_remaining_jobs(tmp_path, monkeypatch):
    """A job-level OOM is not swallowed or converted into CPU offload."""
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
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.16 GiB")
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

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "CUDA out of memory" in str(result.exception)
    assert calls == [str(gsm8k_config)]


def test_matrix_variants_option_filters_sampling_profile(tmp_path, monkeypatch):
    runner = CliRunner()
    model_config = CONFIGS_DIR / "models" / "mock.yaml"
    dataset_config = CONFIGS_DIR / "datasets" / "gsm8k.yaml"
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "seed: 42\n"
        "models:\n"
        f"  - name: mock\n    config: {model_config}\n"
        "    variants: [default, fast]\n"
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
