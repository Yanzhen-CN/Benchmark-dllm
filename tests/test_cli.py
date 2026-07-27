"""End-to-end CLI test: generate -> score -> visualize -> report through the
mock adapter, using Click's test runner (no subprocess needed)."""

from __future__ import annotations

import random
from pathlib import Path

from click.testing import CliRunner

from dllm_bench.cli import main
from dllm_bench.datasets.base import Sample
from dllm_bench.datasets.gsm8k import GSM8KDataset
from dllm_bench.datasets.mbpp import MBPPDataset, MbppSample

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _run(runner, args):
    result = runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def test_generate_score_visualize_report_pipeline(tmp_path):
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

    report_result = _run(runner, [
        "report", "--output-root", str(output_root), "--dataset", "gsm8k",
    ])
    assert "gsm8k" in report_result.output
    assert "mock" in report_result.output
    assert (output_root / "report" / "gsm8k" / "quality_tps.png").exists()


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
