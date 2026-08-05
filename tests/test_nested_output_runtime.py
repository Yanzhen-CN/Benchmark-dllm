from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from dllm_bench.cli import _valid_summary_paths, main
from dllm_bench.runner.output_layout import (
    resolve_model_output_dir,
    resolve_score_output_dir,
)


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_generate_writes_model_config_dataset_under_stage_root(tmp_path):
    generation_root = tmp_path / "output" / "model_output"
    result = CliRunner().invoke(
        main,
        [
            "generate",
            "--model-config",
            str(CONFIGS_DIR / "models" / "mock.yaml"),
            "--variant",
            "default",
            "--dataset-config",
            str(CONFIGS_DIR / "datasets" / "gsm8k.yaml"),
            "--demo",
            "--n-samples",
            "1",
            "--max-new-tokens",
            "16",
            "--output-root",
            str(generation_root),
        ],
    )

    assert result.exit_code == 0, result.output
    expected = generation_root / "mock" / "default" / "gsm8k"
    assert (expected / "_meta.json").is_file()
    assert not (generation_root / "mock_default").exists()


def test_all_stage_readers_fall_back_to_legacy_flat_run_directory(tmp_path):
    generation_root = tmp_path / "output" / "model_output"
    score_root = tmp_path / "output" / "score_output"
    legacy_generation = generation_root / "illada_entropy_eb05" / "gsm8k"
    legacy_score = score_root / "illada_entropy_eb05" / "gsm8k"
    legacy_generation.mkdir(parents=True)
    legacy_score.mkdir(parents=True)

    assert resolve_model_output_dir(
        generation_root, "illada_entropy", "eb05", "gsm8k"
    ) == legacy_generation
    assert resolve_score_output_dir(
        generation_root, "illada_entropy", "eb05", "gsm8k"
    ) == legacy_score


def test_report_validation_follows_nested_summary_metadata(tmp_path):
    output_root = tmp_path / "output"
    model_dir = output_root / "model_output" / "illada_entropy" / "eb05" / "gsm8k"
    score_dir = output_root / "score_output" / "illada_entropy" / "eb05" / "gsm8k"
    model_dir.mkdir(parents=True)
    score_dir.mkdir(parents=True)
    (model_dir / "_meta.json").write_text(
        json.dumps({"test_valid": False, "failure_stage": "generation"}),
        encoding="utf-8",
    )
    summary = score_dir / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "model_name": "illada_entropy",
                "config_name": "eb05",
                "dataset_name": "gsm8k",
            }
        ),
        encoding="utf-8",
    )

    assert _valid_summary_paths([str(summary)]) == []
