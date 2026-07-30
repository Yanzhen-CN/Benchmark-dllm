from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from dllm_bench.cli import main


def _summary(model: str, config: str, *, seconds: float, energy: float, q: float) -> dict:
    return {
        "model_name": model,
        "config_name": config,
        "dataset_name": "gsm8k",
        "q": q,
        "time_per_sample": seconds,
        "energy_per_sample": energy,
        "timing_source": "measured",
        "run_metadata": {"measurement_protocol": "gpu-synced-v4"},
        "scoring_metadata": {
            "sample_set_hash": "sample-hash",
            "dataset_revision": "dataset-hash",
            "prompt_protocol_revision": "prompt-hash",
            "generation_protocol_revision": "generation-hash",
            "expected_sample_count": 100,
        },
    }


def _write_summary(output_root: Path, summary: dict) -> None:
    run = f"{summary['model_name']}_{summary['config_name']}"
    path = output_root / "score_output" / run / summary["dataset_name"] / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary), encoding="utf-8")


def test_pairwise_cli_writes_each_model_in_an_independent_directory(tmp_path):
    output_root = tmp_path / "output"
    _write_summary(
        output_root,
        _summary("qwen3_8b", "ar-baseline", seconds=10, energy=100, q=0.5),
    )
    _write_summary(
        output_root,
        _summary("illada", "best", seconds=5, energy=80, q=0.6),
    )
    _write_summary(
        output_root,
        _summary("dreamreasoner", "fast", seconds=4, energy=70, q=0.55),
    )

    result = CliRunner().invoke(
        main,
        [
            "pairwise-report",
            "--output-root",
            str(output_root),
            "--model",
            "illada",
            "--model",
            "dreamreasoner",
            "--base-model",
            "qwen3_8b",
            "--base-config",
            "ar-baseline",
            "--beta",
            "40",
            "--gamma",
            "25",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    pair_dirs = sorted((output_root / "conversion_output").glob("*__relative_to__*"))
    assert len(pair_dirs) == 2
    for pair_dir in pair_dirs:
        leaf = pair_dir / "gsm8k" / "beta-40_gamma-25"
        assert (leaf / "pairwise.txt").exists()
        assert (leaf / "metadata.json").exists()
        assert (leaf / "pairwise.png").exists()
