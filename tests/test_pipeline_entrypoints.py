from __future__ import annotations

import local_pipeline
import run_conversion
import run_model


def test_run_model_forces_generate_and_real_data(monkeypatch):
    captured = []
    monkeypatch.setattr(run_model.run_bench, "main", lambda argv: captured.extend(argv) or 0)

    assert run_model.main(["--dry-run", "-m", "illada"]) == 0
    assert captured[-2:] == ["--stage", "generate"]
    assert "--real-data" in captured
    assert "--no-measure-compute" in captured
    assert "--measure-compute" not in captured
    assert "--require-all-metrics" in captured


def test_run_model_allows_explicit_compute_opt_in(monkeypatch):
    captured = []
    monkeypatch.setattr(run_model.run_bench, "main", lambda argv: captured.extend(argv) or 0)

    assert run_model.main(["--dry-run", "-m", "illada", "--measure-compute"]) == 0
    assert "--measure-compute" in captured
    assert "--no-measure-compute" not in captured


def test_run_model_allows_explicit_missing_metrics_opt_out(monkeypatch):
    captured = []
    monkeypatch.setattr(run_model.run_bench, "main", lambda argv: captured.extend(argv) or 0)

    assert run_model.main(["--dry-run", "-m", "illada", "--allow-missing-metrics"]) == 0
    assert "--allow-missing-metrics" in captured
    assert "--require-all-metrics" not in captured


def test_local_score_dispatches_every_selected_model_without_model_venvs(capsys):
    assert local_pipeline.main("score", ["--dry-run", "-m", "illada,qwen3_4b"]) == 0
    output = capsys.readouterr().out
    assert "--stage score" in output
    assert "--model illada" in output
    assert "--model qwen3_4b" in output
    assert "venv_scripts" not in output
    assert "--resume" in output


def test_local_score_can_force_rescoring(capsys):
    assert local_pipeline.main(
        "score", ["--dry-run", "-m", "dreamreasoner", "--no-resume"]
    ) == 0
    output = capsys.readouterr().out
    assert "--stage score" in output
    assert "--no-resume" in output


def test_local_score_forwards_sudoku_group_to_shared_matrix_filter(capsys):
    assert local_pipeline.main(
        "score", ["--dry-run", "-m", "qwen3_8b", "-d", "sudoku"]
    ) == 0
    output = capsys.readouterr().out
    assert "--dataset sudoku" in output


def test_local_score_forwards_length_and_variant_sweeps(capsys):
    assert local_pipeline.main(
        "score",
        [
            "--dry-run",
            "-m",
            "illada_vargen",
            "-d",
            "gsm8k",
            "-max",
            "1024",
            "2048",
            "-v",
            "p1",
            "p2",
            "p4",
            "p8",
        ],
    ) == 0
    output = capsys.readouterr().out
    assert "--max-new-tokens 1024 --max-new-tokens 2048" in output
    assert "--variants p1,p2,p4,p8" in output


def test_local_visualization_also_builds_report(capsys):
    assert local_pipeline.main("visualize", ["--dry-run", "-m", "illada"]) == 0
    output = capsys.readouterr().out
    assert "--stage visualize" in output
    assert "dllm_bench.cli report --output-root output --model illada" in output


def test_local_visualization_forwards_report_dataset_filter(capsys):
    assert local_pipeline.main(
        "visualize", ["--dry-run", "-m", "qwen3_8b", "-d", "sudoku"]
    ) == 0
    output = capsys.readouterr().out
    assert "report --output-root output --model qwen3_8b --dataset sudoku" in output


def test_local_visualization_forwards_curated_sample_ids(capsys):
    sample_ids = "gsm8k-test-0177,mbpp-sanitized-0131"
    assert local_pipeline.main(
        "visualize",
        [
            "--dry-run",
            "-m",
            "diffusiongemma",
            "--sample-ids",
            sample_ids,
        ],
    ) == 0
    output = capsys.readouterr().out
    assert f"--sample-ids {sample_ids}" in output


def test_conversion_entrypoint_accepts_parallel_model_selector(capsys):
    assert run_conversion.main(
        [
            "--dry-run",
            "-m",
            "illada",
            "dreamreasoner",
            "--base-model",
            "qwen3_8b",
            "--base-config",
            "ar-baseline",
            "--beta",
            "40",
            "--gamma",
            "25",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "--model illada --model dreamreasoner" in output
    assert "--base-model qwen3_8b" in output
