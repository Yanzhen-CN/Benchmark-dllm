from __future__ import annotations

import local_pipeline
import run_model


def test_run_model_forces_generate_and_real_data(monkeypatch):
    captured = []
    monkeypatch.setattr(run_model.run_bench, "main", lambda argv: captured.extend(argv) or 0)

    assert run_model.main(["--dry-run", "-m", "illada"]) == 0
    assert captured[-2:] == ["--stage", "generate"]
    assert "--real-data" in captured
    assert "--measure-compute" in captured
    assert "--require-all-metrics" in captured


def test_run_model_allows_explicit_compute_opt_out(monkeypatch):
    captured = []
    monkeypatch.setattr(run_model.run_bench, "main", lambda argv: captured.extend(argv) or 0)

    assert run_model.main(["--dry-run", "-m", "illada", "--no-measure-compute"]) == 0
    assert "--no-measure-compute" in captured
    assert "--measure-compute" not in captured


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


def test_local_visualization_also_builds_report(capsys):
    assert local_pipeline.main("visualize", ["--dry-run", "-m", "illada"]) == 0
    output = capsys.readouterr().out
    assert "--stage visualize" in output
    assert "dllm_bench.cli report --output-root output" in output
