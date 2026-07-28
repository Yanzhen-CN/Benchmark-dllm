from __future__ import annotations

import pytest

import run_bench


def test_default_dry_run_selects_every_matrix_model(capsys):
    assert run_bench.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "illada, illada_optimized, dreamreasoner" in output
    assert "venv_scripts\\illada.py run" in output or "venv_scripts/illada.py run" in output
    assert "venv_scripts\\diffusiongemma.py run" in output or "venv_scripts/diffusiongemma.py run" in output
    assert "venv_scripts\\gemma4_26b_a4b.py run" in output or "venv_scripts/gemma4_26b_a4b.py run" in output
    assert "venv_scripts\\qwen3_8b.py run" in output or "venv_scripts/qwen3_8b.py run" in output


def test_model_flag_filters_to_one_model(capsys):
    assert run_bench.main(["--dry-run", "-m", "illada"]) == 0
    output = capsys.readouterr().out
    assert "Models: illada" in output
    assert "illada.py run" in output
    assert "dreamreasoner" not in output


def test_optimized_model_is_a_separate_public_model(capsys):
    assert run_bench.main(["--dry-run", "-m", "illada_optimized"]) == 0
    output = capsys.readouterr().out
    assert "Models: illada_optimized" in output
    assert "illada_optimized.py run" in output


def test_model_flag_accepts_comma_separated_names(capsys):
    assert run_bench.main(["--dry-run", "-m", "illada,qwen3_4b"]) == 0
    output = capsys.readouterr().out
    assert "illada.py run" in output
    assert "qwen3_4b.py run" in output


def test_unknown_model_has_clear_error():
    with pytest.raises(SystemExit, match="unknown model"):
        run_bench.main(["--dry-run", "-m", "missing"])


def test_dataset_flag_accepts_multiple_names(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured["models"] = list(model_names)
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "illada",
        "-d", "ruler", "hellobench",
        "--stage", "generate",
    ]) == 0
    assert captured["models"] == ["illada"]
    assert captured["env_updates"]["DATASETS"] == "ruler,hellobench"


def test_variant_flag_is_forwarded_to_the_selected_model(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured["models"] = list(model_names)
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "illada_optimized", "-v", "fast", "--stage", "generate"
    ]) == 0
    assert captured["models"] == ["illada_optimized"]
    assert captured["env_updates"]["MATRIX_VARIANTS"] == "fast"


def test_hellobench_length_and_total_count_are_forwarded(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "illada", "-d", "hellobench", "--real-data",
        "--hellobench-length", "2k", "--n-samples", "3",
    ]) == 0
    assert captured["env_updates"]["HELLOBENCH_LENGTHS"] == "2k"
    assert captured["env_updates"]["N_SAMPLES"] == "3"


def test_variant_and_variants_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        run_bench.main([
            "--dry-run", "-m", "illada", "--variant", "fast",
            "--variants", "best,fast",
        ])


def test_model_and_dataset_flags_accept_space_separated_names(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured["models"] = list(model_names)
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "illada", "dreamreasoner",
        "-d", "ruler", "hellobench",
        "--stage", "generate",
    ]) == 0
    assert captured["models"] == ["illada", "dreamreasoner"]
    assert captured["env_updates"]["DATASETS"] == "ruler,hellobench"


def test_diffusiongemma_is_the_public_name():
    assert "diffusiongemma" in run_bench.matrix_model_names(run_bench.DEFAULT_MATRIX)
    assert "dg" not in run_bench.matrix_model_names(run_bench.DEFAULT_MATRIX)


def test_same_scale_gemma_ar_reference_is_public():
    assert "gemma4_26b_a4b" in run_bench.matrix_model_names(run_bench.DEFAULT_MATRIX)
