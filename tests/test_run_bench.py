from __future__ import annotations

import pytest

import run_bench


def test_default_dry_run_selects_every_matrix_model(capsys):
    assert run_bench.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Models: qwen3_4b, illada, dreamreasoner, w1, diffusiongemma, gemma4_26b_a4b" in output
    assert "venv_scripts\\illada.py run" in output or "venv_scripts/illada.py run" in output
    assert "venv_scripts\\diffusiongemma.py run" in output or "venv_scripts/diffusiongemma.py run" in output
    assert "venv_scripts\\gemma4_26b_a4b.py run" in output or "venv_scripts/gemma4_26b_a4b.py run" in output


def test_model_flag_filters_to_one_model(capsys):
    assert run_bench.main(["--dry-run", "-m", "illada"]) == 0
    output = capsys.readouterr().out
    assert "Models: illada" in output
    assert "illada.py run" in output
    assert "dreamreasoner" not in output


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
