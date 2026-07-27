from __future__ import annotations

import pytest

import run_bench


def test_default_dry_run_selects_every_matrix_model(capsys):
    assert run_bench.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Models: qwen3_4b, illada, dreamreasoner, w1, diffusiongemma" in output
    assert "venv_scripts\\illada.py run" in output or "venv_scripts/illada.py run" in output
    assert "venv_scripts\\diffusiongemma.py run" in output or "venv_scripts/diffusiongemma.py run" in output


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


def test_diffusiongemma_is_the_public_name():
    assert "diffusiongemma" in run_bench.matrix_model_names(run_bench.DEFAULT_MATRIX)
    assert "dg" not in run_bench.matrix_model_names(run_bench.DEFAULT_MATRIX)
