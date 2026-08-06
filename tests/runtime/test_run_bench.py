from __future__ import annotations

import subprocess
import threading

import pytest

import run_bench


def test_dispatch_can_install_independent_model_venvs_concurrently(monkeypatch):
    barrier = threading.Barrier(2)
    worker_threads: set[int] = set()

    def fake_run(command, **kwargs):
        worker_threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_bench.subprocess, "run", fake_run)

    run_bench.dispatch_model_scripts(
        ["illada", "dreamreasoner"], action="setup", jobs=2
    )

    assert len(worker_threads) == 2


def test_default_dry_run_selects_every_matrix_model(capsys):
    assert run_bench.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "illada, illada_vargen, dreamreasoner" in output
    assert "venv_scripts\\illada_vargen.py run" in output or "venv_scripts/illada_vargen.py run" in output
    assert "venv_scripts\\illada.py run" in output or "venv_scripts/illada.py run" in output
    assert "venv_scripts\\diffusiongemma.py run" in output or "venv_scripts/diffusiongemma.py run" in output
    assert "venv_scripts\\gemma.py run" in output or "venv_scripts/gemma.py run" in output
    assert "venv_scripts\\qwen3_8b.py run" in output or "venv_scripts/qwen3_8b.py run" in output


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


def test_variant_flag_is_forwarded_to_the_selected_model(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured["models"] = list(model_names)
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "illada", "-v", "p2", "--stage", "generate"
    ]) == 0
    assert captured["models"] == ["illada"]
    assert captured["env_updates"]["MATRIX_VARIANTS"] == "p2"


def test_variant_flag_accepts_all_parallelism_points(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "illada", "-v", "p1", "p2", "p4", "p8",
        "--stage", "generate",
    ]) == 0
    assert captured["env_updates"]["MATRIX_VARIANTS"] == "p1,p2,p4,p8"


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


def test_temporary_output_length_override_is_forwarded(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "qwen3_4b", "-d", "sudoku9",
        "--max-new-tokens", "512",
    ]) == 0
    assert captured["env_updates"]["MAX_NEW_TOKENS"] == "512"


def test_multiple_output_lengths_are_forwarded_as_one_sweep(monkeypatch):
    captured = {}

    def fake_dispatch(model_names, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_bench, "dispatch_model_scripts", fake_dispatch)

    assert run_bench.main([
        "-m", "illada_vargen", "-d", "gsm8k",
        "-max", "1024", "2048",
    ]) == 0
    assert captured["env_updates"]["MAX_NEW_TOKENS"] == "1024,2048"


def test_variant_and_variants_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        run_bench.main([
            "--dry-run", "-m", "illada", "--variant", "p2",
            "--variants", "p1,p2",
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
    assert "gemma" in run_bench.matrix_model_names(run_bench.DEFAULT_MATRIX)
