"""prepare_model.py is a standalone script (not part of the dllm_bench
package), so it's exercised via subprocess rather than import."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import prepare_model

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "prepare_model.py"
CONFIGS_DIR = REPO_ROOT / "configs"


def _run(args, cwd):
    # Strip any HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE the outer test
    # process might have set (e.g. another test importing dllm_bench.cli,
    # which sets HF_HOME as a real env var when a CLI command runs) —
    # subprocess.run inherits the parent environment by default, and these
    # tests specifically exercise the "nothing set yet" default path.
    env = {
        k: v for k, v in os.environ.items()
        if k not in (
            "DLLM_DATA_ROOT", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE",
            "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE",
        )
    }
    # These direct-mode tests exercise the implementation entered by a model's
    # isolated venv wrapper.  The public root-Python path is covered separately
    # by the dispatch tests below.
    env["DLLM_VENV"] = "1"
    env["DLLM_MODEL"] = "test-model"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_prepare_model_visits_every_mock_variant_by_default(tmp_path):
    result = _run(["--model-config", str(CONFIGS_DIR / "models" / "mock.yaml")], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "['default', 'fast']" in result.stdout
    assert "[default] mock: no Hugging Face checkpoint" in result.stdout
    assert "[fast] mock: no Hugging Face checkpoint" in result.stdout


def test_prepare_model_respects_single_variant(tmp_path):
    result = _run(
        ["--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"), "--variant", "fast"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "['fast']" in result.stdout
    assert "[default] mock" not in result.stdout


def test_prepare_model_skips_api_backed_adapters(tmp_path):
    result = _run(["--model-config", str(CONFIGS_DIR / "models" / "w1.yaml")], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "no Hugging Face checkpoint to download" in result.stdout
    assert result.stdout.count("no Hugging Face checkpoint to download") == 3


def test_prepare_model_uses_repository_data_dir_by_default(tmp_path):
    result = _run(["--model-config", str(CONFIGS_DIR / "models" / "mock.yaml")], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    expected = CONFIGS_DIR.parent / "data" / "huggingface"
    assert str(expected) in result.stdout
    assert expected.exists()


def test_prepare_model_downloads_shared_checkpoint_once_without_building_adapter(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        prepare_model,
        "_download_snapshot",
        lambda repo_id, revision, cache_dir: calls.append(
            (repo_id, revision, cache_dir)
        ) or "/cached/illada",
    )

    prepare_model._prepare_one(
        str(CONFIGS_DIR / "models" / "illada.yaml"), None, None
    )

    assert len(calls) == 1
    assert calls[0][0] == "GSAI-ML/iLLaDA-8B-Instruct"
    assert calls[0][1] is None
    assert (
        "[p1,p2,p4,p8] cached: /cached/illada"
        in capsys.readouterr().out
    )


def test_prepare_model_downloads_dflash_target_and_draft(monkeypatch):
    calls = []
    monkeypatch.setattr(
        prepare_model,
        "_download_snapshot",
        lambda repo_id, revision, cache_dir: calls.append(repo_id) or "/cached/model",
    )

    prepare_model._prepare_one(
        str(CONFIGS_DIR / "models" / "gemma_dflash.yaml"), None, None
    )

    assert calls == [
        "google/gemma-4-26B-A4B-it",
        "z-lab/gemma-4-26B-A4B-it-DFlash",
    ]


def test_prepare_model_rejects_variant_and_variants_together(tmp_path):
    result = _run(
        [
            "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
            "--variant", "default", "--variants", "default,fast",
        ],
        cwd=tmp_path,
    )
    assert result.returncode != 0


def test_prepare_model_defaults_to_all_matrix_models_via_isolated_scripts(tmp_path):
    result = _run(["--dry-run"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    for model in (
        "qwen3_4b",
        "qwen3_8b",
        "illada",
        "illada_vargen",
        "dreamreasoner",
        "diffusiongemma",
        "gemma",
        "gemma_dflash",
    ):
        assert f"{model}.py prepare" in result.stdout


def test_prepare_model_matrix_mode_can_select_models(tmp_path):
    result = _run(["-m", "illada,qwen3_4b", "--dry-run"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "illada.py prepare" in result.stdout
    assert "qwen3_4b.py prepare" in result.stdout
    assert "diffusiongemma.py prepare" not in result.stdout


def test_prepare_model_direct_mode_dispatches_when_started_outside_venv(tmp_path):
    env = {
        key: value for key, value in os.environ.items()
        if key not in ("DLLM_MODEL", "DLLM_VENV", "VIRTUAL_ENV")
    }
    code = (
        "import runpy, sys; "
        "sys.prefix = sys.base_prefix; "
        f"sys.argv = [{str(SCRIPT)!r}, '--model-config', "
        f"{str(CONFIGS_DIR / 'models' / 'illada.yaml')!r}, '--variant', 'p2', '--dry-run']; "
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "illada.py prepare" in result.stdout


def test_arbitrary_active_venv_is_not_mistaken_for_model_venv(monkeypatch):
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    monkeypatch.setenv("DLLM_VENV", "/some/venv")

    assert prepare_model._running_in_model_venv() is False


def test_model_wrapper_marker_enters_direct_prepare_mode(monkeypatch):
    monkeypatch.setenv("DLLM_MODEL", "illada")
    monkeypatch.setenv("DLLM_VENV", "/managed/.venvs/illada")

    assert prepare_model._running_in_model_venv() is True
