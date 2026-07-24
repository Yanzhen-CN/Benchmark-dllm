"""prepare_model.py is a standalone script (not part of the dllm_bench
package), so it's exercised via subprocess rather than import."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "prepare_model.py"
CONFIGS_DIR = REPO_ROOT / "configs"


def _run(args, cwd):
    # Strip any HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE the outer test
    # process might have set (e.g. another test importing dllm_bench.cli,
    # which sets HF_HOME as a real env var when a CLI command runs) —
    # subprocess.run inherits the parent environment by default, and these
    # tests specifically exercise the "nothing set yet" default path.
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_prepare_model_warms_every_mock_variant_by_default(tmp_path):
    result = _run(["--model-config", str(CONFIGS_DIR / "models" / "mock.yaml")], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "['default', 'fast']" in result.stdout
    assert "[default] mock: ready" in result.stdout
    assert "[fast] mock: ready" in result.stdout


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
    assert "no local weights to warm" in result.stdout
    assert result.stdout.count("no local weights to warm") == 3  # standard/jump/gidd


def test_prepare_model_uses_project_relative_cache_dir_by_default(tmp_path):
    result = _run(["--model-config", str(CONFIGS_DIR / "models" / "mock.yaml")], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert str(tmp_path) in result.stdout
    assert ".hf_cache" in result.stdout
    assert (tmp_path / ".hf_cache").exists()


def test_prepare_model_rejects_variant_and_variants_together(tmp_path):
    result = _run(
        [
            "--model-config", str(CONFIGS_DIR / "models" / "mock.yaml"),
            "--variant", "default", "--variants", "default,fast",
        ],
        cwd=tmp_path,
    )
    assert result.returncode != 0
