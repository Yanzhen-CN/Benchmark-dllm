from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import prepare_data
from venv_scripts import root as root_venv
from dllm_bench.cli import main
from dllm_bench.datasets.base import Dataset, Sample, ScoreResult
from dllm_bench.runner.data_preparation import (
    load_prepared_samples,
    prepare_dataset,
    prepare_matrix_datasets,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
GSM8K_CONFIG = REPO_ROOT / "configs" / "datasets" / "gsm8k.yaml"
FULL_MATRIX_CONFIG = REPO_ROOT / "configs" / "experiments" / "full_matrix.yaml"
PREPARE_SCRIPT = REPO_ROOT / "prepare_data.py"


class _PreparedMatrixStub(Dataset):
    def __init__(self, name: str) -> None:
        self.name = name

    def load_samples(self, n: int | None = None) -> list[Sample]:
        samples = [Sample(f"{self.name}-0", "prompt", "reference")]
        return samples[:n] if n is not None else samples

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        return ScoreResult(1.0)


def _write_gsm8k_source(path: Path) -> None:
    records = [
        {
            "sample_id": f"local-{index}",
            "prompt": f"What is {index}+1?",
            "reference": float(index + 1),
            "meta": {"source_index": index},
        }
        for index in range(4)
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_prepare_dataset_writes_and_reuses_normalized_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / ".data"))
    source = tmp_path / "gsm8k.jsonl"
    _write_gsm8k_source(source)

    first = prepare_dataset(GSM8K_CONFIG, samples_file=source)
    second = prepare_dataset(GSM8K_CONFIG, samples_file=source)

    assert first.prepared_now is True
    assert second.prepared_now is False
    assert first.samples_path == second.samples_path
    assert first.sample_count == 4
    assert [sample.sample_id for sample in load_prepared_samples(second)] == [
        "local-0", "local-1", "local-2", "local-3"
    ]


def test_prepare_dataset_prunes_old_fingerprint_after_success(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / ".data"))
    source = tmp_path / "gsm8k.jsonl"
    _write_gsm8k_source(source)
    first = prepare_dataset(GSM8K_CONFIG, samples_file=source)
    old_dir = first.samples_path.parent

    source.write_text(
        source.read_text(encoding="utf-8")
        + json.dumps({"sample_id": "new", "prompt": "new", "reference": 1})
        + "\n",
        encoding="utf-8",
    )
    second = prepare_dataset(GSM8K_CONFIG, samples_file=source)

    assert second.samples_path.parent != old_dir
    assert second.samples_path.is_file()
    assert not old_dir.exists()
    assert source.is_file()


def test_prepare_data_cli_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / ".data"))
    source = tmp_path / "gsm8k.jsonl"
    _write_gsm8k_source(source)
    runner = CliRunner()
    args = [
        "prepare-data",
        "--dataset-config", str(GSM8K_CONFIG),
        "--samples-file", str(source),
    ]

    first = runner.invoke(main, args)
    second = runner.invoke(main, args)

    assert first.exit_code == 0, first.output
    assert "prepared: 4 samples" in first.output
    assert second.exit_code == 0, second.output
    assert "cached: 4 samples" in second.output


def test_full_matrix_prepare_visits_all_six_datasets(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / ".data"))
    monkeypatch.setattr(
        "dllm_bench.runner.data_preparation.build_dataset",
        lambda config_path: _PreparedMatrixStub(Path(config_path).stem),
    )

    first = prepare_matrix_datasets(FULL_MATRIX_CONFIG)
    second = prepare_matrix_datasets(FULL_MATRIX_CONFIG)

    assert [item.dataset_name for item in first] == [
        "gsm8k", "mbpp", "structeval_t", "sudoku", "ruler", "hellobench"
    ]
    assert all(item.sample_count == 1 and item.prepared_now for item in first)
    assert all(not item.prepared_now for item in second)
    assert all(item.samples_path.is_file() and item.manifest_path.is_file() for item in first)


def test_full_matrix_prepare_can_filter_to_sudoku(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / ".data"))
    monkeypatch.setattr(
        "dllm_bench.runner.data_preparation.build_dataset",
        lambda config_path: _PreparedMatrixStub(Path(config_path).stem),
    )

    prepared = prepare_matrix_datasets(
        FULL_MATRIX_CONFIG, dataset_names=["sudoku"]
    )

    assert [item.dataset_name for item in prepared] == ["sudoku"]


def test_prepare_data_public_dataset_flag_is_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr(prepare_data, "run_in_root_venv", lambda *args: None)
    monkeypatch.setattr(
        "dllm_bench.runner.data_preparation.prepare_matrix_datasets",
        lambda experiment_config, **kwargs: captured.update(kwargs) or [],
    )

    assert prepare_data.main(["-d", "sudoku"]) == 0
    assert captured["dataset_names"] == ["sudoku"]


def test_prepare_data_help_works_outside_repository_without_creating_venv(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--experiment-config" in result.stdout


def test_prepare_data_creates_and_reexecutes_in_root_venv(tmp_path, monkeypatch):
    root_directory = tmp_path / ".venvs" / "root"
    python = root_venv.venv_python(root_directory)
    commands: list[list[str]] = []
    monkeypatch.setattr(root_venv, "ROOT_VENV", root_directory)
    monkeypatch.delenv(root_venv.INSIDE_ROOT_VENV, raising=False)

    def fake_run(command, **kwargs):
        command = [str(value) for value in command]
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()
        if command[-2:] == ["-c", "import dllm_bench, yaml, matplotlib"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(root_venv.subprocess, "run", fake_run)
    monkeypatch.setattr(
        root_venv.os,
        "execve",
        lambda *args: (_ for _ in ()).throw(RuntimeError("reexec")),
    )

    with pytest.raises(RuntimeError, match="reexec"):
        root_venv.run_in_root_venv(PREPARE_SCRIPT, [])

    assert any(command[1:3] == ["-m", "venv"] for command in commands)
    assert any(command[-4:] == ["pip", "install", "-e", "."] for command in commands)
