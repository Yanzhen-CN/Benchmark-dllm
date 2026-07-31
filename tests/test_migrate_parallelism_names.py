from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import migrate_parallelism_names as migration


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dry_run_does_not_mutate_artifacts(tmp_path):
    source = tmp_path / "model_output" / "illada_best" / "gsm8k"
    meta = source / "_meta.json"
    _write_json(
        meta,
        {
            "model_name": "illada",
            "config_name": "best",
            "run_metadata": {"model": "illada", "config": "best"},
        },
    )

    migration.migrate(tmp_path, apply=False)

    assert source.is_dir()
    assert json.loads(meta.read_text())["config_name"] == "best"


def test_apply_renames_directory_and_metadata(tmp_path):
    source = tmp_path / "model_output" / "illada_vargen_fast" / "mbpp"
    meta = source / "_meta.json"
    _write_json(
        meta,
        {
            "model_name": "illada_vargen",
            "config_name": "fast",
            "run_metadata": {"model": "illada_vargen", "config": "fast"},
        },
    )

    migration.migrate(tmp_path, apply=True)

    target = tmp_path / "model_output" / "illada_vargen_p2" / "mbpp"
    payload = json.loads((target / "_meta.json").read_text())
    assert target.is_dir()
    assert payload["config_name"] == "p2"
    assert payload["run_metadata"]["config"] == "p2"
    assert (tmp_path / "parallelism_name_migration.json").is_file()


def test_apply_updates_aggregate_csv(tmp_path):
    table = tmp_path / "report" / "raw_results.csv"
    table.parent.mkdir(parents=True)
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", "Config", "Direction"])
        writer.writeheader()
        writer.writerow(
            {
                "Model": "dreamreasoner",
                "Config": "best",
                "Direction": "dreamreasoner/best relative to qwen3_8b/ar-baseline",
            }
        )

    migration.migrate(tmp_path, apply=True)

    with table.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["Config"] == "p1"
    assert row["Direction"].startswith("dreamreasoner/p1 relative to")


def test_existing_target_aborts_without_partial_changes(tmp_path):
    source = tmp_path / "model_output" / "illada_best"
    target = tmp_path / "model_output" / "illada_p1"
    source.mkdir(parents=True)
    target.mkdir(parents=True)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        migration.migrate(tmp_path, apply=True)

    assert source.is_dir()
    assert target.is_dir()
