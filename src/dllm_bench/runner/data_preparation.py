"""Prepare normalized benchmark samples before any timed model work.

Both the explicit ``prepare_data.py`` entry point and normal ``run`` path use
this module.  A run therefore only prepares data when its exact configured
cache artifact is missing; preparation is never inside a model timing window.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..data_paths import ensure_data_layout
from ..datasets.base import Dataset, Sample
from ..datasets.io import load_samples_file
from ..registry import build_dataset, load_yaml


PREPARED_SCHEMA_VERSION = 1


class DataPreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedDataset:
    dataset_name: str
    samples_path: Path
    manifest_path: Path
    sample_count: int
    prepared_now: bool


def _hash_file(hasher: Any, path: Path) -> None:
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)


def preparation_fingerprint(
    dataset_config: str | Path,
    dataset: Dataset,
    samples_file: str | Path | None = None,
) -> str:
    """Fingerprint the config, loader implementation, and optional raw file."""
    hasher = hashlib.sha256(f"prepared-schema:{PREPARED_SCHEMA_VERSION}\n".encode())
    _hash_file(hasher, Path(dataset_config).resolve())
    source_path = inspect.getsourcefile(type(dataset))
    if source_path and Path(source_path).is_file():
        _hash_file(hasher, Path(source_path))
    signature_fn = getattr(dataset, "preparation_signature", None)
    if callable(signature_fn):
        signature = json.dumps(
            signature_fn(), sort_keys=True, separators=(",", ":"), default=str
        )
        hasher.update(signature.encode("utf-8"))
    if samples_file is not None:
        raw_path = Path(samples_file).resolve()
        if not raw_path.is_file():
            raise DataPreparationError(f"raw samples file does not exist: {raw_path}")
        _hash_file(hasher, raw_path)
    return hasher.hexdigest()[:20]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_samples(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as output:
            for sample in samples:
                record = {
                    "sample_id": sample.sample_id,
                    "prompt": sample.prompt,
                    "reference": _jsonable(sample.reference),
                    "meta": _jsonable(sample.meta),
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _prune_old_prepared_artifacts(dataset_root: Path, keep_dir: Path) -> None:
    """Keep only the active fingerprint under one dataset's prepared root."""
    dataset_root = dataset_root.resolve()
    keep_dir = keep_dir.resolve()
    if keep_dir.parent != dataset_root:
        raise DataPreparationError(
            f"refusing to prune prepared data outside {dataset_root}: {keep_dir}"
        )
    if not dataset_root.is_dir():
        return
    for candidate in dataset_root.iterdir():
        if candidate.resolve() == keep_dir:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate)


def prepare_dataset(
    dataset_config: str | Path,
    *,
    samples_file: str | Path | None = None,
    dataset: Dataset | None = None,
    force: bool = False,
) -> PreparedDataset:
    dataset_config = Path(dataset_config).resolve()
    dataset = dataset or build_dataset(dataset_config)
    fingerprint = preparation_fingerprint(dataset_config, dataset, samples_file)
    dataset_prepared_root = (
        ensure_data_layout()["datasets"] / "prepared" / dataset.name
    )
    prepared_dir = dataset_prepared_root / fingerprint
    samples_path = prepared_dir / "samples.jsonl"
    manifest_path = prepared_dir / "manifest.json"

    if not force and samples_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _prune_old_prepared_artifacts(dataset_prepared_root, prepared_dir)
        return PreparedDataset(
            dataset.name, samples_path, manifest_path,
            int(manifest["sample_count"]), False,
        )

    try:
        samples = (
            load_samples_file(samples_file, dataset.name)
            if samples_file is not None
            else dataset.load_samples()
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise DataPreparationError(
            f"failed to prepare real {dataset.name} data: {exc}"
        ) from exc
    if not samples:
        raise DataPreparationError(
            f"real data preparation is not implemented for {dataset.name!r}; "
            "configure its official loader or provide samples_file"
        )

    _write_samples(list(samples), samples_path)
    manifest = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "dataset": dataset.name,
        "dataset_config": str(dataset_config),
        "source": str(Path(samples_file).resolve()) if samples_file else "dataset.load_samples",
        "fingerprint": fingerprint,
        "sample_count": len(samples),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _prune_old_prepared_artifacts(dataset_prepared_root, prepared_dir)
    return PreparedDataset(
        dataset.name, samples_path, manifest_path, len(samples), True
    )


def load_prepared_samples(prepared: PreparedDataset) -> list[Sample]:
    return load_samples_file(prepared.samples_path, prepared.dataset_name)


def prepare_matrix_datasets(
    experiment_config: str | Path,
    *,
    force: bool = False,
    dataset_names: list[str] | tuple[str, ...] = (),
) -> list[PreparedDataset]:
    """Prepare each unique dataset entry in an experiment matrix once."""
    experiment_path = Path(experiment_config).resolve()
    config = load_yaml(experiment_path)
    base = Path(config.get("base_dir", "."))
    if not base.is_absolute():
        base = (experiment_path.parent / base).resolve()

    prepared: list[PreparedDataset] = []
    seen: set[tuple[Path, Path | None]] = set()
    requested = set(dataset_names)
    available: set[str] = set()
    for entry in config["datasets"]:
        entry = {"config": entry} if isinstance(entry, str) else entry
        dataset_config = Path(entry["config"])
        if not dataset_config.is_absolute():
            dataset_config = (base / dataset_config).resolve()
        dataset_name = str(load_yaml(dataset_config)["dataset"])
        available.add(dataset_name)
        if requested and dataset_name not in requested:
            continue
        samples_file = Path(entry["samples_file"]) if entry.get("samples_file") else None
        if samples_file is not None and not samples_file.is_absolute():
            samples_file = (base / samples_file).resolve()
        key = (dataset_config, samples_file)
        if key in seen:
            continue
        seen.add(key)
        prepared.append(
            prepare_dataset(dataset_config, samples_file=samples_file, force=force)
        )
    unknown = requested.difference(available)
    if unknown:
        raise DataPreparationError(
            f"unknown dataset(s): {', '.join(sorted(unknown))}; available: "
            f"{', '.join(sorted(available))}"
        )
    return prepared
