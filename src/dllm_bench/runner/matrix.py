"""Load and filter the experiment matrix used by both CLI entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..registry import load_yaml


@dataclass(frozen=True)
class MatrixJob:
    model_name: str
    model_config: Path
    variants: tuple[str, ...]
    dataset_config: Path
    samples_file: Path | None = None
    max_new_tokens: int = 256


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def available_matrix_models(path: str | Path) -> list[str]:
    config = load_yaml(path)
    names: list[str] = []
    for model in config["models"]:
        names.append(str(model.get("name") or Path(model["config"]).stem))
    return names


def load_matrix_jobs(
    path: str | Path,
    model_names: Iterable[str] | None = None,
) -> tuple[list[MatrixJob], int]:
    path = Path(path).resolve()
    config = load_yaml(path)
    base = _resolve(path.parent, config.get("base_dir", "."))
    requested = set(model_names or [])
    available = available_matrix_models(path)
    unknown = requested.difference(available)
    if unknown:
        raise ValueError(
            f"unknown model(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(available)}"
        )

    jobs: list[MatrixJob] = []
    for model in config["models"]:
        model_name = str(model.get("name") or Path(model["config"]).stem)
        if requested and model_name not in requested:
            continue
        model_path = _resolve(base, model["config"])
        for dataset_entry in config["datasets"]:
            if isinstance(dataset_entry, str):
                dataset_entry = {"config": dataset_entry}
            jobs.append(
                MatrixJob(
                    model_name=model_name,
                    model_config=model_path,
                    variants=tuple(model["variants"]),
                    dataset_config=_resolve(base, dataset_entry["config"]),
                    samples_file=(
                        _resolve(base, dataset_entry["samples_file"])
                        if dataset_entry.get("samples_file")
                        else None
                    ),
                    max_new_tokens=int(dataset_entry.get("max_new_tokens", 256)),
                )
            )
    return jobs, int(config.get("seed", 42))
