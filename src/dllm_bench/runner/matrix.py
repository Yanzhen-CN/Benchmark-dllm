"""Load and filter the experiment matrix used by both CLI entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..registry import load_yaml


DATASET_GROUP_PREFIXES = {
    "sudoku": "sudoku",
}


def dataset_selector_matches(dataset_name: str, requested: set[str]) -> bool:
    """Whether a matrix dataset is selected by a name or group alias."""
    if not requested or dataset_name in requested:
        return True
    return any(
        alias in requested and dataset_name.startswith(prefix)
        for alias, prefix in DATASET_GROUP_PREFIXES.items()
    )


def unknown_dataset_selectors(
    requested: set[str], available: Iterable[str]
) -> set[str]:
    """Return selectors that are neither exact names nor known groups."""
    return requested.difference(available).difference(DATASET_GROUP_PREFIXES)


@dataclass(frozen=True)
class MatrixJob:
    model_name: str
    model_config: Path
    variants: tuple[str, ...]
    dataset_config: Path
    samples_file: Path | None = None
    max_new_tokens: int = 256
    n_samples: int | None = None
    hellobench_lengths: tuple[str, ...] = ()
    capture_trace: bool | None = None
    profiling_output: bool = False


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
    dataset_names: Iterable[str] | None = None,
) -> tuple[list[MatrixJob], int]:
    path = Path(path).resolve()
    config = load_yaml(path)
    base = _resolve(path.parent, config.get("base_dir", "."))
    requested = set(model_names or [])
    requested_datasets = set(dataset_names or [])
    available = available_matrix_models(path)
    unknown = requested.difference(available)
    if unknown:
        raise ValueError(
            f"unknown model(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(available)}"
        )

    available_datasets = [
        str(
            entry.get("name")
            or Path(entry["config"]).stem
        )
        if isinstance(entry, dict)
        else Path(entry).stem
        for entry in config["datasets"]
    ]
    unknown_datasets = unknown_dataset_selectors(
        requested_datasets, available_datasets
    )
    if unknown_datasets:
        raise ValueError(
            f"unknown dataset(s): {', '.join(sorted(unknown_datasets))}; "
            f"available: {', '.join(available_datasets)}"
        )

    jobs: list[MatrixJob] = []
    for model in config["models"]:
        model_name = str(model.get("name") or Path(model["config"]).stem)
        if requested and model_name not in requested:
            continue
        model_path = _resolve(base, model["config"])
        dataset_overrides = model.get("dataset_overrides", {})
        for dataset_entry in config["datasets"]:
            if isinstance(dataset_entry, str):
                dataset_entry = {"config": dataset_entry}
            dataset_name = str(
                dataset_entry.get("name") or Path(dataset_entry["config"]).stem
            )
            if dataset_entry.get("optional") and dataset_name not in requested_datasets:
                continue
            if not dataset_selector_matches(dataset_name, requested_datasets):
                continue
            override = dataset_overrides.get(dataset_name, {})
            if not isinstance(override, dict):
                raise ValueError(
                    f"dataset override for {model_name} x {dataset_name} must be a mapping"
                )
            jobs.append(
                MatrixJob(
                    model_name=model_name,
                    model_config=model_path,
                    variants=tuple(override.get("variants", model["variants"])),
                    dataset_config=_resolve(base, dataset_entry["config"]),
                    samples_file=(
                        _resolve(base, dataset_entry["samples_file"])
                        if dataset_entry.get("samples_file")
                        else None
                    ),
                    max_new_tokens=int(
                        override.get(
                            "max_new_tokens",
                            dataset_entry.get("max_new_tokens", 256),
                        )
                    ),
                    n_samples=(
                        int(override["n_samples"])
                        if override.get("n_samples") is not None
                        else (
                            int(dataset_entry["n_samples"])
                            if dataset_entry.get("n_samples") is not None
                            else None
                        )
                    ),
                    hellobench_lengths=tuple(
                        str(value)
                        for value in override.get(
                            "hellobench_lengths",
                            dataset_entry.get("hellobench_lengths", ()),
                        )
                    ),
                    capture_trace=(
                        bool(override.get("capture_trace", dataset_entry.get(
                            "capture_trace", config.get("capture_trace")
                        )))
                        if override.get("capture_trace", dataset_entry.get(
                            "capture_trace", config.get("capture_trace")
                        )) is not None
                        else None
                    ),
                    profiling_output=bool(config.get("profiling_output", False)),
                )
            )
    return jobs, int(config.get("seed", 42))
