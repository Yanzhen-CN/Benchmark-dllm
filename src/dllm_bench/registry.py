"""Load YAML model/dataset configs (``configs/models/*.yaml``,
``configs/datasets/*.yaml``) into instantiated objects.

This is what makes "switch models, not the pipeline" (design doc Appendix D)
concrete in code: the orchestrator only ever calls :func:`build_model_adapter`
/ :func:`build_dataset` and gets back something satisfying
:class:`~dllm_bench.interfaces.ModelAdapter` /
:class:`~dllm_bench.datasets.base.Dataset` — which concrete class that is
lives entirely in the YAML file.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .datasets.base import Dataset
from .interfaces import ModelAdapter
from .models.hf_diffusion import DiffusionStepConfig

_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{(\w+)\}$")


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_PLACEHOLDER_RE.match(value)
        if match:
            return os.environ.get(match.group(1), value)
        return value
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _expand_env_placeholders(raw)


def _import_from_string(dotted_path: str) -> Any:
    module_path, _, attr = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"expected a dotted module path, got {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def list_model_variants(config_path: str | Path) -> list[str]:
    """The named configs (e.g. `p1`/`p2`) a model config file declares."""
    config = load_yaml(config_path)
    return list(config["configs"].keys())


def model_name(config_path: str | Path) -> str:
    return load_yaml(config_path)["model"]


def _resolve_variant(config: dict[str, Any], config_path: str | Path, variant: str | None) -> str:
    available = list(config["configs"].keys())
    if variant is not None:
        if variant not in available:
            raise ValueError(
                f"{config_path} has no variant {variant!r}; available: {available}"
            )
        return variant
    if len(available) == 1:
        return available[0]
    raise ValueError(
        f"{config_path} declares multiple configs {available}; pass variant= to pick one"
    )


def build_model_adapter(config_path: str | Path, variant: str | None = None) -> ModelAdapter:
    """Builds the `ModelAdapter` for one named config inside a model's YAML.

    Every `configs/models/*.yaml` file is one model with one or more named
    configs nested under `configs:` (e.g. `illada.yaml`'s `p1`/`p2`).
    `variant` picks which one; it can be omitted only when the file declares
    exactly one (e.g. `qwen3_4b.yaml`'s single `ar-baseline`).
    """
    config = load_yaml(config_path)
    resolved_variant = _resolve_variant(config, config_path, variant)
    variant_config = config["configs"][resolved_variant]

    adapter_cls = _import_from_string(variant_config["adapter"])
    init_kwargs = dict(variant_config.get("init_kwargs", {}))
    init_kwargs.setdefault("config_name", resolved_variant)

    if "step_config" in variant_config:
        init_kwargs["step_config"] = DiffusionStepConfig(**variant_config["step_config"])

    return adapter_cls(**init_kwargs)


def build_dataset(config_path: str | Path) -> Dataset:
    """Instantiates the configured `Dataset` subclass with no samples loaded.

    Constructor kwargs come from the config's `dataset_kwargs:` block (e.g.
    MBPP's execution timeout) — anything that isn't a `samples` list. See
    :func:`dataset_run_defaults` for the `sample_size`/`seed` fields that
    control *how many* samples and with what seed, which the CLI applies
    rather than the dataset constructor.

    Populating real samples (downloading GSM8K/MBPP/etc., generating the
    RULER/Sudoku synthetic sets) is orchestrator/experiment-script territory,
    not the registry's — see each dataset module's `load_samples`.
    """
    config = load_yaml(config_path)
    dataset_cls = _import_from_string(config["dataset_class"])
    dataset_name = Path(config_path).stem
    if dataset_name in {"sudoku4", "sudoku9"}:
        shot_count = int(os.environ.get("DLLM_BENCH_SUDOKU_SHOT", "0"))
        if shot_count not in {0, 1}:
            raise ValueError(f"DLLM_BENCH_SUDOKU_SHOT must be 0 or 1, got {shot_count}")
        if shot_count == 1:
            dataset_cls = _import_from_string(
                "dllm_bench.datasets.sudoku4.Sudoku4OneShotDataset"
                if dataset_name == "sudoku4"
                else "dllm_bench.datasets.sudoku9.Sudoku9OneShotDataset"
            )
    dataset_kwargs = dict(config.get("dataset_kwargs", {}))
    return dataset_cls(**dataset_kwargs)


def dataset_run_defaults(config_path: str | Path) -> dict[str, Any]:
    """The `sample_size`/`seed` a dataset config declares (section 6), for the
    CLI to fall back to when `--n-samples`/`--seed` aren't given explicitly."""
    config = load_yaml(config_path)
    return {
        "sample_size": config.get("sample_size"),
        "seed": config.get("seed", 42),
    }
