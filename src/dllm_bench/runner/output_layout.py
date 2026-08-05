"""The output directory convention every stage writes into and reads from:

    output/
      model_output/<model>/<config>/<dataset>/
        _meta.json          # model/config/dataset name + run_metadata (section 6)
        oom_info.json       # present only when OOM invalidates the complete test
        <sample_id>.json    # full GenerationResult, including trace
      model_profiling/<model>/<config>/<dataset>/
        _meta.json          # profiling protocol and run metadata
        <sample_id>.json    # GenerationResult plus per-step profiles
      score_output/<model>/<config>/<dataset>/
        <sample_id>.json    # ScoreResult for that sample
        summary.json        # RunSummary (section 3.4 raw-results-table row)
      visualization_output/<model>/<config>/<dataset>/
        <sample_id>_*.png / .gif

The old layout encoded both axes into a flat ``<model>_<config>`` run ID.
``run_id`` is retained only for locating those legacy artifacts. New writes
always keep model, config, and dataset as separate directory levels.
Splitting by model first, config second, and dataset third is what lets
each model run independently (skip W1 entirely, run iLLaDA without touching
DreamReasoner's output), lets a dataset resume mid-way (each stage checks per-sample
files before redoing work — see ``runner/generate_stage.py``/``score_stage.py``),
and lets `model_output/` be generated on a GPU box and copied elsewhere for
scoring/visualization without dragging the other two directories along.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

MODEL_OUTPUT = "model_output"
MODEL_PROFILING = "model_profiling"
SCORE_OUTPUT = "score_output"
VISUALIZATION_OUTPUT = "visualization_output"
_OUTPUT_SUFFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


_UNSUFFIXED_CONFIGS = {
    ("gemma_dflash", "dflash"),
    ("qwen3_4b", "ar-baseline"),
    ("qwen3_8b", "ar-baseline"),
}


def run_id(model_name: str, config_name: str) -> str:
    if (model_name, config_name) in _UNSUFFIXED_CONFIGS:
        return model_name
    return f"{model_name}_{config_name}"


def legacy_run_id(model_name: str, config_name: str) -> str | None:
    """Return the old run ID when the canonical naming dropped a suffix."""
    canonical = run_id(model_name, config_name)
    legacy = f"{model_name}_{config_name}"
    return legacy if legacy != canonical else None


def _stage_root(output_root: str | Path, stage: str) -> Path:
    """Resolve a stage root from an exact generation-output root.

    ``output_root`` names the directory whose immediate children are model
    run IDs. The standard roots still work naturally (``output/model_output``
    and ``output/model_profiling``). A custom root such as
    ``output/smoke/illada_entropy`` is used verbatim for generations; score
    and visualization artifacts are written beside it under their stage name.
    """
    root = Path(output_root)
    standard_input_stages = {MODEL_OUTPUT, MODEL_PROFILING}
    if root.name in standard_input_stages:
        return root if root.name == stage else root.parent / stage
    if stage in standard_input_stages:
        return root
    return root.parent / stage


def _dataset_output_name(dataset_name: str) -> str:
    suffix = os.environ.get("DLLM_BENCH_OUTPUT_SUFFIX", "").strip()
    if not suffix:
        return dataset_name
    if not _OUTPUT_SUFFIX_RE.fullmatch(suffix):
        raise ValueError(
            "output suffix must contain only letters, digits, underscores, and hyphens"
        )
    return f"{dataset_name}_{suffix}"


def _stage_dir(output_root: str | Path, stage: str, model_name: str, config_name: str, dataset_name: str) -> Path:
    """Use the same <model>/<config>/<dataset> layout for every artifact stage."""
    return (
        _stage_root(output_root, stage)
        / model_name
        / config_name
        / _dataset_output_name(dataset_name)
    )


def model_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _stage_dir(output_root, MODEL_OUTPUT, model_name, config_name, dataset_name)


def model_profiling_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _stage_dir(output_root, MODEL_PROFILING, model_name, config_name, dataset_name)


def _resolve_existing_stage_dir(
    output_root: str | Path,
    stage: str,
    model_name: str,
    config_name: str,
    dataset_name: str,
) -> Path:
    canonical = _stage_dir(output_root, stage, model_name, config_name, dataset_name)
    if canonical.exists():
        return canonical
    legacy_ids = [run_id(model_name, config_name)]
    fully_suffixed = f"{model_name}_{config_name}"
    if fully_suffixed not in legacy_ids:
        legacy_ids.append(fully_suffixed)
    for legacy in legacy_ids:
        legacy_path = (
            _stage_root(output_root, stage)
            / legacy
            / _dataset_output_name(dataset_name)
        )
        if legacy_path.exists():
            return legacy_path
    return canonical


def resolve_model_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    """Read canonical output, falling back to any legacy flat run directory."""
    return _resolve_existing_stage_dir(
        output_root, MODEL_OUTPUT, model_name, config_name, dataset_name
    )


def score_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _stage_dir(output_root, SCORE_OUTPUT, model_name, config_name, dataset_name)


def resolve_score_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _resolve_existing_stage_dir(
        output_root, SCORE_OUTPUT, model_name, config_name, dataset_name
    )


def visualization_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _stage_dir(output_root, VISUALIZATION_OUTPUT, model_name, config_name, dataset_name)


def resolve_visualization_output_dir(
    output_root: str | Path,
    model_name: str,
    config_name: str,
    dataset_name: str,
) -> Path:
    return _resolve_existing_stage_dir(
        output_root, VISUALIZATION_OUTPUT, model_name, config_name, dataset_name
    )


def model_comparison_visualization_output_dir(
    output_root: str | Path,
    model_name: str,
    dataset_name: str,
) -> Path:
    """Return the canonical location for model-specific cross-variant plots."""
    return (
        _stage_root(output_root, VISUALIZATION_OUTPUT)
        / model_name
        / "model_comparison"
        / _dataset_output_name(dataset_name)
    )


def resolve_model_profiling_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _resolve_existing_stage_dir(
        output_root, MODEL_PROFILING, model_name, config_name, dataset_name
    )
