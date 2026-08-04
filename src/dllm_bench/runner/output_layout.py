"""The output directory convention every stage writes into and reads from:

    output/
      model_output/<run-id>/<dataset>/
        _meta.json          # model/config/dataset name + run_metadata (section 6)
        oom_info.json       # present only when OOM invalidates the complete test
        <sample_id>.json    # full GenerationResult, including trace
      model_profiling/<run-id>/<dataset>/
        _meta.json          # profiling protocol and run metadata
        <sample_id>.json    # GenerationResult plus per-forward profiles
      score_output/<model>_<config>/<dataset>/
        <sample_id>.json    # ScoreResult for that sample
        summary.json        # RunSummary (section 3.4 raw-results-table row)
      visualization_output/<model>_<config>/<dataset>/
        <sample_id>_*.png / .gif

The run ID is normally ``<model>_<config>``. Each Qwen model's sole
``ar-baseline`` configuration uses just its model name because the suffix is
redundant. DFlash's model config is already named ``gemma_dflash``, so its
``dflash`` variant is not appended a second time.
Splitting by run ID first, ``<dataset>`` second is what lets
each model run independently (skip W1 entirely, run iLLaDA without touching
DreamReasoner's output), lets a dataset resume mid-way (each stage checks per-sample
files before redoing work — see ``runner/generate_stage.py``/``score_stage.py``),
and lets `model_output/` be generated on a GPU box and copied elsewhere for
scoring/visualization without dragging the other two directories along.
"""

from __future__ import annotations

from pathlib import Path

MODEL_OUTPUT = "model_output"
MODEL_PROFILING = "model_profiling"
SCORE_OUTPUT = "score_output"
VISUALIZATION_OUTPUT = "visualization_output"


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


def _stage_dir(output_root: str | Path, stage: str, model_name: str, config_name: str, dataset_name: str) -> Path:
    """Use the same <model_config>/<dataset> layout for every artifact stage."""
    return Path(output_root) / stage / run_id(model_name, config_name) / dataset_name


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
    legacy = legacy_run_id(model_name, config_name)
    if legacy is not None:
        legacy_path = Path(output_root) / stage / legacy / dataset_name
        if legacy_path.exists():
            return legacy_path
    return canonical


def resolve_model_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    """Read canonical output, falling back to the pre-rename Qwen directory."""
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


def model_comparison_visualization_output_dir(
    output_root: str | Path,
    model_name: str,
    dataset_name: str,
) -> Path:
    """Return the canonical location for model-specific cross-variant plots."""
    return (
        Path(output_root)
        / VISUALIZATION_OUTPUT
        / model_name
        / "model_comparison"
        / dataset_name
    )


def resolve_model_profiling_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _resolve_existing_stage_dir(
        output_root, MODEL_PROFILING, model_name, config_name, dataset_name
    )
