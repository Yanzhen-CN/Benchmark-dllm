"""The output directory convention every stage writes into and reads from:

    output/
      model_output/<model>_<config>/<dataset>/
        _meta.json          # model/config/dataset name + run_metadata (section 6)
        <sample_id>.json    # full GenerationResult, including trace
      score_output/<model>_<config>/<dataset>/
        <sample_id>.json    # ScoreResult for that sample
        summary.json        # RunSummary (section 3.4 raw-results-table row)
      visualization_output/<model>_<config>/<dataset>/
        <sample_id>_*.png / .gif

Splitting by ``<model>_<config>`` first, ``<dataset>`` second is what lets
each model run independently (skip W1 entirely, run iLLaDA without touching
DreamReasoner's output), lets a dataset resume mid-way (each stage checks per-sample
files before redoing work — see ``runner/generate_stage.py``/``score_stage.py``),
and lets `model_output/` be generated on a GPU box and copied elsewhere for
scoring/visualization without dragging the other two directories along.
"""

from __future__ import annotations

from pathlib import Path

MODEL_OUTPUT = "model_output"
SCORE_OUTPUT = "score_output"
VISUALIZATION_OUTPUT = "visualization_output"


def run_id(model_name: str, config_name: str) -> str:
    return f"{model_name}_{config_name}"


def _stage_dir(output_root: str | Path, stage: str, model_name: str, config_name: str, dataset_name: str) -> Path:
    return Path(output_root) / stage / run_id(model_name, config_name) / dataset_name


def model_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _stage_dir(output_root, MODEL_OUTPUT, model_name, config_name, dataset_name)


def score_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _stage_dir(output_root, SCORE_OUTPUT, model_name, config_name, dataset_name)


def visualization_output_dir(output_root: str | Path, model_name: str, config_name: str, dataset_name: str) -> Path:
    return _stage_dir(output_root, VISUALIZATION_OUTPUT, model_name, config_name, dataset_name)
