"""Curated aggregate visualizations built from existing benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from ...runner.persistence import load_generation_result
from .paper_assets import render_paper_assets
from .profiling_comparison import (
    build_profiling_comparison_series,
    render_profiling_comparison_report,
)


def _selected(value: str, names: set[str]) -> bool:
    if not names:
        return True
    return value in names or ("sudoku" in names and value.startswith("sudoku"))


def render_report_assets_from_output(
    output_root: str | Path,
    *,
    model_names: Iterable[str] = (),
    dataset_names: Iterable[str] = (),
) -> list[str]:
    """Render only the two paper overview figures, not the full raw report."""
    output = Path(output_root)
    wanted_models = set(model_names)
    wanted_datasets = set(dataset_names)
    summaries = []
    for path in sorted((output / "score_output").rglob("summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary.setdefault("dataset_name", path.parent.name)
        if wanted_models and summary.get("model_name") not in wanted_models:
            continue
        if not _selected(str(summary.get("dataset_name", "")), wanted_datasets):
            continue
        summaries.append(summary)
    for summary in summaries:
        model_name = str(summary.get("model_name", ""))
        config_name = str(summary.get("config_name", ""))
        dataset_name = str(summary.get("dataset_name", ""))
        trace_path = (
            output
            / "visualization_output"
            / model_name
            / config_name
            / dataset_name
            / "dataset_trace_summary.json"
        )
        if trace_path.exists():
            continue
        tpf = summary.get("tpf", summary.get("accepted_tokens_per_forward"))
        if not isinstance(tpf, (int, float)):
            continue
        trace_summary = {
            "dataset": dataset_name,
            "model": model_name,
            "config": config_name,
            "selected_samples": summary.get("n_samples"),
            "trace_samples": 0,
            "trace_coverage_rate": 0.0,
            "tpf": float(tpf),
            "mean_tpf": {
                "mean": float(tpf),
                "basis": "total accepted-token events / productive model forwards",
            },
            "overview_only": True,
            "note": (
                "Minimal report dependency: accepted-token TPF from score summary. "
                "Generate dataset scope for the complete Task 4 diagnostics."
            ),
        }
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return [
        str(path)
        for path in render_paper_assets(summaries, output / "report")
    ]


def render_profiling_comparison_from_output(
    output_root: str | Path,
    *,
    model_names: Iterable[str] = (),
    dataset_names: Iterable[str] = (),
) -> list[str]:
    """Render cross-model profiling figures from saved per-sample JSON files."""
    output = Path(output_root)
    wanted_models = set(model_names)
    wanted_datasets = set(dataset_names)
    series = []
    profiling_root = output / "model_profiling"
    if not profiling_root.exists():
        return []
    for model_dir in sorted(path for path in profiling_root.iterdir() if path.is_dir()):
        if wanted_models and model_dir.name not in wanted_models:
            continue
        for variant_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            for dataset_dir in sorted(path for path in variant_dir.iterdir() if path.is_dir()):
                if not _selected(dataset_dir.name, wanted_datasets):
                    continue
                records = []
                for path in sorted(dataset_dir.glob("*.json")):
                    if path.name.startswith("_") or path.name == "oom_info.json":
                        continue
                    try:
                        generation = load_generation_result(path)
                    except (OSError, ValueError, TypeError):
                        continue
                    sample = SimpleNamespace(sample_id=generation.request.sample_id)
                    records.append((sample, generation))
                if not records:
                    continue
                series.append(
                    build_profiling_comparison_series(
                        label=f"{model_dir.name}/{variant_dir.name}",
                        dataset_name=dataset_dir.name,
                        records=records,
                        model_name=model_dir.name,
                        config_name=variant_dir.name,
                    )
                )
    written = render_profiling_comparison_report(
        series,
        output / "visualization_output" / "profiling_comparison",
    )
    return list(written.values())
