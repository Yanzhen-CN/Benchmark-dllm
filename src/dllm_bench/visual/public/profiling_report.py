"""Dataset-level reports for public per-forward profiling metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...datasets.base import Sample
from ...interfaces import GenerationResult
from .trace_metrics import (
    StepProfilingRow,
    build_step_profiling,
    plot_step_profiling,
)


def build_dataset_profiling_summary(
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    *,
    model_name: str | None = None,
    config_name: str | None = None,
) -> tuple[dict[str, Any], list[StepProfilingRow]]:
    rows: list[StepProfilingRow] = []
    samples: list[dict[str, Any]] = []
    for sample, result in records:
        sample_summary, sample_rows = build_step_profiling(
            dataset_name=dataset_name,
            sample=sample,
            result=result,
            model_name=model_name,
            config_name=config_name,
        )
        if sample_rows:
            rows.extend(sample_rows)
            samples.append({"sample_id": sample.sample_id, **sample_summary})

    total_time = sum(row.time_seconds or 0.0 for row in rows)
    compute_values = [row.compute_tflops for row in rows]
    compute_complete = bool(rows) and all(value is not None for value in compute_values)
    total_compute = (
        sum(float(value) for value in compute_values if value is not None)
        if compute_complete
        else None
    )
    total_accepted = sum(row.accepted_tokens or 0 for row in rows)

    phases: dict[str, dict[str, float | int | None]] = {}
    for row in rows:
        phase = phases.setdefault(
            row.phase,
            {"forwards": 0, "time_seconds": 0.0, "compute_tflops": 0.0},
        )
        phase["forwards"] = int(phase["forwards"] or 0) + 1
        phase["time_seconds"] = float(phase["time_seconds"] or 0.0) + float(
            row.time_seconds or 0.0
        )
        if row.compute_tflops is None:
            phase["compute_tflops"] = None
        elif phase["compute_tflops"] is not None:
            phase["compute_tflops"] = float(phase["compute_tflops"]) + float(
                row.compute_tflops
            )

    for phase in phases.values():
        phase["time_share"] = (
            float(phase["time_seconds"] or 0.0) / total_time
            if total_time > 0
            else None
        )
        phase_compute = phase["compute_tflops"]
        phase["compute_share"] = (
            float(phase_compute) / total_compute
            if phase_compute is not None and total_compute
            else None
        )

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "config": config_name,
        "measurement_status": "complete" if rows else "unavailable",
        "selected_samples": len(records),
        "profiled_samples": len(samples),
        "forward_count": len(rows),
        "time_seconds": total_time if rows else None,
        "compute_tflops": total_compute,
        "accepted_tokens": total_accepted if rows else None,
        "time_per_accepted_token": (
            total_time / total_accepted if total_accepted > 0 else None
        ),
        "compute_per_accepted_token": (
            total_compute / total_accepted
            if total_compute is not None and total_accepted > 0
            else None
        ),
        "phase_contribution": phases,
        "samples": samples,
    }
    return summary, rows


def render_dataset_profiling_report(
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    out_dir: str | Path,
    *,
    model_name: str | None = None,
    config_name: str | None = None,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_path = out / "dataset_step_profiling.png"

    _, rows = build_dataset_profiling_summary(
        dataset_name,
        records,
        model_name=model_name,
        config_name=config_name,
    )
    if not rows:
        plot_path.unlink(missing_ok=True)
        return {}
    written: dict[str, str] = {}
    if plot_step_profiling(rows, plot_path):
        written["step_profiling_plot"] = str(plot_path)
    return written
