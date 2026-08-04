"""Dataset-level reports for per-forward and real-stage profiling metrics."""

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


def build_stage_profiling_summary(
    records: list[tuple[Sample, GenerationResult]],
) -> dict[str, dict[str, float | int | None]]:
    stages: dict[str, dict[str, float | int | None]] = {}
    for _, result in records:
        for profile in result.extra.get("stage_profiles", []):
            name = str(profile["stage"])
            stage = stages.setdefault(
                name,
                {"calls": 0, "time_seconds": 0.0, "compute_flops": 0},
            )
            stage["calls"] = int(stage["calls"] or 0) + 1
            elapsed = profile.get("wall_clock_seconds")
            if elapsed is not None:
                stage["time_seconds"] = float(stage["time_seconds"] or 0.0) + float(elapsed)
            raw_flops = profile.get("compute_flops")
            if raw_flops is None:
                stage["compute_flops"] = None
            elif stage["compute_flops"] is not None:
                stage["compute_flops"] = int(stage["compute_flops"] or 0) + int(raw_flops)
    for stage in stages.values():
        raw_flops = stage["compute_flops"]
        stage["compute_tflops"] = (
            int(raw_flops) / 1e12 if raw_flops is not None else None
        )
    return stages


def plot_stage_profiling(
    stages: dict[str, dict[str, float | int | None]], path: str | Path
) -> bool:
    output = Path(path)
    if not stages:
        output.unlink(missing_ok=True)
        return False
    import matplotlib.pyplot as plt

    names = list(stages)
    times = [float(stages[name]["time_seconds"] or 0.0) for name in names]
    compute = [float(stages[name].get("compute_tflops") or 0.0) for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(14, max(4, len(names) * 0.42)))
    axes[0].barh(names, times)
    axes[0].set(title="Measured time by generation stage", xlabel="Seconds")
    axes[1].barh(names, compute)
    axes[1].set(title="Compute by generation stage", xlabel="TFLOP")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


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
        "stage_contribution": build_stage_profiling_summary(records),
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
    stage_plot_path = out / "dataset_stage_profiling.png"

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
    if plot_stage_profiling(build_stage_profiling_summary(records), stage_plot_path):
        written["stage_profiling_plot"] = str(stage_plot_path)
    return written
