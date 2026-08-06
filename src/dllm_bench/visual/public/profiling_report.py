"""Dataset-level reports for per-step and real-stage profiling metrics."""

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
            if elapsed is None:
                stage["time_seconds"] = None
            elif stage["time_seconds"] is not None:
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
    times = [
        float(stages[name]["time_seconds"])
        if stages[name]["time_seconds"] is not None
        else float("nan")
        for name in names
    ]
    compute = [
        float(stages[name]["compute_tflops"])
        if stages[name].get("compute_tflops") is not None
        else float("nan")
        for name in names
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15.5, max(4.8, len(names) * 0.48)))
    axes[0].barh(names, times)
    axes[0].set(title="Measured time by generation stage", xlabel="Seconds")
    axes[1].barh(names, compute)
    axes[1].set(title="Compute by generation stage", xlabel="TFLOP")
    figure.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.90,
        bottom=0.14,
        wspace=0.38,
    )
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

    time_complete = bool(rows) and all(
        row.time_seconds is not None for row in rows
    )
    total_time = (
        sum(float(row.time_seconds) for row in rows)
        if time_complete
        else None
    )
    compute_values = [row.compute_tflops for row in rows]
    compute_complete = bool(rows) and all(value is not None for value in compute_values)
    total_compute = (
        sum(float(value) for value in compute_values if value is not None)
        if compute_complete
        else None
    )
    productive_rows = [
        row
        for row in rows
        if row.phase not in {"prefill", "prefill_or_cache_build", "finalization"}
    ]
    acceptance_complete = bool(productive_rows) and all(
        row.accepted_tokens is not None for row in productive_rows
    )
    total_accepted = (
        sum(row.accepted_tokens or 0 for row in rows)
        if acceptance_complete
        else None
    )

    phases: dict[str, dict[str, float | int | None]] = {}
    for row in rows:
        phase = phases.setdefault(
            row.phase,
            {"steps": 0, "time_seconds": 0.0, "compute_tflops": 0.0},
        )
        phase["steps"] = int(phase["steps"] or 0) + 1
        if row.time_seconds is None:
            phase["time_seconds"] = None
        elif phase["time_seconds"] is not None:
            phase["time_seconds"] = float(phase["time_seconds"]) + float(
                row.time_seconds
            )
        if row.compute_tflops is None:
            phase["compute_tflops"] = None
        elif phase["compute_tflops"] is not None:
            phase["compute_tflops"] = float(phase["compute_tflops"]) + float(
                row.compute_tflops
            )

    for phase in phases.values():
        phase["time_share"] = (
            float(phase["time_seconds"]) / total_time
            if phase["time_seconds"] is not None and total_time
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
        "measurement_status": (
            "complete"
            if rows and time_complete and compute_complete and acceptance_complete
            else "partial" if rows else "unavailable"
        ),
        "time_status": "complete" if time_complete else "unavailable",
        "compute_status": "complete" if compute_complete else "unavailable",
        "acceptance_status": (
            "complete" if acceptance_complete else "unavailable"
        ),
        "selected_samples": len(records),
        "profiled_samples": len(samples),
        "step_count": len(rows),
        "time_seconds": total_time,
        "compute_tflops": total_compute,
        "accepted_tokens": total_accepted,
        "time_per_accepted_token": (
            total_time / total_accepted
            if total_time is not None
            and total_accepted is not None
            and total_accepted > 0
            else None
        ),
        "accepted_token_tps": (
            total_accepted / total_time
            if total_time is not None
            and total_time > 0
            and total_accepted is not None
            and total_accepted > 0
            else None
        ),
        "compute_per_accepted_token": (
            total_compute / total_accepted
            if total_compute is not None
            and total_accepted is not None
            and total_accepted > 0
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
