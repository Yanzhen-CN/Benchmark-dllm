"""Canonical model-agnostic trace and profiling metrics.

This module only reports measurements already present in generation artifacts.
Whole-generation values are never divided across steps. Per-step time and FLOPs
are emitted only when the profiling run captured real inference-step boundaries.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from ...datasets.base import Sample
from ...interfaces import GenerationResult
from ...metrics.stats_utils import summarize
from ...metrics.trace_parallelism import compute_final_stable_steps
from .token_grid_viz import meaningful_committed_positions


@dataclass(frozen=True)
class TraceMetricRow:
    model: str
    config: str
    sample_id: str
    trace_index: int
    forward_step: int
    valid_length: int
    accepted_tokens: int
    revision_events: int | None
    helpful_revision_events: int | None
    harmful_revision_events: int | None
    lateral_revision_events: int | None
    cumulative_revised_positions: int | None
    final_stable_tokens_gained: int
    final_stable_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuxiliaryPerformanceRow:
    model: str
    config: str
    dataset: str
    sample_id: str
    final_valid_tokens: int
    observed_trace_steps: int
    reported_forward_passes: int
    accepted_events: int
    mean_accepted_events_per_step: float
    revision_events: int | None
    mean_revision_events_per_step: float | None
    revised_position_share: float | None
    revisions_per_final_position: float | None
    helpful_revision_share: float | None
    harmful_revision_share: float | None
    lateral_revision_share: float | None
    mean_final_stable_tokens_per_step: float
    step_to_90pct_stable: int
    wall_clock_ms: float | None
    ms_per_forward: float | None
    ms_per_final_token: float | None
    ms_per_accepted_event: float | None
    total_flops: float | None
    flops_per_forward: float | None
    flops_per_final_token: float | None
    flops_per_accepted_event: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepProfilingRow:
    model: str
    config: str
    dataset: str
    sample_id: str
    step_index: int
    phase: str
    time_seconds: float | None
    compute_tflops: float | None
    accepted_tokens: int | None
    active_tokens: int | None
    eligible_tokens: int | None
    input_tokens: int | None
    kv_cache_tokens: int | None
    attention_tokens: int | None
    uses_kv_cache: bool | None
    stores_kv: bool | None
    cumulative_compute_tflops: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _valid_length(result: GenerationResult) -> int:
    if not result.trace:
        return 0
    observed = len(result.trace[-1].token_ids)
    configured = int(result.final_valid_length or observed)
    return max(0, min(configured, observed))


def visible_revision_profile(
    result: GenerationResult, valid_length: int
) -> dict[str, Any] | None:
    """Describe accepted-token identity revisions, excluding re-noising.

    A revision requires the position to be committed again with a token that
    differs from its previously committed token. Adjacent visible-canvas
    changes are deliberately ignored because they include proposal refresh
    and re-noising. Direction labels remain relative to the final generated
    token, not ground truth.
    """
    trace = result.trace
    if not trace or valid_length <= 0 or not any(
        step.committed_positions and step.token_ids for step in trace
    ):
        return None

    final_tokens = trace[-1].token_ids[:valid_length]
    previous_committed_tokens: dict[int, int] = {}
    revised_positions: set[int] = set()
    events_by_step = [0] * len(trace)
    helpful_by_step = [0] * len(trace)
    harmful_by_step = [0] * len(trace)
    lateral_by_step = [0] * len(trace)
    cumulative_positions: list[int] = []

    for step_index, step in enumerate(trace):
        for position in sorted(set(int(value) for value in step.committed_positions)):
            if position < 0 or position >= valid_length or position >= len(step.token_ids):
                continue
            token = step.token_ids[position]
            previous = previous_committed_tokens.get(position)
            if previous is not None and previous != token:
                events_by_step[step_index] += 1
                revised_positions.add(position)
                final_token = final_tokens[position]
                if previous != final_token and token == final_token:
                    helpful_by_step[step_index] += 1
                elif previous == final_token and token != final_token:
                    harmful_by_step[step_index] += 1
                else:
                    lateral_by_step[step_index] += 1
            previous_committed_tokens[position] = token
        cumulative_positions.append(len(revised_positions))

    total_events = sum(events_by_step)
    return {
        "events_by_step": events_by_step,
        "helpful_by_step": helpful_by_step,
        "harmful_by_step": harmful_by_step,
        "lateral_by_step": lateral_by_step,
        "cumulative_revised_positions": cumulative_positions,
        "revision_events": total_events,
        "revised_position_share": len(revised_positions) / valid_length,
        "revisions_per_final_position": total_events / valid_length,
    }


def build_trace_step_rows(
    *,
    model: str,
    config: str,
    sample_id: str,
    result: GenerationResult,
) -> list[TraceMetricRow]:
    """Build rows on the trace's real integer forward-step axis."""
    trace = result.trace
    valid_length = _valid_length(result)
    if not trace or valid_length <= 0:
        return []

    sequences = [step.token_ids[:valid_length] for step in trace]
    stable_indices = compute_final_stable_steps(sequences)
    stable_gains = [0] * len(trace)
    for index in stable_indices:
        stable_gains[index] += 1
    revision_profile = visible_revision_profile(result, valid_length)

    rows: list[TraceMetricRow] = []
    cumulative_stable = 0
    for index, step in enumerate(trace):
        stable_gain = stable_gains[index]
        cumulative_stable += stable_gain
        rows.append(
            TraceMetricRow(
                model=model,
                config=config,
                sample_id=sample_id,
                trace_index=index,
                forward_step=int(step.forward_index),
                valid_length=valid_length,
                accepted_tokens=len(meaningful_committed_positions(trace, index)),
                revision_events=(
                    revision_profile["events_by_step"][index]
                    if revision_profile is not None
                    else None
                ),
                helpful_revision_events=(
                    revision_profile["helpful_by_step"][index]
                    if revision_profile is not None
                    else None
                ),
                harmful_revision_events=(
                    revision_profile["harmful_by_step"][index]
                    if revision_profile is not None
                    else None
                ),
                lateral_revision_events=(
                    revision_profile["lateral_by_step"][index]
                    if revision_profile is not None
                    else None
                ),
                cumulative_revised_positions=(
                    revision_profile["cumulative_revised_positions"][index]
                    if revision_profile is not None
                    else None
                ),
                final_stable_tokens_gained=stable_gain,
                final_stable_fraction=cumulative_stable / valid_length,
            )
        )
    return rows


def summarize_profiling(
    rows: list[TraceMetricRow],
    *,
    total_time_ms: float | None = None,
    total_flops: float | None = None,
    total_forward_passes: int | None = None,
) -> dict[str, Any]:
    """Summarize only whole-generation measurements and native trace events."""
    if not rows:
        return {
            "measurement_status": "unavailable",
            "observed_trace_steps": 0,
        }

    final_tokens = rows[-1].valid_length
    observed_steps = len(rows)
    reported_forwards = (
        int(total_forward_passes)
        if total_forward_passes is not None and total_forward_passes > 0
        else observed_steps
    )
    accepted_events = sum(row.accepted_tokens for row in rows)
    stable_events = sum(row.final_stable_tokens_gained for row in rows)
    revision_rows = [row for row in rows if row.revision_events is not None]
    revision_events = (
        sum(row.revision_events or 0 for row in revision_rows)
        if revision_rows
        else None
    )
    helpful_revisions = (
        sum(row.helpful_revision_events or 0 for row in revision_rows)
        if revision_rows
        else None
    )
    harmful_revisions = (
        sum(row.harmful_revision_events or 0 for row in revision_rows)
        if revision_rows
        else None
    )
    lateral_revisions = (
        sum(row.lateral_revision_events or 0 for row in revision_rows)
        if revision_rows
        else None
    )
    at_90 = next(
        (row for row in rows if row.final_stable_fraction >= 0.90),
        rows[-1],
    )
    clean_time = _finite_number(total_time_ms)
    clean_flops = _finite_number(total_flops)

    return {
        "measurement_status": {
            "trace": "complete",
            "visible_draft_revision": (
                "complete" if revision_rows else "unavailable"
            ),
            "whole_generation_time": (
                "complete" if clean_time is not None else "unavailable"
            ),
            "whole_generation_flops": (
                "complete" if clean_flops is not None else "unavailable"
            ),
        },
        "final_valid_tokens": final_tokens,
        "observed_trace_steps": observed_steps,
        "reported_forward_passes": reported_forwards,
        "accepted_events": accepted_events,
        "mean_accepted_events_per_step": accepted_events / observed_steps,
        "revision_events": revision_events,
        "mean_revision_events_per_step": (
            revision_events / observed_steps
            if revision_events is not None
            else None
        ),
        "revised_position_share": (
            revision_rows[-1].cumulative_revised_positions / final_tokens
            if revision_rows
            and revision_rows[-1].cumulative_revised_positions is not None
            and final_tokens > 0
            else None
        ),
        "revisions_per_final_position": (
            revision_events / final_tokens
            if revision_events is not None and final_tokens > 0
            else None
        ),
        "helpful_revision_share": (
            helpful_revisions / revision_events
            if revision_events is not None and revision_events > 0
            else None
        ),
        "harmful_revision_share": (
            harmful_revisions / revision_events
            if revision_events is not None and revision_events > 0
            else None
        ),
        "lateral_revision_share": (
            lateral_revisions / revision_events
            if revision_events is not None and revision_events > 0
            else None
        ),
        "mean_final_stable_tokens_per_step": stable_events / observed_steps,
        "step_to_90pct_stable": at_90.forward_step,
        "wall_clock_ms": clean_time,
        "ms_per_forward": (
            clean_time / reported_forwards if clean_time is not None else None
        ),
        "ms_per_final_token": (
            clean_time / final_tokens
            if clean_time is not None and final_tokens > 0
            else None
        ),
        "ms_per_accepted_event": (
            clean_time / accepted_events
            if clean_time is not None and accepted_events > 0
            else None
        ),
        "total_flops": clean_flops,
        "flops_per_forward": (
            clean_flops / reported_forwards if clean_flops is not None else None
        ),
        "flops_per_final_token": (
            clean_flops / final_tokens
            if clean_flops is not None and final_tokens > 0
            else None
        ),
        "flops_per_accepted_event": (
            clean_flops / accepted_events
            if clean_flops is not None and accepted_events > 0
            else None
        ),
    }


def build_auxiliary_performance_rows(
    *,
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    model_name: str | None,
    config_name: str | None,
) -> list[AuxiliaryPerformanceRow]:
    output: list[AuxiliaryPerformanceRow] = []
    model = model_name or "unknown"
    config = config_name or "unknown"
    for sample, result in records:
        trace_rows = build_trace_step_rows(
            model=model,
            config=config,
            sample_id=sample.sample_id,
            result=result,
        )
        if not trace_rows:
            continue
        metrics = summarize_profiling(
            trace_rows,
            total_time_ms=(
                result.timing.wall_clock_seconds * 1000.0
                if result.timing and result.timing.wall_clock_seconds > 0
                else None
            ),
            total_flops=(
                float(result.compute_tflops) * 1e12
                if result.compute_tflops is not None and result.compute_tflops >= 0
                else None
            ),
            total_forward_passes=result.num_forward_passes,
        )
        output.append(
            AuxiliaryPerformanceRow(
                model=model,
                config=config,
                dataset=dataset_name,
                sample_id=sample.sample_id,
                final_valid_tokens=metrics["final_valid_tokens"],
                observed_trace_steps=metrics["observed_trace_steps"],
                reported_forward_passes=metrics["reported_forward_passes"],
                accepted_events=metrics["accepted_events"],
                mean_accepted_events_per_step=metrics[
                    "mean_accepted_events_per_step"
                ],
                revision_events=metrics["revision_events"],
                mean_revision_events_per_step=metrics[
                    "mean_revision_events_per_step"
                ],
                revised_position_share=metrics["revised_position_share"],
                revisions_per_final_position=metrics[
                    "revisions_per_final_position"
                ],
                helpful_revision_share=metrics["helpful_revision_share"],
                harmful_revision_share=metrics["harmful_revision_share"],
                lateral_revision_share=metrics["lateral_revision_share"],
                mean_final_stable_tokens_per_step=metrics[
                    "mean_final_stable_tokens_per_step"
                ],
                step_to_90pct_stable=metrics["step_to_90pct_stable"],
                wall_clock_ms=metrics["wall_clock_ms"],
                ms_per_forward=metrics["ms_per_forward"],
                ms_per_final_token=metrics["ms_per_final_token"],
                ms_per_accepted_event=metrics["ms_per_accepted_event"],
                total_flops=metrics["total_flops"],
                flops_per_forward=metrics["flops_per_forward"],
                flops_per_final_token=metrics["flops_per_final_token"],
                flops_per_accepted_event=metrics["flops_per_accepted_event"],
            )
        )
    return output


def _summary(values: list[float | int]) -> dict[str, Any] | None:
    return asdict(summarize([float(value) for value in values])) if values else None


def summarize_auxiliary_performance(
    rows: list[AuxiliaryPerformanceRow],
    *,
    selected_samples: int,
) -> dict[str, Any]:
    timing_rows = [row for row in rows if row.wall_clock_ms is not None]
    compute_rows = [row for row in rows if row.total_flops is not None]
    revision_rows = [row for row in rows if row.revision_events is not None]
    return {
        "selected_samples": selected_samples,
        "trace_eligible_samples": len(rows),
        "trace_eligible_ratio": len(rows) / selected_samples if selected_samples else 0.0,
        "timing_eligible_samples": len(timing_rows),
        "timing_eligible_ratio": (
            len(timing_rows) / selected_samples if selected_samples else 0.0
        ),
        "compute_eligible_samples": len(compute_rows),
        "compute_eligible_ratio": (
            len(compute_rows) / selected_samples if selected_samples else 0.0
        ),
        "mean_accepted_events_per_step": _summary(
            [row.mean_accepted_events_per_step for row in rows]
        ),
        "revision_eligible_samples": len(revision_rows),
        "revision_eligible_ratio": (
            len(revision_rows) / selected_samples if selected_samples else 0.0
        ),
        "mean_revision_events_per_step": _summary(
            [
                row.mean_revision_events_per_step
                for row in revision_rows
                if row.mean_revision_events_per_step is not None
            ]
        ),
        "revised_position_share": _summary(
            [
                row.revised_position_share
                for row in revision_rows
                if row.revised_position_share is not None
            ]
        ),
        "revisions_per_final_position": _summary(
            [
                row.revisions_per_final_position
                for row in revision_rows
                if row.revisions_per_final_position is not None
            ]
        ),
        "helpful_revision_share": _summary(
            [
                row.helpful_revision_share
                for row in revision_rows
                if row.helpful_revision_share is not None
            ]
        ),
        "harmful_revision_share": _summary(
            [
                row.harmful_revision_share
                for row in revision_rows
                if row.harmful_revision_share is not None
            ]
        ),
        "lateral_revision_share": _summary(
            [
                row.lateral_revision_share
                for row in revision_rows
                if row.lateral_revision_share is not None
            ]
        ),
        "mean_final_stable_tokens_per_step": _summary(
            [row.mean_final_stable_tokens_per_step for row in rows]
        ),
        "step_to_90pct_stable": _summary(
            [row.step_to_90pct_stable for row in rows]
        ),
        "ms_per_forward": _summary(
            [row.ms_per_forward for row in timing_rows if row.ms_per_forward is not None]
        ),
        "ms_per_final_token": _summary(
            [
                row.ms_per_final_token
                for row in timing_rows
                if row.ms_per_final_token is not None
            ]
        ),
        "ms_per_accepted_event": _summary(
            [
                row.ms_per_accepted_event
                for row in timing_rows
                if row.ms_per_accepted_event is not None
            ]
        ),
        "flops_per_forward": _summary(
            [
                row.flops_per_forward
                for row in compute_rows
                if row.flops_per_forward is not None
            ]
        ),
        "flops_per_final_token": _summary(
            [
                row.flops_per_final_token
                for row in compute_rows
                if row.flops_per_final_token is not None
            ]
        ),
        "flops_per_accepted_event": _summary(
            [
                row.flops_per_accepted_event
                for row in compute_rows
                if row.flops_per_accepted_event is not None
            ]
        ),
        "compute_scope": "whole-generation FLOP replay; no per-step allocation",
    }


def build_auxiliary_performance_summary(
    *,
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    model_name: str | None,
    config_name: str | None,
) -> tuple[dict[str, Any], list[AuxiliaryPerformanceRow]]:
    rows = build_auxiliary_performance_rows(
        dataset_name=dataset_name,
        records=records,
        model_name=model_name,
        config_name=config_name,
    )
    return summarize_auxiliary_performance(rows, selected_samples=len(records)), rows


def write_auxiliary_performance_csv(
    rows: list[AuxiliaryPerformanceRow], path: str | Path
) -> bool:
    output = Path(path)
    if not rows:
        output.unlink(missing_ok=True)
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [row.to_dict() for row in rows]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    return True


def build_step_profiling(
    *,
    dataset_name: str,
    sample: Sample,
    result: GenerationResult,
    model_name: str | None,
    config_name: str | None,
) -> tuple[dict[str, Any], list[StepProfilingRow]]:
    profiles = result.forward_profiles
    if not profiles:
        return {"measurement_status": "unavailable"}, []

    compute_complete = all(
        profile.compute_tflops is not None for profile in profiles
    )
    time_complete = all(
        profile.wall_clock_seconds is not None for profile in profiles
    )
    productive_profiles = [
        profile
        for profile in profiles
        if profile.phase not in {"prefill", "prefill_or_cache_build", "finalization"}
    ]
    acceptance_complete = bool(productive_profiles) and all(
        profile.accepted_tokens is not None for profile in productive_profiles
    )
    cumulative_compute = 0.0
    rows: list[StepProfilingRow] = []
    for profile in profiles:
        if profile.compute_tflops is not None:
            cumulative_compute += profile.compute_tflops
        rows.append(
            StepProfilingRow(
                model=model_name or "unknown",
                config=config_name or "unknown",
                dataset=dataset_name,
                sample_id=sample.sample_id,
                step_index=profile.forward_index,
                phase=profile.phase,
                time_seconds=profile.wall_clock_seconds,
                compute_tflops=profile.compute_tflops,
                accepted_tokens=profile.accepted_tokens,
                active_tokens=profile.active_tokens,
                eligible_tokens=profile.eligible_tokens,
                input_tokens=profile.input_tokens,
                kv_cache_tokens=profile.kv_cache_tokens,
                attention_tokens=profile.attention_tokens,
                uses_kv_cache=profile.uses_kv_cache,
                stores_kv=profile.stores_kv,
                cumulative_compute_tflops=(
                    cumulative_compute if compute_complete else None
                ),
            )
        )

    accepted = (
        sum(row.accepted_tokens or 0 for row in rows)
        if acceptance_complete
        else None
    )
    total_time = (
        sum(float(row.time_seconds) for row in rows)
        if time_complete
        else None
    )
    total_compute = (
        sum(float(row.compute_tflops) for row in rows)
        if compute_complete
        else None
    )
    phases: dict[str, dict[str, float]] = {}
    for row in rows:
        phase = phases.setdefault(
            row.phase,
            {"time_seconds": 0.0, "compute_tflops": 0.0},
        )
        if row.time_seconds is None:
            phase["time_seconds"] = None
        elif phase["time_seconds"] is not None:
            phase["time_seconds"] += row.time_seconds
        if row.compute_tflops is None:
            phase["compute_tflops"] = None
        elif phase["compute_tflops"] is not None:
            phase["compute_tflops"] += row.compute_tflops
    for values in phases.values():
        values["time_share"] = (
            values["time_seconds"] / total_time
            if values["time_seconds"] is not None and total_time
            else None
        )
        values["compute_share"] = (
            values["compute_tflops"] / total_compute
            if values["compute_tflops"] is not None and total_compute
            else None
        )

    return {
        "measurement_status": (
            "complete"
            if time_complete and compute_complete and acceptance_complete
            else "partial"
        ),
        "time_status": "complete" if time_complete else "unavailable",
        "compute_status": "complete" if compute_complete else "unavailable",
        "acceptance_status": (
            "complete" if acceptance_complete else "unavailable"
        ),
        "profiled_steps": len(rows),
        "time_per_step": [row.time_seconds for row in rows],
        "flops_per_step": [row.compute_tflops for row in rows],
        "accepted_tokens_per_step": [row.accepted_tokens for row in rows],
        "time_per_accepted_token": (
            total_time / accepted
            if total_time is not None and accepted is not None and accepted > 0
            else None
        ),
        "accepted_token_tps": (
            accepted / total_time
            if total_time is not None and total_time > 0
            and accepted is not None and accepted > 0
            else None
        ),
        "compute_per_accepted_token": (
            total_compute / accepted
            if total_compute is not None and accepted is not None and accepted > 0
            else None
        ),
        "cumulative_compute": total_compute,
        "phase_contribution": phases,
        "compute_scope": "model inference steps captured during deterministic FLOP replay",
        "time_scope": "GPU-synchronized model inference steps in the timed profiling run",
    }, rows


def plot_step_profiling(rows: list[StepProfilingRow], path: str | Path) -> bool:
    output = Path(path)
    if not rows:
        output.unlink(missing_ok=True)
        return False
    import matplotlib.pyplot as plt

    x = [row.step_index for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    axes[0, 0].plot(x, [row.time_seconds for row in rows], marker="o", ms=3)
    axes[0, 0].set(title="Time per step", xlabel="Step", ylabel="Seconds")
    axes[0, 1].plot(
        x,
        [row.compute_tflops for row in rows],
        marker="o",
        ms=3,
        label="Per step",
    )
    cumulative_axis = axes[0, 1].twinx()
    cumulative_axis.plot(
        x,
        [row.cumulative_compute_tflops for row in rows],
        color="tab:orange",
        label="Cumulative",
    )
    axes[0, 1].set(title="Compute", xlabel="Step", ylabel="TFLOP / step")
    cumulative_axis.set_ylabel("Cumulative TFLOP")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    extra_handles, extra_labels = cumulative_axis.get_legend_handles_labels()
    axes[0, 1].legend(
        handles + extra_handles,
        labels + extra_labels,
        frameon=False,
        loc="best",
    )
    axes[1, 0].plot(x, [row.accepted_tokens for row in rows])
    axes[1, 0].set(title="Accepted tokens per step", xlabel="Step", ylabel="Tokens")

    accepted = [row.accepted_tokens for row in rows]
    accepted_token_tps = [
        count / row.time_seconds
        if row.time_seconds is not None and row.time_seconds > 0
        and count is not None and count > 0
        else None
        for row, count in zip(rows, accepted)
    ]
    compute_cost = [
        row.compute_tflops / count
        if row.compute_tflops is not None and count is not None and count > 0
        else None
        for row, count in zip(rows, accepted)
    ]
    axes[1, 1].plot(x, accepted_token_tps, label="Accepted-token TPS")
    cost_axis = axes[1, 1].twinx()
    cost_axis.plot(
        x,
        compute_cost,
        color="tab:orange",
        label="TFLOP / accepted token",
    )
    axes[1, 1].set(
        title="Accepted-token throughput and compute cost",
        xlabel="Step",
        ylabel="Accepted tokens / second",
    )
    cost_axis.set_ylabel("TFLOP")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    extra_handles, extra_labels = cost_axis.get_legend_handles_labels()
    axes[1, 1].legend(
        handles + extra_handles,
        labels + extra_labels,
        frameon=False,
        loc="best",
    )

    axes[2, 0].plot(x, [row.input_tokens for row in rows], label="Step input")
    axes[2, 0].plot(x, [row.kv_cache_tokens for row in rows], label="KV cache")
    axes[2, 0].plot(x, [row.attention_tokens for row in rows], label="Attention span")
    axes[2, 0].set(
        title="Input and KV-cache lengths",
        xlabel="Step",
        ylabel="Tokens",
    )
    axes[2, 0].legend(frameon=False, loc="best")

    phases = list(dict.fromkeys(row.phase for row in rows))
    phase_time = [
        sum(
            row.time_seconds
            for row in rows
            if row.phase == phase and row.time_seconds is not None
        )
        if all(
            row.time_seconds is not None
            for row in rows
            if row.phase == phase
        )
        else None
        for phase in phases
    ]
    phase_compute = [
        sum(
            row.compute_tflops
            for row in rows
            if row.phase == phase and row.compute_tflops is not None
        )
        if all(
            row.compute_tflops is not None
            for row in rows
            if row.phase == phase
        )
        else None
        for phase in phases
    ]
    total_time = sum(phase_time) if all(value is not None for value in phase_time) else None
    total_compute = (
        sum(phase_compute) if all(value is not None for value in phase_compute) else None
    )
    positions = list(range(len(phases)))
    width = 0.38
    axes[2, 1].bar(
        [position - width / 2 for position in positions],
        [
            value / total_time
            if value is not None and total_time
            else float("nan")
            for value in phase_time
        ],
        width,
        label="Time share",
    )
    axes[2, 1].bar(
        [position + width / 2 for position in positions],
        [
            value / total_compute
            if value is not None and total_compute
            else float("nan")
            for value in phase_compute
        ],
        width,
        label="Compute share",
    )
    axes[2, 1].set(
        title="Phase contribution",
        ylabel="Share",
        xticks=positions,
        xticklabels=phases,
    )
    axes[2, 1].legend(frameon=False, loc="best")
    fig.suptitle(f"{rows[0].model} / {rows[0].config} / {rows[0].dataset}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True
