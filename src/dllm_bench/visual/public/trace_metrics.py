"""Canonical model-agnostic trace and profiling metrics.

This module only reports measurements already present in generation artifacts:
whole-generation time, whole-generation FLOPs, and native model traces. It does
not distribute aggregate time or FLOPs across denoising steps.
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
