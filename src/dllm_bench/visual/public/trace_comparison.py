"""Shared cross-variant trace selection, metrics, figures, and audit files."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ...datasets.base import Sample
from ...interfaces import GenerationResult, PositionState
from .style import (
    BACKGROUND,
    MUTED,
    STATE_COLORS,
    place_figure_legend,
    variant_color,
)
from .trace_metrics import (
    TraceMetricRow,
    build_trace_step_rows,
    summarize_profiling,
)

Record = tuple[Sample, GenerationResult]


def select_common_sample(
    records_by_variant: dict[str, list[Record]],
) -> tuple[str, dict[str, Record]] | None:
    """Pick one deterministic, median-length sample present in every variant."""
    indexed: dict[str, dict[str, Record]] = {
        variant: {
            sample.sample_id: (sample, result)
            for sample, result in records
            if result.trace and result.final_valid_length > 0
        }
        for variant, records in records_by_variant.items()
        if records
    }
    if len(indexed) < 2:
        return None
    common = set.intersection(*(set(records) for records in indexed.values()))
    if not common:
        return None

    mean_lengths = {
        sample_id: sum(
            indexed[variant][sample_id][1].final_valid_length
            for variant in indexed
        )
        / len(indexed)
        for sample_id in common
    }
    target = median(mean_lengths.values())
    sample_id = min(common, key=lambda key: (abs(mean_lengths[key] - target), key))
    return sample_id, {
        variant: records[sample_id] for variant, records in indexed.items()
    }


def _state_code(state: PositionState) -> int:
    if state == PositionState.MASKED:
        return 0
    if state == PositionState.VISIBLE:
        return 1
    return 2


def _plot_position_state(
    *,
    dataset_name: str,
    sample_id: str,
    selected: dict[str, Record],
    path: Path,
) -> None:
    variants = list(selected)
    all_steps = [
        int(step.forward_index)
        for _, result in selected.values()
        for step in result.trace
    ]
    min_step, max_step = min(all_steps), max(all_steps)
    max_positions = max(
        min(result.final_valid_length, len(result.trace[-1].position_states))
        for _, result in selected.values()
    )
    ncols = min(4, max(1, len(variants)))
    nrows = math.ceil(len(variants) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, max(3.6, 3.15 * nrows)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    cmap = ListedColormap(
        [STATE_COLORS["masked"], STATE_COLORS["visible"], STATE_COLORS["accepted"]]
    )
    cmap.set_bad(BACKGROUND)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    image = None
    for ax, variant in zip(axes.flat, variants):
        _, result = selected[variant]
        matrix = np.full((max_step - min_step + 1, max_positions), np.nan)
        valid_length = min(result.final_valid_length, max_positions)
        for step in result.trace:
            row = int(step.forward_index) - min_step
            for position, state in enumerate(step.position_states[:valid_length]):
                matrix[row, position] = _state_code(state)
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=(-0.5, max_positions - 0.5, min_step - 0.5, max_step + 0.5),
        )
        stop = int(result.trace[-1].forward_index)
        ax.set_title(f"{variant} | stop step {stop}", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(False)
    for ax in axes.flat[len(variants):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Final-output token position")
    for ax in axes[:, 0]:
        ax.set_ylabel("Forward step (actual)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes.flat[: len(variants)]), shrink=0.78, pad=0.015)
        colorbar.set_ticks([0, 1, 2], labels=["masked", "visible", "accepted"])
    fig.suptitle(f"{dataset_name} | {sample_id} | token state by real forward step", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _trace_event_points(
    result: GenerationResult,
) -> tuple[dict[int, int], list[tuple[int, int, int]], bool]:
    """Return first accepts and changed-token re-accepts; exclude re-noising."""
    valid_length = result.final_valid_length
    first_commits: dict[int, int] = {}
    revisions: list[tuple[int, int, int]] = []
    revision_rank: Counter[int] = Counter()
    previous_committed_tokens: dict[int, int] = {}
    saw_commit_identity = False

    for step in result.trace:
        tokens = list((getattr(step, "token_ids", None) or [])[:valid_length])
        committed = sorted(set(int(value) for value in step.committed_positions))
        saw_commit_identity = saw_commit_identity or bool(tokens and committed)
        for position in committed:
            if position < 0 or position >= valid_length or position >= len(tokens):
                continue
            token = tokens[position]
            previous = previous_committed_tokens.get(position)
            if previous is None:
                first_commits[position] = int(step.forward_index)
            elif previous != token:
                revision_rank[position] += 1
                revisions.append(
                    (position, int(step.forward_index), revision_rank[position])
                )
            previous_committed_tokens[position] = token

    return first_commits, revisions, saw_commit_identity


def _event_figure_axes(
    variants: list[str],
    *,
    sharey: bool = True,
) -> tuple[Any, Any]:
    ncols = min(4, max(1, len(variants)))
    nrows = math.ceil(len(variants) / ncols)
    return plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, max(3.6, 3.25 * nrows)),
        sharex=True,
        sharey=sharey,
        squeeze=False,
        constrained_layout=True,
    )


def _plot_first_commit(
    *,
    dataset_name: str,
    sample_id: str,
    selected: dict[str, Record],
    path: Path,
) -> None:
    variants = list(selected)
    fig, axes = _event_figure_axes(variants)
    max_position = max(result.final_valid_length for _, result in selected.values())
    all_steps = [
        int(step.forward_index)
        for _, result in selected.values()
        for step in result.trace
    ]
    for ax, variant in zip(axes.flat, variants):
        _, result = selected[variant]
        first_commits, _, _ = _trace_event_points(result)
        if first_commits:
            ax.scatter(
                list(first_commits),
                list(first_commits.values()),
                color="#2b6cb0",
                s=17,
                alpha=0.88,
                linewidths=0,
            )
        else:
            ax.text(0.5, 0.5, "No commit events", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        ax.set_title(variant, fontsize=10, weight="bold")
        ax.set_xlim(-1, max_position)
        ax.set_ylim(min(all_steps) - 0.5, max(all_steps) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(True, alpha=0.20)
    for ax in axes.flat[len(variants):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Final-output token position")
    for ax in axes[:, 0]:
        ax.set_ylabel("First commit forward")
    fig.suptitle(
        f"{dataset_name} | {sample_id} | token position vs first commit",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_updates(
    *,
    dataset_name: str,
    sample_id: str,
    selected: dict[str, Record],
    path: Path,
) -> None:
    variants = list(selected)
    event_sets = {
        variant: _trace_event_points(result)
        for variant, (_, result) in selected.items()
    }
    max_position = max(result.final_valid_length for _, result in selected.values())
    all_steps = [
        int(step.forward_index)
        for _, result in selected.values()
        for step in result.trace
    ]
    max_rank = max(
        (rank for _, revisions, _ in event_sets.values() for _, _, rank in revisions),
        default=1,
    )
    rank_norm = LogNorm(vmin=1, vmax=max_rank) if max_rank > 1 else Normalize(vmin=0.5, vmax=1.5)
    fig, axes = _event_figure_axes(variants)
    revision_artist = None
    for ax, variant in zip(axes.flat, variants):
        first_commits, revisions, observable = event_sets[variant]
        if first_commits:
            ax.scatter(
                list(first_commits),
                list(first_commits.values()),
                s=18,
                facecolors="none",
                edgecolors="#2b6cb0",
                linewidths=0.8,
                alpha=0.72,
            )
        if revisions:
            revision_artist = ax.scatter(
                [position for position, _, _ in revisions],
                [step for _, step, _ in revisions],
                c=[rank for _, _, rank in revisions],
                cmap="viridis_r",
                norm=rank_norm,
                s=17,
                alpha=0.82,
                linewidths=0,
            )
        elif not observable:
            ax.text(0.5, 0.5, "Revision N/A\nno accepted-token identity trace", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        else:
            ax.text(0.5, 0.5, "0 observed revisions", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        ax.set_title(variant, fontsize=10, weight="bold")
        ax.set_xlim(-1, max_position)
        ax.set_ylim(min(all_steps) - 0.5, max(all_steps) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(True, alpha=0.20)
    for ax in axes.flat[len(variants):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Final-output token position")
    for ax in axes[:, 0]:
        ax.set_ylabel("Real forward step")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#2b6cb0", label="first accept"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#6d28d9", markeredgecolor="none", label="changed-token re-accept"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    if revision_artist is not None:
        colorbar = fig.colorbar(revision_artist, ax=list(axes.flat[: len(variants)]), shrink=0.76, pad=0.012)
        colorbar.set_label("Cumulative changed-token re-accepts at this position")
    fig.suptitle(
        f"{dataset_name} | {sample_id} | accepted-mask events\n"
        "Hollow: first accept | Filled: changed-token re-accept | Re-noise excluded",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_step_events(
    *,
    dataset_name: str,
    sample_id: str,
    rows_by_variant: dict[str, list[TraceMetricRow]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True, constrained_layout=True)
    for index, (variant, rows) in enumerate(rows_by_variant.items()):
        color = variant_color(index)
        steps = [row.forward_step for row in rows]
        accepted = [row.accepted_tokens for row in rows]
        revisions = [row.revision_events for row in rows]
        if any(value is not None for value in accepted):
            axes[0].plot(steps, accepted, color=color, linewidth=1.8, marker="o", markersize=3, label=variant)
        if any(value is not None for value in revisions):
            axes[1].plot(steps, revisions, color=color, linewidth=1.8, marker="o", markersize=3, label=variant)
    axes[0].set_title("Accepted-mask events per real forward")
    axes[1].set_title("Changed-token re-accepts per real forward")
    if not axes[1].lines:
        axes[1].text(0.5, 0.5, "N/A\nno accepted-token identity trace", ha="center", va="center", transform=axes[1].transAxes, color=MUTED)
    for ax in axes:
        ax.set_xlabel("Forward step (actual)")
        ax.set_ylabel("Tokens")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    place_figure_legend(fig, handles, labels)
    fig.suptitle(f"{dataset_name} | {sample_id} | absolute events by real forward", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _answer_cell_position_map(
    result: GenerationResult,
    *,
    cell_count: int,
    max_digit: int,
) -> dict[int, list[int]]:
    """Map the strongest final digit run's token positions to Sudoku cells."""
    if not result.trace or not result.trace[-1].token_texts:
        return {}
    allowed = set("123456789"[:max_digit])
    runs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for position, raw_text in enumerate(result.trace[-1].token_texts):
        text = raw_text.strip()
        if text and all(char in allowed for char in text):
            current.append((position, text))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return {}
    run = max(
        runs,
        key=lambda value: (sum(len(text) for _, text in value), -value[0][0]),
    )
    mapping: dict[int, list[int]] = {}
    cell_index = 0
    for position, text in run:
        for _ in text:
            if cell_index >= cell_count:
                return mapping
            mapping.setdefault(position, []).append(cell_index)
            cell_index += 1
    return mapping


def _plot_answer_trace(
    *,
    dataset_name: str,
    sample_id: str,
    selected: dict[str, tuple[object, GenerationResult]],
    path: Path,
) -> None:
    size = 4 if dataset_name.startswith("sudoku4") else 9
    cell_count = size * size
    variants = list(selected)
    event_sets = {
        variant: _trace_event_points(result)
        for variant, (_, result) in selected.items()
    }
    all_steps = [
        int(step.forward_index)
        for _, result in selected.values()
        for step in result.trace
    ] or [0]
    mapped_sets: dict[
        str,
        tuple[list[tuple[int, int]], list[tuple[int, int, int]], int, bool],
    ] = {}
    max_rank = 1
    for variant, (_, result) in selected.items():
        first_accepts, revisions, observable = event_sets[variant]
        mapping = _answer_cell_position_map(
            result,
            cell_count=cell_count,
            max_digit=size,
        )
        first_cells = [
            (cell, step)
            for position, step in first_accepts.items()
            for cell in mapping.get(position, [])
        ]
        revision_cells = [
            (cell, step, rank)
            for position, step, rank in revisions
            for cell in mapping.get(position, [])
        ]
        max_rank = max(max_rank, *(rank for _, _, rank in revision_cells), 1)
        mapped_count = len({cell for cells in mapping.values() for cell in cells})
        mapped_sets[variant] = (
            first_cells,
            revision_cells,
            mapped_count,
            observable,
        )

    rank_norm = (
        LogNorm(vmin=1, vmax=max_rank)
        if max_rank > 1
        else Normalize(vmin=0.5, vmax=1.5)
    )
    fig, axes = _event_figure_axes(variants)
    revision_artist = None
    for ax, variant in zip(axes.flat, variants):
        first_cells, revision_cells, mapped_count, observable = mapped_sets[variant]
        if first_cells:
            ax.scatter(
                [cell for cell, _ in first_cells],
                [step for _, step in first_cells],
                s=18,
                facecolors="none",
                edgecolors="#2b6cb0",
                linewidths=0.8,
                alpha=0.72,
            )
        if revision_cells:
            revision_artist = ax.scatter(
                [cell for cell, _, _ in revision_cells],
                [step for _, step, _ in revision_cells],
                c=[rank for _, _, rank in revision_cells],
                cmap="viridis_r",
                norm=rank_norm,
                s=17,
                alpha=0.82,
                linewidths=0,
            )
        elif not observable:
            ax.text(
                0.5,
                0.5,
                "Revision N/A\nno accepted-token identity trace",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=MUTED,
            )
        if mapped_count == 0:
            ax.text(
                0.5,
                0.16,
                "No final digit run mapped",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=MUTED,
            )
        else:
            ax.text(
                0.98,
                0.04,
                f"mapped {mapped_count}/{cell_count}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                color=MUTED,
                fontsize=8,
            )
        ax.set_title(variant, fontsize=10, weight="bold")
        ax.set_xlim(-1, cell_count)
        ax.set_ylim(min(all_steps) - 0.5, max(all_steps) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.set_xlabel("Sudoku answer cell")
        ax.set_ylabel("Real forward step")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#2b6cb0",
            label="first accept",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#6d28d9",
            markeredgecolor="none",
            label="changed-token re-accept",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    if revision_artist is not None:
        colorbar = fig.colorbar(
            revision_artist,
            ax=list(axes.flat[: len(variants)]),
            shrink=0.76,
            pad=0.012,
        )
        colorbar.set_label("Cumulative changed-token re-accepts")
    fig.suptitle(
        f"{dataset_name} | {sample_id} | Sudoku answer trace only",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_rows(rows_by_variant: dict[str, list[TraceMetricRow]], path: Path) -> None:
    rows = [row.to_dict() for variant_rows in rows_by_variant.values() for row in variant_rows]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_trace_comparison(
    *,
    model_name: str,
    dataset_name: str,
    records_by_variant: dict[str, list[Record]],
    out_dir: str | Path,
    block_length: int | None = None,
    figures: set[str] | None = None,
) -> dict[str, str]:
    """Render every benchmark-common cross-variant trace artifact."""
    figures = figures or {"all"}
    allowed = {"all", "trace", "state", "convergence", "yield"}
    unsupported = figures.difference(allowed)
    if unsupported:
        requested = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported public trace figure(s): {requested}")
    render_trace = "all" in figures or "trace" in figures
    selection = select_common_sample(records_by_variant)
    if selection is None:
        return {}
    sample_id, selected = selection
    rows_by_variant = {
        variant: build_trace_step_rows(
            model=model_name,
            config=variant,
            sample_id=sample_id,
            result=result,
        )
        for variant, (_, result) in selected.items()
    }
    rows_by_variant = {variant: rows for variant, rows in rows_by_variant.items() if rows}
    if len(rows_by_variant) < 2:
        return {}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in (
        "trace_comparison.png",
        "trace_comparison.csv",
        "trace_convergence_comparison.png",
        "trace_step_yield.png",
        "entropy_by_real_forward_step.png",
        "entropy_by_real_forward_step.csv",
        "position_step_entropy.png",
        "block_local_tau_comparison.png",
    ):
        (out / stale).unlink(missing_ok=True)

    state_path = out / "trace_position_state.png"
    for stale in (
        "trace_first_commit.png",
        "trace_first_accept.png",
        "trace_updates.png",
        "trace_acceptance_events.png",
    ):
        (out / stale).unlink(missing_ok=True)
    updates_path = out / "accept_trace.png"
    answer_path = out / "answer_trace.png"
    step_events_path = out / "trace_step_events.png"
    csv_path = out / "trace_stepwise.csv"
    metadata_path = out / "trace_metadata.json"
    block_tau_path = out / "block_local_tau_comparison.png"
    if render_trace and block_length:
        from .dataset_trace_report import build_dataset_trace_summary
        from .plots import plot_task4_block_local_tau

        tau_rows: list[dict[str, Any]] = []
        for variant, records in records_by_variant.items():
            trace_summary, _ = build_dataset_trace_summary(
                dataset_name,
                records,
                model_name=model_name,
                config_name=variant,
                block_length=block_length,
            )
            tau_rows.append(
                {
                    "Model": model_name,
                    "Config": variant,
                    "N": len(records),
                    "Trace Summary": trace_summary,
                }
            )
        plot_task4_block_local_tau(tau_rows, str(block_tau_path))
    if render_trace or "state" in figures:
        _plot_position_state(
            dataset_name=dataset_name,
            sample_id=sample_id,
            selected={variant: selected[variant] for variant in rows_by_variant},
            path=state_path,
        )
    if render_trace or "convergence" in figures:
        _plot_updates(
            dataset_name=dataset_name,
            sample_id=sample_id,
            selected={variant: selected[variant] for variant in rows_by_variant},
            path=updates_path,
        )
        if dataset_name.startswith("sudoku"):
            _plot_answer_trace(
                dataset_name=dataset_name,
                sample_id=sample_id,
                selected={variant: selected[variant] for variant in rows_by_variant},
                path=answer_path,
            )
    if render_trace or "yield" in figures:
        _plot_step_events(
            dataset_name=dataset_name,
            sample_id=sample_id,
            rows_by_variant=rows_by_variant,
            path=step_events_path,
        )
    _write_rows(rows_by_variant, csv_path)

    metadata: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "selected_common_sample": sample_id,
        "sample_selection": "common traced sample nearest the median cross-variant final-output length",
        "step_axis": "actual integer TraceStep.forward_index; no progress normalization",
        "short_trace_policy": "stop at the observed final step and leave later shared-axis rows blank",
        "state_semantics": STATE_COLORS,
        "display_semantics": {
            "trace_position_state": "small multiples of masked, visible, and accepted states by real forward",
            "accept_trace": "first accepts plus changed-token re-acceptance events; same-token re-acceptance, re-noising, and visible proposal refresh are excluded",
            "answer_trace": "Sudoku-only projection of accept_trace onto final answer cells; long digit runs are clipped to the first size-squared cells",
            "trace_step_events": "absolute accepted-mask and changed-token re-accept counts; no normalized progress axis",
        },
        "definitions": {
            "accepted_tokens": "positions selected by the sampler's accepted_token_mask at this forward",
            "revision_events": "positions accepted again with a token different from their previously accepted token",
            "revision_direction": "helpful moves toward the final generated token; harmful moves away; lateral changes between non-final tokens",
            "final_stable_tokens_gained": "positions whose final token first becomes permanent at this forward",
            "final_stable_fraction": "cumulative final-stable positions divided by final valid generated tokens",
        },
        "variants": {
            variant: {
                "observed_steps": len(rows),
                "first_forward_step": rows[0].forward_step,
                "last_forward_step": rows[-1].forward_step,
                "final_valid_tokens": rows[-1].valid_length,
                "profiling": summarize_profiling(
                    rows,
                    total_time_ms=(
                        selected[variant][1].timing.wall_clock_seconds * 1000.0
                        if selected[variant][1].timing
                        and selected[variant][1].timing.wall_clock_seconds > 0
                        else None
                    ),
                    total_flops=(
                        selected[variant][1].compute_tflops * 1e12
                        if selected[variant][1].compute_tflops is not None
                        else None
                    ),
                    total_forward_passes=selected[variant][1].num_forward_passes,
                ),
            }
            for variant, rows in rows_by_variant.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    written = {
        "trace_stepwise": str(csv_path),
        "trace_metadata": str(metadata_path),
    }
    if state_path.exists() and (render_trace or "state" in figures):
        written["trace_position_state"] = str(state_path)
    if updates_path.exists() and (render_trace or "convergence" in figures):
        written["accept_trace"] = str(updates_path)
    if answer_path.exists() and (render_trace or "convergence" in figures):
        written["answer_trace"] = str(answer_path)
    if step_events_path.exists() and (render_trace or "yield" in figures):
        written["trace_step_events"] = str(step_events_path)
    if block_tau_path.exists() and render_trace:
        written["block_local_tau_comparison"] = str(block_tau_path)
    return written
