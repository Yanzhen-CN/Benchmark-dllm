"""Static matplotlib companions to ``token_grid_viz``'s GIF/PNG, adapted from
``Gemma/DGtest/visual.py``'s ``plot_position_step_figures``/``plot_speed_figures``:

- token position vs first-committed forward step, colored by how many times
  that position was ultimately revised (same gradient as the token grid)
- commit activity per forward step (raw commits, and cumulative unique
  positions committed) — a "how busy is each step" view, complementary to
  :mod:`dllm_bench.metrics.trace_parallelism`'s Effective-Tokens-per-Forward
  (which counts a position only at its *final* stable step, not every commit).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

from ..interfaces import TraceStep
from .token_grid_viz import (
    compute_running_accept_counts,
    count_ticks,
    matplotlib_gradient_cmap,
    meaningful_committed_positions,
)


def _draw_block_boundaries(
    ax,
    *,
    n_positions: int,
    block_length: int | None,
) -> None:
    """Separate block-diffusion canvases without overwhelming long plots."""
    if not block_length or block_length <= 0 or block_length >= n_positions:
        return
    n_blocks = (n_positions + block_length - 1) // block_length
    for block_index in range(n_blocks):
        start = block_index * block_length
        end = min((block_index + 1) * block_length, n_positions)
        if block_index % 2:
            ax.axvspan(start - 0.5, end - 0.5, color="#607d8b", alpha=0.045, zorder=0)
        if block_index:
            ax.axvline(start - 0.5, color="#7d8790", linestyle="--", linewidth=0.8, alpha=0.72)

    # Keep at most about 16 labels on very long 2K/4K canvases. Every block
    # boundary remains visible even when only a subset receives a label.
    label_stride = max(1, math.ceil(n_blocks / 16))
    for block_index in range(0, n_blocks, label_stride):
        start = block_index * block_length
        end = min((block_index + 1) * block_length, n_positions)
        ax.text(
            (start + end - 1) / 2,
            0.985,
            f"B{block_index}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
            color="#58616a",
            clip_on=True,
        )


def _revision_count_cmap() -> LinearSegmentedColormap:
    """Low update count is blue-green; repeated revisions progress to red."""
    return LinearSegmentedColormap.from_list(
        "dllm_bench_revision_count",
        [
            (45 / 255, 165 / 255, 170 / 255),
            (86 / 255, 190 / 255, 120 / 255),
            (232 / 255, 205 / 255, 75 / 255),
            (240 / 255, 128 / 255, 55 / 255),
            (205 / 255, 45 / 255, 45 / 255),
            (110 / 255, 18 / 255, 35 / 255),
        ],
        N=256,
    )


def plot_position_vs_first_commit(
    trace: list[TraceStep], out_path: str | Path, title: str = ""
) -> None:
    if not trace:
        return
    accept_counts_by_step = compute_running_accept_counts(trace)
    final_counts = accept_counts_by_step[-1]
    n_positions = len(trace[-1].position_states)

    first_step: dict[int, int] = {}
    for index, step in enumerate(trace):
        for position in meaningful_committed_positions(trace, index):
            first_step.setdefault(position, step.forward_index)

    positions = sorted(first_step)
    steps = [first_step[p] for p in positions]
    counts = [final_counts.get(p, 0) for p in positions]

    max_count = max(counts, default=1) or 1
    cmap = matplotlib_gradient_cmap()
    norm = LogNorm(vmin=1, vmax=max_count) if max_count > 1 else Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mappable = ax.scatter(
        positions, steps,
        c=[max(c, 1) for c in counts] if max_count > 1 else ["0.35"] * len(counts),
        cmap=cmap if max_count > 1 else None,
        norm=norm if max_count > 1 else None,
        s=24, alpha=0.9, linewidths=0,
    )
    ax.set_xlim(-1, n_positions)
    ax.set_ylim(-1, len(trace))
    ax.set_xlabel("Token position")
    ax.set_ylabel("First committed forward step")
    ax.set_title(title or "Token position vs first commit")
    ax.grid(True, alpha=0.25)

    if max_count > 1:
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Cumulative revisions (log scale)")
        ticks = count_ticks(max_count)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_updates(
    trace: list[TraceStep],
    out_path: str | Path,
    title: str = "",
    block_length: int | None = None,
) -> None:
    """DGtest-style position vs every accepted/revision step."""
    if not trace:
        return
    counts: dict[int, int] = {}
    positions: list[int] = []
    steps: list[int] = []
    ranks: list[int] = []
    for step_index, step in enumerate(trace):
        for position in sorted(meaningful_committed_positions(trace, step_index)):
            rank = counts.get(position, 0) + 1
            counts[position] = rank
            positions.append(position)
            steps.append(step.forward_index)
            ranks.append(rank)
    if not positions:
        return

    max_rank = max(ranks)
    cmap = _revision_count_cmap()
    norm = LogNorm(vmin=1, vmax=max_rank) if max_rank > 1 else Normalize(vmin=0, vmax=1)
    colors = ranks if max_rank > 1 else [0.0] * len(ranks)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    mappable = ax.scatter(
        positions,
        steps,
        c=colors,
        cmap=cmap,
        norm=norm,
        s=[16 + 3 * min(rank - 1, 8) for rank in ranks],
        alpha=0.96,
        linewidths=0,
    )
    n_positions = max(len(step.position_states) for step in trace)
    ax.set_xlim(-1, n_positions)
    ax.set_ylim(-1, max(step.forward_index for step in trace) + 1)
    ax.set_xlabel("Global token position")
    if max_rank > 1:
        ax.set_ylabel("Accepted / revision step")
        plot_description = "token position vs every accepted/revision step"
    else:
        # Commitment-only adapters expose each final token once. Calling these
        # events "revisions" would overstate what the trace can observe.
        ax.set_ylabel("First accepted step")
        plot_description = "token position vs first accepted step"
    ax.set_title(f"{title + ': ' if title else ''}{plot_description}")
    ax.grid(True, alpha=0.22)
    _draw_block_boundaries(
        ax,
        n_positions=n_positions,
        block_length=block_length,
    )
    if max_rank > 1:
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
        colorbar.set_label("Cumulative acceptance / revision count (log scale)")
        ticks = count_ticks(max_rank)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([str(tick) for tick in ticks])
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_token_entropy_heatmap(
    trace: list[TraceStep],
    out_path: str | Path,
    title: str = "",
    block_length: int | None = None,
) -> None:
    """Token entropy on a deliberately separate yellow-to-red color scale."""
    if not trace:
        return
    n_positions = max(len(step.position_states) for step in trace)
    values = np.full((len(trace), n_positions), np.nan, dtype=float)
    for row, step in enumerate(trace):
        for raw_position, entropy in (step.entropy_by_position or {}).items():
            position = int(raw_position)
            if 0 <= position < n_positions:
                values[row, position] = float(entropy)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return

    # Spectral_r maps high entropy to red/orange and low entropy to blue/green,
    # matching the visual direction of an entropy-decay process.
    cmap = plt.get_cmap("Spectral_r").with_extremes(bad="#eeeeee")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.max(finite)),
    )
    ax.set_xlabel("Global token position")
    ax.set_ylabel("Forward step")
    ax.set_title(f"{title + ': ' if title else ''}token entropy")
    _draw_block_boundaries(
        ax,
        n_positions=n_positions,
        block_length=block_length,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Token entropy")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_commit_speed(trace: list[TraceStep], out_path: str | Path, title: str = "") -> None:
    if not trace:
        return
    committed_per_step = [
        len(meaningful_committed_positions(trace, index))
        for index in range(len(trace))
    ]
    seen: set[int] = set()
    cumulative_unique = []
    for index, step in enumerate(trace):
        seen.update(meaningful_committed_positions(trace, index))
        cumulative_unique.append(len(seen))

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(range(len(committed_per_step)), committed_per_step, marker="o", markersize=3, linewidth=1.3)
    axes[0].set_title("Committed positions per forward step")
    axes[0].set_ylabel("Committed this step")
    axes[1].plot(range(len(cumulative_unique)), cumulative_unique, marker="o", markersize=3, linewidth=1.3)
    axes[1].set_title("Cumulative unique committed positions")
    axes[1].set_ylabel("Cumulative committed")
    axes[1].set_xlabel("Forward step")
    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle(title or "Commit speed")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
