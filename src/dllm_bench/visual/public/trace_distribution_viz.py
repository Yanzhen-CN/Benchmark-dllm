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
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

from ...interfaces import TraceStep
from ...trace_events import extract_acceptance_events
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


FIRST_ACCEPTANCE_COLOR = "#2563eb"
REVISION_COLOR = "#dc2626"
REVISION_FINAL_COLOR = "#6d28d9"
REVISION_MARKER_SIZE = 82


def _reaccept_rank_cmap(max_rank: int) -> tuple[ListedColormap, BoundaryNorm]:
    """High-contrast red-to-purple colors for accept ranks 2 and later."""
    if max_rank < 2:
        raise ValueError("max_rank must be at least two")
    start = np.asarray(matplotlib.colors.to_rgb(REVISION_COLOR))
    end = np.asarray(matplotlib.colors.to_rgb(REVISION_FINAL_COLOR))
    colors = [
        tuple(start * (1.0 - amount) + end * amount)
        for amount in np.linspace(0.0, 1.0, max_rank - 1)
    ]
    cmap = ListedColormap(colors, name="dllm_bench_reaccept_rank")
    boundaries = np.arange(1.5, max_rank + 1.5, 1.0)
    return cmap, BoundaryNorm(boundaries, cmap.N)


def _acceptance_rank_cmap(
    max_rank: int,
) -> tuple[ListedColormap, BoundaryNorm]:
    """Blue for first acceptance; yellow-to-red for later accepted events."""
    if max_rank < 1:
        raise ValueError("max_rank must be at least one")
    warm = plt.get_cmap("YlOrRd")
    later_colors = [
        warm(value) for value in np.linspace(0.18, 0.92, max_rank - 1)
    ]
    cmap = ListedColormap(
        [FIRST_ACCEPTANCE_COLOR, *later_colors],
        name="dllm_bench_acceptance_rank",
    )
    boundaries = np.arange(0.5, max_rank + 1.5, 1.0)
    return cmap, BoundaryNorm(boundaries, cmap.N)


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
        cbar.set_label("Cumulative accepted-event count (log scale)")
        ticks = count_ticks(max_count)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accept_revisions(
    trace: list[TraceStep],
    out_path: str | Path,
    title: str = "",
    block_length: int | None = None,
) -> None:
    """Plot every accept-state transition on real forwards."""
    if not trace:
        return

    events = extract_acceptance_events(trace)
    accept_positions = events["accept_positions"]
    accept_steps = events["accept_steps"]
    accept_ranks = events["accept_ranks"]
    renoise_positions = events["renoise_positions"]
    renoise_steps = events["renoise_steps"]
    reaccept_positions = events["reaccept_positions"]
    revision_positions = events["revision_positions"]
    revision_accept_indices = events["revision_accept_indices"]
    accept_count_by_position = events["accept_count_by_position"]
    if not accept_positions:
        return

    fig, (ax, side_ax) = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.2),
        gridspec_kw={"width_ratios": [4.6, 1.45]},
    )
    side_ax.axis("off")
    maximum_accept_rank = max(accept_ranks)
    first_accept_indices = [
        index for index, rank in enumerate(accept_ranks) if rank == 1
    ]
    reaccept_indices = [
        index for index, rank in enumerate(accept_ranks) if rank > 1
    ]
    revision_accept_index_set = set(revision_accept_indices)
    same_token_reaccept_indices = [
        index
        for index in reaccept_indices
        if index not in revision_accept_index_set
    ]
    ax.scatter(
        [accept_positions[index] for index in first_accept_indices],
        [accept_steps[index] for index in first_accept_indices],
        color=FIRST_ACCEPTANCE_COLOR,
        marker="o",
        s=30,
        alpha=0.9,
        linewidths=0,
        label="First accept",
    )
    reaccept_scatter = None
    reaccept_cmap, reaccept_norm = _reaccept_rank_cmap(
        max(2, maximum_accept_rank)
    )
    if same_token_reaccept_indices:
        reaccept_scatter = ax.scatter(
            [accept_positions[index] for index in same_token_reaccept_indices],
            [accept_steps[index] for index in same_token_reaccept_indices],
            c=[accept_ranks[index] for index in same_token_reaccept_indices],
            cmap=reaccept_cmap,
            norm=reaccept_norm,
            marker="o",
            s=42,
            alpha=1.0,
            edgecolors="white",
            linewidths=0.7,
            label="Re-accept, same token",
        )
    revision_scatter = None
    if revision_accept_indices:
        revision_scatter = ax.scatter(
            [accept_positions[index] for index in revision_accept_indices],
            [accept_steps[index] for index in revision_accept_indices],
            c=[accept_ranks[index] for index in revision_accept_indices],
            cmap=reaccept_cmap,
            norm=reaccept_norm,
            marker="X",
            s=REVISION_MARKER_SIZE,
            alpha=1.0,
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
            label="Re-accept, token changed",
        )
    if renoise_positions:
        ax.scatter(
            renoise_positions,
            renoise_steps,
            color="#9ca3af",
            marker="v",
            s=28,
            alpha=0.8,
            linewidths=0,
            label="Re-noise",
        )
    # Rank 2 starts at deep red instead of pale yellow; later re-accepts move
    # toward purple. Marker shape still separates same-token and changed-token
    # events while color preserves the re-accept number.
    color_mappable = (
        revision_scatter
        if revision_scatter is not None
        else reaccept_scatter
    )
    if color_mappable is not None:
        colorbar = fig.colorbar(
            color_mappable,
            ax=ax,
            pad=0.02,
            fraction=0.04,
        )
        colorbar.set_label("Accept number at this position")
        colorbar.set_ticks(range(2, maximum_accept_rank + 1))
    n_positions = max(len(step.position_states) for step in trace)
    ax.set_xlim(-1, n_positions)
    ax.set_ylim(-1, max(step.forward_index for step in trace) + 1)
    ax.set_xlabel("Global token position")
    ax.set_ylabel("Forward step")
    ax.set_title(f"{title + ': ' if title else ''}accept trace")
    ax.grid(True, alpha=0.22)
    _draw_block_boundaries(
        ax,
        n_positions=n_positions,
        block_length=block_length,
    )
    revised_positions = len(set(revision_positions))
    repeatedly_accepted_positions = sum(
        count > 1 for count in accept_count_by_position.values()
    )
    handles, labels = ax.get_legend_handles_labels()
    side_ax.legend(handles, labels, loc="upper left", frameon=False)
    side_ax.text(
        0.0,
        0.72,
        f"accept events: {len(accept_positions)}\n"
        f"re-noise events: {len(renoise_positions)}\n"
        f"re-accept events: {len(reaccept_positions) + len(revision_positions)}\n"
        f"token-changing accepts: {len(revision_positions)}\n"
        f"revised positions: {revised_positions}\n"
        f"accepted more than once: {repeatedly_accepted_positions}\n"
        f"maximum accepts: {maximum_accept_rank}",
        transform=side_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_updates(
    trace: list[TraceStep],
    out_path: str | Path,
    title: str = "",
    block_length: int | None = None,
) -> None:
    """Backward-compatible name for the accept/re-noise/revision trace."""
    plot_accept_revisions(trace, out_path, title=title, block_length=block_length)


def plot_block_acceptance_zoom(
    trace: list[TraceStep],
    out_path: str | Path,
    title: str = "",
    block_length: int | None = None,
) -> None:
    """Zoom every sequential block and report within-block acceptance order.

    Global commit-order tau is dominated by models that must finish B0 before
    entering B1. Resetting both position and step inside each block separates
    that scheduler constraint from genuinely autoregressive behavior within
    the active block.
    """
    if not trace or not block_length or block_length < 2:
        return

    from ...metrics.commit_order import kendall_tau_b

    first_step: dict[int, int] = {}
    for step_index, step in enumerate(trace):
        for position in meaningful_committed_positions(trace, step_index):
            first_step.setdefault(int(position), int(step.forward_index))
    if not first_step:
        return

    n_positions = max(first_step) + 1
    n_blocks = (n_positions + block_length - 1) // block_length
    columns = min(4, n_blocks)
    rows = (n_blocks + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.45 * columns, 2.85 * rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for block_index, ax in enumerate(axes.flat):
        if block_index >= n_blocks:
            ax.set_visible(False)
            continue
        start = block_index * block_length
        end = min(start + block_length, n_positions)
        positions = [position for position in range(start, end) if position in first_step]
        if len(positions) < 2:
            ax.set_visible(False)
            continue
        global_steps = [first_step[position] for position in positions]
        step_origin = min(global_steps)
        local_positions = [position - start for position in positions]
        local_steps = [step - step_origin for step in global_steps]
        tau = kendall_tau_b(
            [float(position) for position in local_positions],
            [float(step) for step in local_steps],
        )
        ax.scatter(
            local_positions,
            local_steps,
            s=24,
            color="#2563eb",
            alpha=0.94,
            linewidths=0,
        )
        ax.set_xlim(-1, block_length)
        ax.set_ylim(-1, max(local_steps) + 1)
        ax.set_title(
            f"B{block_index} | local tau={tau:.3f} | {max(local_steps) + 1} steps",
            fontsize=9,
        )
        ax.set_xlabel("Position in block")
        ax.set_ylabel("Step in block")
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        f"{title + ': ' if title else ''}within-block acceptance order",
        fontsize=15,
        y=1.01,
    )
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
    """DGtest-style token-level entropy convergence heatmap.

    Entropy uses DGtest's viridis scale (low=purple, high=yellow), while the
    accepted/revision plot deliberately keeps its separate blue-green-to-red
    count scale. Missing observations stay light grey instead of being
    misrepresented as zero entropy.
    """
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

    observed_steps = len(trace)
    cmap = plt.get_cmap("viridis").with_extremes(bad="#eeeeee")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.max(finite)),
        extent=(-0.5, n_positions - 0.5, 0.5, observed_steps + 0.5),
    )
    ax.set_xlabel("Global token position")
    ax.set_ylabel("Denoising / forward step")
    ax.set_ylim(0.5, observed_steps + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(
        f"{title + ': ' if title else ''}token-level entropy convergence"
    )
    ax.text(
        n_positions - 1,
        observed_steps,
        f"stop {observed_steps}",
        ha="right",
        va="top",
        fontsize=8,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
            "pad": 1.5,
        },
    )
    _draw_block_boundaries(
        ax,
        n_positions=n_positions,
        block_length=block_length,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Normalized token entropy")
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
