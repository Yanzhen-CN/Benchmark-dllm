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

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from ..interfaces import TraceStep
from .token_grid_viz import (
    compute_running_accept_counts,
    count_ticks,
    matplotlib_gradient_cmap,
    meaningful_committed_positions,
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
