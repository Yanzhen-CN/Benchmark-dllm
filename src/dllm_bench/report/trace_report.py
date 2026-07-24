"""Section 4.6's per-representative-sample display — the single entry point
every dataset renders through (unified drawing method across all 7
datasets): token-grid GIF/PNG, position-vs-first-commit scatter, commit
speed, Effective Tokens per Forward, Structure/Constraint-vs-Content curves,
Accepted Ratio x Certainty, and the final task result.

:func:`render_sample_report` is what ``dllm-bench visualize`` calls once per
sample. Sudoku gets one more artifact on top of the unified set — an animated
9x9 grid walking through the solve — via a lazy import of
:mod:`dllm_bench.report.sudoku_trace_viz` so that dataset-specific piece
doesn't force every other dataset to depend on it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datasets.base import Sample
from ..interfaces import TraceStep
from ..metrics.certainty import build_certainty_curve
from ..metrics.strategy_score import normalized_progress_series
from ..metrics.trace_parallelism import (
    compute_final_stable_steps,
    effective_tokens_per_forward,
    normalized_forward_progress,
)
from .token_grid_viz import render_token_grid_final_png, render_token_grid_gif
from .trace_distribution_viz import plot_commit_speed, plot_position_vs_first_commit


def plot_effective_tokens_curve(trace: list[TraceStep], out_path: str) -> None:
    if not trace:
        return
    token_id_sequences = [step.token_ids for step in trace]
    final_stable_steps = compute_final_stable_steps(token_id_sequences)
    num_steps = len(trace)
    counts = effective_tokens_per_forward(final_stable_steps, num_steps)

    progresses = [normalized_forward_progress(t, num_steps) for t in range(num_steps)]
    values = [counts[t] for t in range(num_steps)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(progresses, values, marker="o")
    ax.set_xlabel("Normalized Forward Progress")
    ax.set_ylabel("Effective Tokens per Forward")
    ax.set_title("Forward Effective Parallelism")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_strategy_curve(
    form_scores: list[float],
    content_scores: list[float],
    out_path: str,
    form_label: str = "Structure",
    content_label: str = "Content",
) -> None:
    if not form_scores or not content_scores:
        return
    form_norm = normalized_progress_series(form_scores)
    content_norm = normalized_progress_series(content_scores)
    n = len(form_norm)
    progresses = [i / (n - 1) if n > 1 else 0.0 for i in range(n)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(progresses, form_norm, marker="o", label=form_label)
    ax.plot(progresses, content_norm, marker="s", label=content_label)
    ax.set_xlabel("Normalized Forward Progress")
    ax.set_ylabel("Normalized Progress")
    ax.set_title(f"{form_label} vs {content_label} formation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_certainty_curve(trace: list[TraceStep], final_valid_length: int, out_path: str) -> None:
    curve = build_certainty_curve(trace, final_valid_length)
    if not curve:
        return
    ratios = [c[0] for c in curve]
    certainties = [c[1] for c in curve]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ratios, certainties, marker="o")
    ax.set_xlabel("Accepted Ratio")
    ax.set_ylabel("Certainty")
    ax.set_title("Remaining-token Certainty")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _maybe_render_sudoku_gif(
    dataset_name: str, sample: Sample | None, trace: list[TraceStep], out_dir: Path, sample_id: str
) -> str | None:
    if dataset_name != "sudoku" or sample is None or not trace:
        return None
    n_positions = len(trace[-1].position_states)
    if n_positions != 81:
        return None  # trace doesn't align to a row-major 9x9 canvas; skip rather than guess

    from ..datasets.sudoku import SudokuReference
    from .sudoku_trace_viz import derive_sudoku_frames, render_sudoku_gif

    ref: SudokuReference = sample.reference
    frames = derive_sudoku_frames(trace, ref.puzzle, ref.solution)
    path = out_dir / f"{sample_id}_sudoku.gif"
    render_sudoku_gif(frames, ref.puzzle, ref.solution, path)
    return str(path)


def render_sample_report(
    sample_id: str,
    trace: list[TraceStep],
    final_valid_length: int,
    out_dir: str,
    form_scores: list[float] | None = None,
    content_scores: list[float] | None = None,
    final_output_text: str = "",
    final_score: float | None = None,
    dataset_name: str | None = None,
    sample: Sample | None = None,
) -> dict[str, str]:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    title = f"{dataset_name or ''} - {sample_id}".strip(" -")

    if trace:
        gif_path = out_dir_path / f"{sample_id}_trace.gif"
        render_token_grid_gif(trace, gif_path, title=title)
        written["token_grid_gif"] = str(gif_path)

        final_png_path = out_dir_path / f"{sample_id}_trace_final.png"
        render_token_grid_final_png(trace, final_png_path, title=title)
        written["token_grid_final"] = str(final_png_path)

        position_vs_commit_path = out_dir_path / f"{sample_id}_position_vs_commit.png"
        plot_position_vs_first_commit(trace, position_vs_commit_path, title=title)
        written["position_vs_commit"] = str(position_vs_commit_path)

        speed_path = out_dir_path / f"{sample_id}_speed.png"
        plot_commit_speed(trace, speed_path, title=title)
        written["speed"] = str(speed_path)

    parallelism_path = out_dir_path / f"{sample_id}_parallelism.png"
    plot_effective_tokens_curve(trace, str(parallelism_path))
    written["parallelism"] = str(parallelism_path)

    if form_scores and content_scores:
        strategy_path = out_dir_path / f"{sample_id}_strategy.png"
        plot_strategy_curve(form_scores, content_scores, str(strategy_path))
        written["strategy"] = str(strategy_path)

    certainty_path = out_dir_path / f"{sample_id}_certainty.png"
    plot_certainty_curve(trace, final_valid_length, str(certainty_path))
    written["certainty"] = str(certainty_path)

    if dataset_name is not None:
        sudoku_gif = _maybe_render_sudoku_gif(dataset_name, sample, trace, out_dir_path, sample_id)
        if sudoku_gif:
            written["sudoku_gif"] = sudoku_gif

    summary_path = out_dir_path / f"{sample_id}_result.txt"
    summary_path.write_text(
        f"sample_id: {sample_id}\nfinal_score: {final_score}\noutput:\n{final_output_text}\n",
        encoding="utf-8",
    )
    written["result"] = str(summary_path)

    return written
