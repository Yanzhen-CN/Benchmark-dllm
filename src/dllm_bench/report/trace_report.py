"""Section 4.1's per-representative-sample display — the single entry point
every dataset renders through (one drawing method across all datasets).
Design doc 4.1 is explicit that this display includes only
things that are *inherently single-sample visualizations*:

- Token Position x Forward heatmap (static; `token_grid_viz.plot_token_position_forward_heatmap`)
- Accepted Ratio x Certainty curve (`plot_certainty_curve`, this module)
- final task result

Effective Tokens per Forward (4.2.1) and Structure/Constraint-vs-Content
formation (4.2.2) are explicitly *dataset-level averages* in the design doc
("4.2 里的各项指标都是数据集级平均，不在这里对单个样本重复展示") — computed
once per sample as a building block, then aggregated across the whole
dataset, never shown redundantly for one sample here. Their dataset-level
aggregate versions live in `report/dataset_trace_report.py`.

The optional animated token-canvas GIF is the moving form of the same generic
position/forward data and is useful for a few curated diffusion examples.  It
must not be confused with the separate Sudoku-only 9x9 board animation.
Redundant final-frame, first-commit scatter, and per-sample speed plots are not
emitted; their information is already clearer in the heatmap and dataset-level
Task 4 summaries.

:func:`render_sample_report` is what ``dllm-bench visualize`` calls once per
sample. Sudoku gets one more artifact on top — an animated 9x9 grid walking
through the solve — via a lazy import of
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
from ..metrics.certainty import build_observed_certainty_curve
from .token_grid_viz import (
    plot_token_position_forward_heatmap,
    render_token_grid_gif,
)


def plot_certainty_curve(trace: list[TraceStep], final_valid_length: int, out_path: str) -> None:
    curve = build_observed_certainty_curve(trace, final_valid_length)
    if len(curve) < 2:
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
    if dataset_name not in {"sudoku9", "sudoku_trace"} or sample is None or not trace:
        return None

    from ..datasets.sudoku9 import SudokuReference
    from .sudoku_trace_viz import derive_sudoku_frames, render_sudoku_gif

    ref: SudokuReference = sample.reference
    frames = derive_sudoku_frames(trace, ref.puzzle, ref.solution)
    if not frames:
        return None
    path = out_dir / f"{sample_id}_sudoku.gif"
    render_sudoku_gif(frames, ref.puzzle, ref.solution, path)
    return str(path)


def render_sample_report(
    sample_id: str,
    trace: list[TraceStep],
    final_valid_length: int,
    out_dir: str,
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
        heatmap_path = out_dir_path / f"{sample_id}_heatmap.png"
        plot_token_position_forward_heatmap(trace, heatmap_path, title=title)
        written["heatmap"] = str(heatmap_path)

        gif_path = out_dir_path / f"{sample_id}_trace.gif"
        render_token_grid_gif(trace, gif_path, title=title)
        written["token_grid_gif"] = str(gif_path)

    certainty_path = out_dir_path / f"{sample_id}_certainty.png"
    plot_certainty_curve(trace, final_valid_length, str(certainty_path))
    if certainty_path.exists():
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
