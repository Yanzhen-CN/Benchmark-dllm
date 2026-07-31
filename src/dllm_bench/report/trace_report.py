"""Section 4.1's per-representative-sample display — the single entry point
every dataset renders through (one drawing method across all datasets).
Design doc 4.1 is explicit that this display includes only
things that are *inherently single-sample visualizations*:

- DGtest-style Token Position x Forward all-update trace; later accepted /
  revision events progress from blue-green to orange-red
- token entropy heatmap with high entropy red/orange and low entropy
  blue/green, when available
- final task result

Effective Tokens per Forward (4.2.1) and Structure/Constraint-vs-Content
formation (4.2.2) are explicitly *dataset-level averages* in the design doc
("4.2 里的各项指标都是数据集级平均，不在这里对单个样本重复展示") — computed
once per sample as a building block, then aggregated across the whole
dataset, never shown redundantly for one sample here. Their dataset-level
aggregate versions live in `report/dataset_trace_report.py`.

The generic GIF, final-frame, certainty curve, and speed plots are not emitted.
The separate Sudoku-only board animation remains available for curated Sudoku
examples.

:func:`render_sample_report` is what ``dllm-bench visualize`` calls once per
sample. Sudoku gets one more artifact on top — an animated 9x9 grid walking
through the solve — via a lazy import of
:mod:`dllm_bench.report.sudoku_trace_viz` so that dataset-specific piece
doesn't force every other dataset to depend on it.
"""

from __future__ import annotations

from pathlib import Path

from ..datasets.base import Sample
from ..interfaces import TraceStep
from .trace_distribution_viz import plot_all_updates, plot_token_entropy_heatmap


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
    block_length: int | None = None,
) -> dict[str, str]:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    title = f"{dataset_name or ''} - {sample_id}".strip(" -")

    if trace:
        updates_path = out_dir_path / f"{sample_id}_all_updates.png"
        plot_all_updates(
            trace,
            updates_path,
            title=title,
            block_length=block_length,
        )
        if updates_path.exists():
            written["all_updates"] = str(updates_path)

        entropy_path = out_dir_path / f"{sample_id}_entropy.png"
        plot_token_entropy_heatmap(
            trace,
            entropy_path,
            title=title,
            block_length=block_length,
        )
        if entropy_path.exists():
            written["entropy"] = str(entropy_path)

    for suffix in ("heatmap.png", "certainty.png", "update_layers.png", "trace.gif"):
        (out_dir_path / f"{sample_id}_{suffix}").unlink(missing_ok=True)

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
