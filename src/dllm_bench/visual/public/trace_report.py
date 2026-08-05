"""Section 4.1's per-representative-sample display — the single entry point
every dataset renders through (one drawing method across all datasets).
Design doc 4.1 is explicit that this display includes only
things that are *inherently single-sample visualizations*:

- Token Position x Forward accepted-event trace; first acceptance is blue and
  later accepted events progress from yellow through orange to red
- final task result

Effective Tokens per Forward (4.2.1) and Structure/Constraint-vs-Content
formation (4.2.2) are explicitly *dataset-level averages* in the design doc
("4.2 里的各项指标都是数据集级平均，不在这里对单个样本重复展示") — computed
once per sample as a building block, then aggregated across the whole
dataset, never shown redundantly for one sample here. Their dataset-level
aggregate versions live in `visual/public/dataset_trace_report.py`.

The generic GIF, final-frame, certainty curve, and speed plots are not emitted.
The separate Sudoku-only board animation remains available for curated Sudoku
examples.

:func:`render_sample_report` is what ``dllm-bench visualize`` calls once per
sample. Sudoku gets one more artifact on top — an animated 9x9 grid walking
through the solve — via a lazy import of
:mod:`dllm_bench.visual.public.sudoku_trace_viz` so that dataset-specific piece
doesn't force every other dataset to depend on it.
"""

from __future__ import annotations

from pathlib import Path

from ...datasets.base import Sample
from ...interfaces import TraceStep
from .trace_distribution_viz import plot_all_updates, plot_block_acceptance_zoom


def _maybe_render_sudoku_gif(
    dataset_name: str,
    sample: Sample | None,
    trace: list[TraceStep],
    final_valid_length: int,
    final_output_text: str,
    out_dir: Path,
    sample_id: str,
) -> str | None:
    if dataset_name not in {
        "sudoku4",
        "sudoku4_1shot",
        "sudoku4_thinking",
        "sudoku9",
        "sudoku9_1shot",
        "sudoku_trace",
    } or sample is None or not trace:
        return None

    from .sudoku_trace_viz import (
        derive_sudoku_frames,
        derive_sudoku_layout_frames,
        render_sudoku_board_gif,
        render_sudoku_layout_gif,
    )

    ref = sample.reference
    frames, forward_steps = derive_sudoku_layout_frames(
        trace,
        ref.puzzle,
        ref.solution,
        final_valid_length,
    )
    if not frames:
        board_frames = derive_sudoku_frames(trace, ref.puzzle, ref.solution)
        if not board_frames:
            return None
        path = out_dir / f"{sample_id}_sudoku_context_trace.gif"
        board_steps = (
            [int(step.forward_index) for step in trace]
            if len(board_frames) == len(trace)
            else None
        )
        render_sudoku_board_gif(
            board_frames,
            ref.puzzle,
            path,
            title=f"{dataset_name} | {sample_id}",
            forward_steps=board_steps,
        )
        return str(path)
    import re

    size = 4 if dataset_name.startswith("sudoku4") else 9
    visible_text = final_output_text or trace[-1].decoded_text
    matches = list(
        re.finditer(
            rf"(?<![0-9])[1-{size}]{{{size * size}}}(?![0-9])",
            visible_text,
        )
    )
    if matches:
        match = matches[-1]
        prefix_context = visible_text[max(0, match.start() - 48) : match.start()]
        suffix_context = visible_text[match.end() : match.end() + 48]
    else:
        prefix_context = ""
        suffix_context = ""
    path = out_dir / f"{sample_id}_sudoku_context_trace.gif"
    render_sudoku_layout_gif(
        frames,
        ref.puzzle,
        path,
        title=f"{dataset_name} | {sample_id}",
        prefix_context=prefix_context,
        suffix_context=suffix_context,
        forward_steps=forward_steps,
    )
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

        block_zoom_path = out_dir_path / f"{sample_id}_block_acceptance.png"
        plot_block_acceptance_zoom(
            trace,
            block_zoom_path,
            title=title,
            block_length=block_length,
        )
        if block_zoom_path.exists():
            written["block_acceptance"] = str(block_zoom_path)

    for suffix in (
        "entropy.png",
        "heatmap.png",
        "certainty.png",
        "update_layers.png",
        "trace.gif",
        "sudoku.gif",
        "sudoku_trace.gif",
        "sudoku_board_trace.gif",
        "sudoku_layout_trace.gif",
        "sudoku_real_trace.gif",
    ):
        (out_dir_path / f"{sample_id}_{suffix}").unlink(missing_ok=True)

    if dataset_name is not None:
        sudoku_gif = _maybe_render_sudoku_gif(
            dataset_name,
            sample,
            trace,
            final_valid_length,
            final_output_text,
            out_dir_path,
            sample_id,
        )
        if sudoku_gif:
            written["sudoku_gif"] = sudoku_gif

    summary_path = out_dir_path / f"{sample_id}_result.txt"
    summary_path.write_text(
        f"sample_id: {sample_id}\nfinal_score: {final_score}\noutput:\n{final_output_text}\n",
        encoding="utf-8",
    )
    written["result"] = str(summary_path)

    return written
