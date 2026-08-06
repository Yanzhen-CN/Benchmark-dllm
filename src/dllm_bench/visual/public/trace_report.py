"""Section 4.1's per-representative-sample display — the single entry point
every dataset renders through (one drawing method across all datasets).
Design doc 4.1 is explicit that this display includes only
things that are *inherently single-sample visualizations*:

- Token Position x Forward accept/revision trace; first acceptance is blue and
  only token-changing re-acceptance is red
- final task result

Effective Tokens per Forward (4.2.1) and Structure/Constraint-vs-Content
formation (4.2.2) are explicitly *dataset-level averages* in the design doc
("4.2 里的各项指标都是数据集级平均，不在这里对单个样本重复展示") — computed
once per sample as a building block, then aggregated across the whole
dataset, never shown redundantly for one sample here. Their dataset-level
aggregate versions live in `visual/public/dataset_trace_report.py`.

The generic GIF is not emitted for ordinary tasks. Curated Sudoku examples do
emit a model-agnostic token-canvas GIF so traces from models that never form a
parseable board can still be compared. A Sudoku board/layout GIF is added when
the decoded trace can be mapped to cells.

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
from .trace_distribution_viz import plot_accept_revisions


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
        show_context=False,
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
        accept_trace_path = out_dir_path / f"{sample_id}_accept_trace.png"
        plot_accept_revisions(
            trace,
            accept_trace_path,
            title=title,
            block_length=block_length,
        )
        if accept_trace_path.exists():
            written["accept_trace"] = str(accept_trace_path)

    for suffix in (
        "all_updates.png",
        "block_acceptance.png",
        "position_state.png",
        "position_states.png",
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
        "token_trace.gif",
    ):
        (out_dir_path / f"{sample_id}_{suffix}").unlink(missing_ok=True)

    if trace:
        from .token_grid_viz import render_token_grid_gif

        token_gif_path = out_dir_path / f"{sample_id}_token_trace.gif"
        render_token_grid_gif(trace, token_gif_path, title=title)
        if token_gif_path.exists():
            written["token_trace_gif"] = str(token_gif_path)

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
