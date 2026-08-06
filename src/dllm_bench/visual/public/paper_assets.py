"""Reproducible figures referenced by the benchmark results document."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ...runner.output_layout import run_id
from .style import variant_color

CORE_DATASETS = ("gsm8k", "mbpp", "structeval_t", "sudoku4", "sudoku9")


def _trace_summary(summary: dict[str, Any], output_root: Path) -> dict[str, Any]:
    path = (
        output_root
        / "visualization_output"
        / summary["model_name"]
        / summary["config_name"]
        / summary["dataset_name"]
        / "dataset_trace_summary.json"
    )
    if not path.exists():
        path = (
            output_root
            / "visualization_output"
            / run_id(summary["model_name"], summary["config_name"])
            / summary["dataset_name"]
            / "dataset_trace_summary.json"
        )
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _paper_rows(
    summaries: list[dict[str, Any]], output_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        trace = _trace_summary(summary, output_root)
        mean_tpf = summary.get("tpf", summary.get("accepted_tokens_per_forward"))
        if not isinstance(mean_tpf, (int, float)):
            mean_tpf = (trace.get("mean_tpf") or {}).get("mean")
        q = summary.get("q")
        seconds = summary.get("time_per_sample")
        if not all(isinstance(value, (int, float)) for value in (q, seconds, mean_tpf)):
            continue
        rows.append(
            {
                "dataset": summary["dataset_name"],
                "model": summary["model_name"],
                "config": summary["config_name"],
                "label": f"{summary['model_name']}/{summary['config_name']}",
                "q": float(q),
                "seconds": float(seconds),
                "mean_tpf": float(mean_tpf),
                "n": summary.get("n_samples"),
                "sample_set": (summary.get("scoring_metadata") or {}).get("sample_set_hash"),
            }
        )
    return rows


def _small_multiple_scatter(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    title: str,
    subtitle: str,
    path: Path,
) -> None:
    datasets = [name for name in CORE_DATASETS if any(row["dataset"] == name for row in rows)]
    if not datasets:
        return
    comparison_labels = list(dict.fromkeys(row["label"] for row in rows))
    label_colors = {
        label: variant_color(index) for index, label in enumerate(comparison_labels)
    }
    columns = min(3, len(datasets))
    rows_count = math.ceil(len(datasets) / columns)
    fig = plt.figure(
        figsize=(6.2 * columns + 2.8, 4.7 * rows_count),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        rows_count,
        columns + 1,
        width_ratios=[1.0] * columns + [0.34],
    )
    axes = np.empty((rows_count, columns), dtype=object)
    for row_index in range(rows_count):
        for column_index in range(columns):
            axes[row_index, column_index] = fig.add_subplot(
                grid[row_index, column_index]
            )
    legend_ax = fig.add_subplot(grid[:, columns])
    legend_ax.axis("off")
    for index, dataset in enumerate(datasets):
        ax = axes[index // columns, index % columns]
        values = [row for row in rows if row["dataset"] == dataset]
        for row in values:
            ax.scatter(
                row[x_key],
                row["q"],
                s=52,
                color=label_colors[row["label"]],
                edgecolor="#1f2937",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_title(dataset, weight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Official primary quality")
        ax.grid(color="#d1d5db", alpha=0.55, linewidth=0.7)
        if x_key == "seconds" and values:
            positive = [row[x_key] for row in values if row[x_key] > 0]
            if positive and max(positive) / min(positive) >= 20:
                ax.set_xscale("log")
    for index in range(len(datasets), rows_count * columns):
        axes[index // columns, index % columns].axis("off")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=label_colors[label],
            markeredgecolor="#1f2937",
            markersize=7,
        )
        for label in comparison_labels
    ]
    legend = legend_ax.legend(
        legend_handles,
        comparison_labels,
        loc="center left",
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
    )
    heading = fig.suptitle(
        f"{title}\n{subtitle}",
        fontsize=15,
        weight="bold",
    )
    extra_artists = tuple(
        artist for artist in (legend, heading) if artist is not None
    )
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        bbox_extra_artists=extra_artists,
    )
    plt.close(fig)


def render_paper_assets(
    summaries: list[dict[str, Any]], report_root: str | Path
) -> list[Path]:
    report_root = Path(report_root)
    output_root = report_root.parent
    out_dir = report_root / "paper_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _paper_rows(summaries, output_root)
    written = []

    latency_path = out_dir / "fig1_parallelism_quality_latency.png"
    _small_multiple_scatter(
        rows,
        x_key="seconds",
        x_label="Measured seconds / sample (lower is better)",
        title="Quality and measured latency",
        subtitle="Faceted by dataset; labels are model/config and quality is not pooled across tasks",
        path=latency_path,
    )
    if latency_path.exists():
        written.append(latency_path)

    tpf_path = out_dir / "fig2_parallelism_quality_tpf.png"
    _small_multiple_scatter(
        rows,
        x_key="mean_tpf",
        x_label="Accepted-token events / model forward (TPF; higher is better)",
        title="Quality and measured algorithmic parallelism",
        subtitle="Faceted by dataset; repeated acceptance after re-noise is counted again",
        path=tpf_path,
    )
    if tpf_path.exists():
        written.append(tpf_path)

    return written
