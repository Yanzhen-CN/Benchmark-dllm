"""Reproducible figures referenced by the benchmark results document."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


CORE_DATASETS = ("gsm8k", "mbpp", "structeval_t")
DATASET_LABELS = {
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "structeval_t": "StructEval-T",
}
MAIN_RUNS = (
    ("illada", "p2"),
    ("dreamreasoner", "p2"),
    ("diffusiongemma", "official"),
    ("gemma", "ar-baseline"),
)
RUN_LABELS = {
    ("illada", "p2"): "iLLaDA P2",
    ("dreamreasoner", "p2"): "DreamReasoner P2",
    ("diffusiongemma", "official"): "DiffusionGemma",
    ("gemma", "ar-baseline"): "Gemma AR",
}
RUN_COLORS = {
    ("illada", "p2"): "#d97706",
    ("dreamreasoner", "p2"): "#2563eb",
    ("diffusiongemma", "official"): "#0f766e",
    ("gemma", "ar-baseline"): "#64748b",
}
HARDWARE_GROUPS = (
    (
        "A100 80GB",
        (("diffusiongemma", "official"), ("gemma", "ar-baseline")),
    ),
    (
        "RTX 4090",
        (("dreamreasoner", "p2"), ("illada", "p2")),
    ),
)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _paper_rows(summaries: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float | None]]:
    rows: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for summary in summaries:
        run = (str(summary.get("model_name")), str(summary.get("config_name")))
        dataset = str(summary.get("dataset_name"))
        if run not in MAIN_RUNS or dataset not in CORE_DATASETS:
            continue
        rows[(run[0], run[1], dataset)] = {
            "quality": _number(summary.get("q")),
            "seconds": _number(summary.get("time_per_sample")),
            "energy": _number(summary.get("energy_per_sample")),
        }
    return rows


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="#f8f6ef")
    plt.close(fig)


def _quality_comparison(
    rows: dict[tuple[str, str, str], dict[str, float | None]], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(CORE_DATASETS), figsize=(11.8, 4.3))
    positions = np.arange(len(MAIN_RUNS))
    colors = [RUN_COLORS[run] for run in MAIN_RUNS]
    for index, (axis, dataset) in enumerate(zip(axes, CORE_DATASETS)):
        values = [
            (rows.get((run[0], run[1], dataset)) or {}).get("quality")
            for run in MAIN_RUNS
        ]
        plotted = [value if value is not None else 0.0 for value in values]
        bars = axis.barh(positions, plotted, color=colors, height=0.62)
        axis.set_xlim(0, 1.08)
        axis.set_title(DATASET_LABELS[dataset], fontsize=11, weight="bold", pad=10)
        axis.set_yticks(
            positions,
            [RUN_LABELS[run] for run in MAIN_RUNS] if index == 0 else [],
            fontsize=9,
        )
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.24, linewidth=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            label = "N/A" if value is None else f"{value:.3f}"
            axis.text(
                min(bar.get_width() + 0.018, 1.035),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=8,
            )
    fig.suptitle(
        "Primary quality by task",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Scores are task-specific and are not pooled across datasets.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.19, right=0.985, top=0.82, bottom=0.16, wspace=0.12)
    _save(fig, path)


def _resource_comparison(
    rows: dict[tuple[str, str, str], dict[str, float | None]], path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.5))
    positions = np.arange(len(CORE_DATASETS))
    metrics = (("seconds", "Time / sample (s)"), ("energy", "Energy / sample (J)"))
    bar_height = 0.34

    for row_index, (hardware, runs) in enumerate(HARDWARE_GROUPS):
        for column_index, (metric, metric_label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            maximum = 0.0
            plotted_groups = []
            for run_index, run in enumerate(runs):
                values = [
                    (rows.get((run[0], run[1], dataset)) or {}).get(metric)
                    for dataset in CORE_DATASETS
                ]
                plotted = [value if value is not None else 0.0 for value in values]
                offset = (run_index - (len(runs) - 1) / 2) * bar_height
                bars = axis.barh(
                    positions + offset,
                    plotted,
                    height=bar_height * 0.88,
                    color=RUN_COLORS[run],
                    label=RUN_LABELS[run],
                )
                plotted_groups.append((bars, values))
                maximum = max(maximum, max(plotted, default=0.0))
            axis.set_xlim(0, maximum * 1.24 if maximum > 0 else 1.0)
            axis.set_yticks(
                positions,
                [DATASET_LABELS[name] for name in CORE_DATASETS]
                if column_index == 0
                else [],
                fontsize=9,
            )
            axis.invert_yaxis()
            axis.set_title(f"{hardware} | {metric_label}", fontsize=11, weight="bold", pad=10)
            axis.grid(axis="x", alpha=0.24, linewidth=0.7)
            axis.set_axisbelow(True)
            for bars, values in plotted_groups:
                for bar, value in zip(bars, values):
                    if value is None:
                        label = "N/A"
                    elif metric == "energy":
                        label = f"{value:,.0f}"
                    else:
                        label = f"{value:.2f}"
                    axis.text(
                        bar.get_width() + maximum * 0.018,
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        va="center",
                        fontsize=8,
                    )

    legend = [
        Patch(facecolor=RUN_COLORS[run], label=RUN_LABELS[run])
        for run in MAIN_RUNS
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Measured latency and energy within hardware groups", fontsize=15, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.025,
        "A100 and RTX 4090 rows are separate comparison groups and must not be ranked against each other.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.83, bottom=0.1, wspace=0.15, hspace=0.42)
    _save(fig, path)


def render_paper_assets(
    summaries: list[dict[str, Any]], report_root: str | Path
) -> list[Path]:
    report_root = Path(report_root)
    out_dir = report_root / "paper_assets"
    rows = _paper_rows(summaries)
    written: list[Path] = []

    quality_path = out_dir / "fig1_main_quality_comparison.png"
    _quality_comparison(rows, quality_path)
    if quality_path.exists():
        written.append(quality_path)

    resource_path = out_dir / "fig2_resource_comparison.png"
    _resource_comparison(rows, resource_path)
    if resource_path.exists():
        written.append(resource_path)

    return written
