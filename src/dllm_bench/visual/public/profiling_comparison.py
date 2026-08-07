"""Cross-model visual summaries for measured step and stage profiling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Any

from ...datasets.base import Sample
from ...interfaces import GenerationResult
from .profiling_report import build_dataset_profiling_summary
from .trace_metrics import StepProfilingRow


@dataclass(frozen=True)
class ProfilingComparisonSeries:
    """One model/config/dataset profiling series ready for comparison."""

    label: str
    dataset: str
    summary: dict[str, Any]
    rows: list[StepProfilingRow]


def build_profiling_comparison_series(
    *,
    label: str,
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    model_name: str | None = None,
    config_name: str | None = None,
) -> ProfilingComparisonSeries:
    """Build a reusable comparison series from measured generation records."""
    summary, rows = build_dataset_profiling_summary(
        dataset_name,
        records,
        model_name=model_name,
        config_name=config_name,
    )
    return ProfilingComparisonSeries(label, dataset_name, summary, rows)


def write_profiling_comparison_csv(
    series: list[ProfilingComparisonSeries], path: str | Path
) -> bool:
    """Write the exact summary values used by comparison figures."""
    output = Path(path)
    rows = []
    for item in series:
        summary = item.summary
        rows.append(
            {
                "label": item.label,
                "model": summary.get("model"),
                "config": summary.get("config"),
                "dataset": item.dataset,
                "measurement_status": summary.get("measurement_status"),
                "profiled_samples": summary.get("profiled_samples"),
                "step_count": summary.get("step_count"),
                "time_seconds": summary.get("time_seconds"),
                "compute_tflops": summary.get("compute_tflops"),
                "accepted_tokens": summary.get("accepted_tokens"),
                "time_per_accepted_token": summary.get(
                    "time_per_accepted_token"
                ),
                "accepted_token_tps": summary.get("accepted_token_tps"),
                "compute_per_accepted_token": summary.get(
                    "compute_per_accepted_token"
                ),
            }
        )
    if not rows:
        output.unlink(missing_ok=True)
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return True


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else float("nan")


def _ordered_values(series: list[ProfilingComparisonSeries], attribute: str) -> list[str]:
    return list(dict.fromkeys(str(getattr(item, attribute)) for item in series))


def plot_profiling_totals_comparison(
    series: list[ProfilingComparisonSeries], path: str | Path
) -> bool:
    """Compare total measured cost and productive-token cost across datasets."""
    output = Path(path)
    available = [item for item in series if item.rows]
    if not available:
        output.unlink(missing_ok=True)
        return False
    import matplotlib.pyplot as plt
    import numpy as np

    labels = _ordered_values(available, "label")
    datasets = _ordered_values(available, "dataset")
    colors = plt.get_cmap("Dark2").colors
    metrics = [
        ("time_seconds", "Measured model-step time", "Seconds", False),
        ("compute_tflops", "Measured model-step compute", "TFLOP", True),
        (
            "accepted_token_tps",
            "Accepted-token throughput",
            "Accepted-token TPS",
            False,
        ),
        (
            "compute_per_accepted_token",
            "Compute per accepted token",
            "TFLOP / token",
            True,
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(datasets), dtype=float)
    width = 0.78 / max(len(labels), 1)
    lookup = {(item.label, item.dataset): item for item in available}
    for axis, (key, title, ylabel, log_scale) in zip(axes.flat, metrics):
        for index, label in enumerate(labels):
            offset = (index - (len(labels) - 1) / 2) * width
            values = [
                _finite(lookup[(label, dataset)].summary.get(key))
                if (label, dataset) in lookup
                else float("nan")
                for dataset in datasets
            ]
            axis.bar(
                x + offset,
                values,
                width,
                label=label,
                color=colors[index % len(colors)],
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title, weight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, datasets)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Profiling cost comparison (one deterministic replay per cell)",
        fontsize=15,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_step_comparison(
    series: list[ProfilingComparisonSeries], path: str | Path
) -> bool:
    """Compare different-length executions using their recorded forward steps."""
    output = Path(path)
    available = [item for item in series if item.rows]
    if not available:
        output.unlink(missing_ok=True)
        return False
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = plt.get_cmap("Dark2").colors
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    has_kv = False
    for index, item in enumerate(available):
        color = colors[index % len(colors)]
        forward_steps = [row.step_index for row in item.rows]
        cumulative_accepted = 0.0
        cumulative_time = 0.0
        cumulative_accepted_tps = []
        for row in item.rows:
            if row.accepted_tokens is not None and row.accepted_tokens > 0:
                cumulative_accepted += float(row.accepted_tokens)
            if row.time_seconds is not None and row.time_seconds > 0:
                cumulative_time += float(row.time_seconds)
            cumulative_accepted_tps.append(
                cumulative_accepted / cumulative_time
                if cumulative_time > 0
                else float("nan")
            )
        axes[0, 0].plot(
            forward_steps,
            cumulative_accepted_tps,
            color=color,
            label=item.label,
        )
        compute_values = [_finite(row.compute_tflops) for row in item.rows]
        axes[0, 1].plot(
            forward_steps,
            compute_values,
            color=color,
            label=item.label,
        )
        if item.label.split("/", 1)[0] == "dreamreasoner":
            denoise_compute = [
                float(row.compute_tflops)
                for row in item.rows
                if row.phase not in {"prefill", "prefill_or_cache_build", "finalization"}
                and row.compute_tflops is not None
                and math.isfinite(float(row.compute_tflops))
                and row.compute_tflops > 0
            ]
            if denoise_compute:
                typical_compute = statistics.median(denoise_compute)
                axes[0, 1].annotate(
                    f"Dream denoise ≈ {typical_compute:.2f} TFLOP / step",
                    xy=(0.58, typical_compute),
                    xytext=(0.34, 0.14),
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.1},
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": color,
                        "alpha": 0.92,
                    },
                    color=color,
                    fontsize=9,
                )
        axes[1, 0].plot(
            forward_steps,
            [_finite(row.attention_tokens) for row in item.rows],
            color=color,
        )
        axes[1, 0].plot(
            forward_steps,
            [_finite(row.input_tokens) for row in item.rows],
            color=color,
            linestyle="--",
            alpha=0.75,
        )
        kv_values = [_finite(row.kv_cache_tokens) for row in item.rows]
        if any(math.isfinite(value) for value in kv_values):
            has_kv = True
            axes[1, 1].plot(
                forward_steps,
                kv_values,
                color=color,
                label=item.label,
            )
    panels = [
        (
            axes[0, 0],
            "Cumulative accepted-token throughput by step",
            "Cumulative accepted-token TPS",
        ),
        (axes[0, 1], "Compute per step", "TFLOP"),
        (axes[1, 0], "Input and effective attention span", "Tokens"),
        (axes[1, 1], "KV-cache length", "Tokens"),
    ]
    for axis, title, ylabel in panels:
        axis.set_title(title, weight="bold")
        axis.set_xlabel("Recorded forward step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    if not has_kv:
        axes[1, 1].text(
            0.5,
            0.5,
            "KV cache unavailable for these execution paths",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color="#444444", linewidth=2, label="Attention span"),
            Line2D(
                [0],
                [0],
                color="#444444",
                linewidth=2,
                linestyle="--",
                label="Step input",
            ),
        ],
        frameon=False,
        fontsize=9,
        loc="best",
    )
    if has_kv:
        axes[1, 1].legend(frameon=False, fontsize=9)
    dataset = available[0].dataset
    figure.suptitle(
        f"{dataset}: execution cost and context growth by step",
        fontsize=15,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


# Backward-compatible public name retained for existing visualization callers.
plot_normalized_step_comparison = plot_step_comparison


def plot_stage_share_comparison(
    series: list[ProfilingComparisonSeries], path: str | Path
) -> bool:
    """Compare measured generation-stage time and compute shares."""
    output = Path(path)
    available = [
        item for item in series if item.summary.get("stage_contribution")
    ]
    if not available:
        output.unlink(missing_ok=True)
        return False
    import matplotlib.pyplot as plt
    import numpy as np

    preferred = [
        "input_preparation",
        "prefill",
        "denoise_step",
        "token_selection",
        "canvas_update",
        "cache_finalization",
        "output_decode",
    ]
    observed = list(
        dict.fromkeys(
            stage
            for item in available
            for stage in item.summary["stage_contribution"]
        )
    )
    stages = [stage for stage in preferred if stage in observed]
    stages.extend(stage for stage in observed if stage not in stages)
    colors = plt.get_cmap("Set2").colors
    stage_labels = {
        "input_preparation": "input prep",
        "canvas_initialization": "canvas init",
        "prefill": "prefill",
        "denoise_step": "model step",
        "token_selection": "token selection",
        "canvas_update": "canvas update",
        "cache_finalization": "cache finalization",
        "output_decode": "output decode",
    }
    labels = [
        (
            f"{item.label}\n(prefill: N/A)"
            if "prefill" not in item.summary["stage_contribution"]
            else item.label
        )
        for item in available
    ]
    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    for axis, key, title in (
        (axes[0], "time_seconds", "Recorded time share by emitted stage"),
        (axes[1], "compute_tflops", "Recorded compute share by emitted stage"),
    ):
        left = np.zeros(len(available), dtype=float)
        totals = []
        for item in available:
            values = item.summary["stage_contribution"].values()
            totals.append(
                sum(
                    float(stage[key])
                    for stage in values
                    if stage.get(key) is not None
                )
            )
        for stage_index, stage in enumerate(stages):
            shares = []
            for item, total in zip(available, totals):
                value = item.summary["stage_contribution"].get(stage, {}).get(key)
                shares.append(float(value) / total if value is not None and total else 0.0)
            axis.barh(
                y,
                shares,
                left=left,
                color=colors[stage_index % len(colors)],
                label=stage_labels.get(stage, stage.replace("_", " ")),
            )
            left += np.asarray(shares)
        axis.set_title(title, weight="bold")
        axis.set_xlim(0, 1)
        axis.set_xlabel("Share")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, max(1, len(stages))),
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"{available[0].dataset}: generation-stage composition",
        fontsize=15,
        weight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.115,
        "N/A means the adapter has no separate prefill event; its prompt cost is included in model steps.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.82,
        bottom=0.27,
        wspace=0.34,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def render_profiling_comparison_report(
    series: list[ProfilingComparisonSeries], out_dir: str | Path
) -> dict[str, str]:
    """Write table data plus cross-model summary, step, and stage figures."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    csv_path = output / "profiling_comparison.csv"
    if write_profiling_comparison_csv(series, csv_path):
        written["comparison_csv"] = str(csv_path)
    totals_path = output / "profiling_totals_comparison.png"
    if plot_profiling_totals_comparison(series, totals_path):
        written["totals_comparison_plot"] = str(totals_path)
    for dataset in _ordered_values(series, "dataset"):
        selected = [item for item in series if item.dataset == dataset]
        safe_name = dataset.replace("/", "_").replace("\\", "_")
        step_path = output / f"profiling_step_comparison_{safe_name}.png"
        if plot_step_comparison(selected, step_path):
            written[f"step_comparison_{safe_name}"] = str(step_path)
        stage_path = output / f"profiling_stage_comparison_{safe_name}.png"
        if plot_stage_share_comparison(selected, stage_path):
            written[f"stage_comparison_{safe_name}"] = str(stage_path)
    return written
