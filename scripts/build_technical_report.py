#!/usr/bin/env python3
"""Build the tracked, data-first technical report from local benchmark output."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = ROOT / "output" / "report" / "current_summary.csv"
TRACE_CSV = ROOT / "output" / "analysis" / "trace_comparison" / "trace_metrics.csv"
STRUCTURE_CSV = (
    ROOT / "output" / "analysis" / "trace_comparison" / "structure_first_metrics.csv"
)
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
DATA = DOCS / "data"
REPORT = DOCS / "TECHNICAL_DATA_REPORT.md"

RUN_ORDER = [
    "qwen3_4b",
    "illada_best",
    "illada_fast",
    "dreamreasoner_best",
    "dreamreasoner_fast",
]
RUN_LABELS = {
    "qwen3_4b": "Qwen3-4B AR",
    "illada_best": "iLLaDA Best",
    "illada_fast": "iLLaDA Fast",
    "dreamreasoner_best": "DreamReasoner Best",
    "dreamreasoner_fast": "DreamReasoner Fast",
}
RUN_COLORS = {
    "qwen3_4b": "#286F9B",
    "illada_best": "#C64E3D",
    "illada_fast": "#E58B2F",
    "dreamreasoner_best": "#3F7652",
    "dreamreasoner_fast": "#77A86C",
}
DATASET_ORDER = ["gsm8k", "mbpp", "structeval_t", "sudoku", "ruler", "hellobench"]
DATASET_LABELS = {
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "structeval_t": "StructEval-T",
    "sudoku": "Sudoku",
    "ruler": "RULER",
    "hellobench": "HelloBench",
}
CHECKPOINTS = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "illada_best": "GSAI-ML/iLLaDA-8B-Instruct",
    "illada_fast": "GSAI-ML/iLLaDA-8B-Instruct",
    "dreamreasoner_best": "Dream-org/DreamReasoner-8B",
    "dreamreasoner_fast": "Dream-org/DreamReasoner-8B",
}


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing report input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric = {
        "n_samples",
        "q",
        "tps",
        "sps",
        "eps",
        "cps",
        "time_per_sample_s",
        "energy_per_sample_j",
        "compute_per_sample_tflops",
        "peak_vram_gib",
        "score_per_energy",
        "score_per_compute",
    }
    for raw in _load_csv(SUMMARY_CSV):
        row: dict[str, Any] = dict(raw)
        for key in numeric:
            row[key] = _float(raw.get(key))
        row["aux"] = json.loads(raw.get("aux") or "{}")
        row["status_counts"] = json.loads(raw.get("status_counts") or "{}")
        summary_path = (
            ROOT
            / "output"
            / "score_output"
            / str(row["run"])
            / str(row["dataset"])
            / "summary.json"
        )
        row["run_metadata"] = {}
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            row["run_metadata"] = payload.get("run_metadata", {})
        rows.append(row)
    order = {name: index for index, name in enumerate(RUN_ORDER)}
    dataset_order = {name: index for index, name in enumerate(DATASET_ORDER)}
    rows.sort(
        key=lambda row: (
            dataset_order.get(str(row["dataset"]), 999),
            order.get(str(row["run"]), 999),
        )
    )
    return rows


def load_trace_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _load_csv(TRACE_CSV):
        row: dict[str, Any] = dict(raw)
        for key in ("trace_samples", "mean_tpf", "peak_tpf", "early", "middle", "late", "tau_32", "tau_64"):
            row[key] = _float(raw.get(key))
        rows.append(row)
    return rows


def load_structure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _load_csv(STRUCTURE_CSV):
        row: dict[str, Any] = dict(raw)
        for key in (
            "selected_samples",
            "trace_samples",
            "eligible_samples",
            "eligible_ratio",
            "structure_first_mean",
            "ci_low",
            "ci_high",
        ):
            row[key] = _float(raw.get(key))
        rows.append(row)
    return rows


def value(row: dict[str, Any], key: str) -> float | None:
    direct = _float(row.get(key))
    if direct is not None:
        return direct
    return _float(row.get("aux", {}).get(key))


def success_count(row: dict[str, Any]) -> int:
    return int(row.get("status_counts", {}).get("success", 0))


def fmt(value_: Any, digits: int = 3) -> str:
    number = _float(value_)
    if number is None:
        return "-"
    if digits == 0:
        return str(int(round(number)))
    if number == 0:
        return "0"
    if abs(number) >= 10000 or abs(number) < 0.001:
        return f"{number:.3g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def pct(value_: Any, digits: int = 0) -> str:
    number = _float(value_)
    return "-" if number is None else f"{100 * number:.{digits}f}%"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def clean(cell: Any) -> str:
        return str(cell).replace("|", "\\|").replace("\n", "<br>")

    rendered = [[clean(cell) for cell in row] for row in rows]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *("| " + " | ".join(row) + " |" for row in rendered),
        ]
    )


def row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["run"]), str(row["dataset"])): row for row in rows}


def matrix(
    rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float | None]
) -> np.ndarray:
    indexed = row_index(rows)
    output = np.full((len(RUN_ORDER), len(DATASET_ORDER)), np.nan)
    for run_i, run in enumerate(RUN_ORDER):
        for dataset_i, dataset in enumerate(DATASET_ORDER):
            row = indexed.get((run, dataset))
            if row is not None:
                metric = getter(row)
                if metric is not None:
                    output[run_i, dataset_i] = metric
    return output


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
        }
    )


def _heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    annotations: np.ndarray,
    title: str,
    *,
    cmap_name: str = "viridis",
    norm: Normalize | None = None,
) -> None:
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad("#E4E4E4")
    image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks(range(len(DATASET_ORDER)), [DATASET_LABELS[d] for d in DATASET_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(RUN_ORDER)), [RUN_LABELS[r] for r in RUN_ORDER])
    ax.grid(False)
    for row_i in range(data.shape[0]):
        for col_i in range(data.shape[1]):
            label = annotations[row_i, col_i]
            if label:
                ax.text(col_i, row_i, label, ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.03, pad=0.02)


def plot_coverage(rows: list[dict[str, Any]]) -> None:
    indexed = row_index(rows)
    data = np.full((len(RUN_ORDER), len(DATASET_ORDER)), np.nan)
    annotations = np.full(data.shape, "", dtype=object)
    for run_i, run in enumerate(RUN_ORDER):
        for dataset_i, dataset in enumerate(DATASET_ORDER):
            row = indexed.get((run, dataset))
            if row is None:
                annotations[run_i, dataset_i] = "not run"
                continue
            selected = int(row["n_samples"] or 0)
            success = success_count(row)
            data[run_i, dataset_i] = success / selected if selected else 0
            oom = int(row["status_counts"].get("oom", 0))
            annotations[run_i, dataset_i] = f"{success}/{selected}" + (f"\nOOM {oom}" if oom else "")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _heatmap(ax, data, annotations, "Generation coverage and OOM", cmap_name="RdYlGn", norm=Normalize(0, 1))
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_matrix.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_quality(rows: list[dict[str, Any]]) -> None:
    data = matrix(rows, lambda row: value(row, "q"))
    annotations = np.full(data.shape, "", dtype=object)
    for row_i in range(data.shape[0]):
        for col_i in range(data.shape[1]):
            annotations[row_i, col_i] = "-" if np.isnan(data[row_i, col_i]) else fmt(data[row_i, col_i], 3)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _heatmap(ax, data, annotations, "Primary task score q (task-specific; not cross-task comparable)", norm=Normalize(0, 1))
    fig.tight_layout()
    fig.savefig(FIGURES / "quality_matrix.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_resources(rows: list[dict[str, Any]]) -> None:
    specs = [
        ("tps", "Throughput (tokens/s)", "magma"),
        ("time_per_sample_s", "Latency (s/sample)", "plasma"),
        ("energy_per_sample_j", "Energy (J/sample)", "cividis"),
        ("peak_vram_gib", "Peak VRAM (GiB)", "YlGnBu"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (key, title, cmap) in zip(axes.flat, specs):
        data = matrix(rows, lambda row, metric=key: value(row, metric))
        finite = data[np.isfinite(data) & (data > 0)]
        annotations = np.full(data.shape, "", dtype=object)
        for row_i in range(data.shape[0]):
            for col_i in range(data.shape[1]):
                annotations[row_i, col_i] = "-" if np.isnan(data[row_i, col_i]) else fmt(data[row_i, col_i], 2)
        norm: Normalize | None = None
        if key in {"tps", "time_per_sample_s", "energy_per_sample_j"} and finite.size:
            norm = LogNorm(vmin=max(float(finite.min()), 1e-6), vmax=float(finite.max()))
        _heatmap(ax, data, annotations, title, cmap_name=cmap, norm=norm)
    fig.tight_layout()
    fig.savefig(FIGURES / "resource_matrices.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_quality_cost(rows: list[dict[str, Any]], key: str, label: str, filename: str) -> None:
    indexed = row_index(rows)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for dataset, ax in zip(DATASET_ORDER, axes.flat):
        plotted = False
        max_q = 0.0
        for run in RUN_ORDER:
            row = indexed.get((run, dataset))
            if row is None:
                continue
            x = value(row, key)
            y = value(row, "q")
            if x is None or y is None or x <= 0:
                continue
            ax.scatter(x, y, s=58, color=RUN_COLORS[run], edgecolor="white", linewidth=0.6, zorder=3)
            y_offset = -12 if run in {"illada_best", "dreamreasoner_fast"} else 4
            ax.annotate(RUN_LABELS[run].replace("DreamReasoner", "DreamR."), (x, y), xytext=(4, y_offset), textcoords="offset points", fontsize=7)
            plotted = True
            max_q = max(max_q, y)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel(label)
        ax.set_ylabel("q")
        if plotted:
            ax.set_xscale("log")
            ax.set_ylim(-0.02, min(1.02, max(0.12, max_q * 1.14 + 0.01)))
        else:
            ax.text(0.5, 0.5, "no comparable measurement", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def matched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = row_index(rows)
    output: list[dict[str, Any]] = []
    for family, best, fast in (
        ("iLLaDA", "illada_best", "illada_fast"),
        ("DreamReasoner", "dreamreasoner_best", "dreamreasoner_fast"),
    ):
        for dataset in DATASET_ORDER:
            best_row = indexed.get((best, dataset))
            fast_row = indexed.get((fast, dataset))
            if best_row is None or fast_row is None:
                continue
            best_time = value(best_row, "time_per_sample_s")
            fast_time = value(fast_row, "time_per_sample_s")
            best_energy = value(best_row, "energy_per_sample_j")
            fast_energy = value(fast_row, "energy_per_sample_j")
            output.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "best_q": value(best_row, "q"),
                    "fast_q": value(fast_row, "q"),
                    "q_delta": (
                        value(fast_row, "q") - value(best_row, "q")
                        if value(fast_row, "q") is not None and value(best_row, "q") is not None
                        else None
                    ),
                    "latency_speedup": best_time / fast_time if best_time and fast_time else None,
                    "energy_reduction": best_energy / fast_energy if best_energy and fast_energy else None,
                    "tps_speedup": (
                        value(fast_row, "tps") / value(best_row, "tps")
                        if value(fast_row, "tps") and value(best_row, "tps")
                        else None
                    ),
                    "vram_delta": (
                        value(fast_row, "peak_vram_gib") - value(best_row, "peak_vram_gib")
                        if value(fast_row, "peak_vram_gib") is not None and value(best_row, "peak_vram_gib") is not None
                        else None
                    ),
                }
            )
    return output


def plot_best_fast(rows: list[dict[str, Any]]) -> None:
    pairs = [row for row in matched_rows(rows) if row["latency_speedup"] is not None]
    labels = [f"{row['family']}\n{DATASET_LABELS[row['dataset']]}" for row in pairs]
    x = np.arange(len(pairs))
    colors = ["#D96C3F" if row["family"] == "iLLaDA" else "#5E9365" for row in pairs]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, key, title, baseline in (
        (axes[0], "latency_speedup", "Best latency / Fast latency (higher means Fast is faster)", 1),
        (axes[1], "energy_reduction", "Best energy / Fast energy (higher means Fast uses less energy)", 1),
        (axes[2], "q_delta", "Fast q - Best q", 0),
    ):
        values = [row[key] for row in pairs]
        bars = ax.bar(x, values, color=colors, width=0.72)
        ax.axhline(baseline, color="#555555", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.bar_label(bars, labels=[fmt(v, 3) for v in values], padding=3, fontsize=8)
    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "best_fast_tradeoffs.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_validity(rows: list[dict[str, Any]]) -> None:
    indexed = row_index(rows)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    x = np.arange(len(RUN_ORDER))
    for dataset, ax in zip(DATASET_ORDER, axes.flat):
        valid = []
        complete = []
        for run in RUN_ORDER:
            row = indexed.get((run, dataset))
            valid.append(value(row, "valid_rate") if row else None)
            complete.append(value(row, "complete_rate") if row else None)
        valid_values = [np.nan if item is None else item for item in valid]
        complete_values = [np.nan if item is None else item for item in complete]
        ax.bar(x - 0.19, valid_values, width=0.38, color="#2F7E9E", label="valid")
        ax.bar(x + 0.19, complete_values, width=0.38, color="#D98032", label="complete")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x, [RUN_LABELS[run].replace("DreamReasoner", "DreamR.") for run in RUN_ORDER], rotation=40, ha="right", fontsize=7)
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "validity_completion.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_trace(trace_rows: list[dict[str, Any]]) -> None:
    by_key = {(str(row["run"]), str(row["dataset"])): row for row in trace_rows}
    datasets = ["gsm8k", "mbpp", "structeval_t", "sudoku"]
    x = np.arange(len(datasets))
    width = 0.15
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for run_i, run in enumerate(RUN_ORDER):
        offset = (run_i - (len(RUN_ORDER) - 1) / 2) * width
        mean_tpf = [value(by_key.get((run, dataset), {}), "mean_tpf") for dataset in datasets]
        tau64 = [value(by_key.get((run, dataset), {}), "tau_64") for dataset in datasets]
        axes[0].bar(x + offset, [np.nan if item is None else item for item in mean_tpf], width=width, color=RUN_COLORS[run], label=RUN_LABELS[run])
        axes[1].bar(x + offset, [np.nan if item is None else item for item in tau64], width=width, color=RUN_COLORS[run])
    axes[0].set_title("Mean accepted tokens per forward")
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].set_title("tau64: fraction of final token state reached by normalized step 0.64")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, [DATASET_LABELS[d] for d in datasets])
    fig.tight_layout()
    fig.savefig(FIGURES / "trace_parallelism.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    averages: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for row in trace_rows:
        if row.get("run") not in RUN_ORDER or value(row, "early") is None:
            continue
        aggregate = averages[str(row["run"])]
        aggregate[0] += value(row, "early") or 0
        aggregate[1] += value(row, "middle") or 0
        aggregate[2] += value(row, "late") or 0
        aggregate[3] += 1
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(RUN_ORDER))
    for index, (key, label, color) in enumerate(
        (("early", "early third", "#3B82A0"), ("middle", "middle third", "#D59632"), ("late", "late third", "#B44B4A"))
    ):
        values = [averages[run][index] / averages[run][3] if averages[run][3] else 0 for run in RUN_ORDER]
        ax.bar(range(len(RUN_ORDER)), values, bottom=bottoms, color=color, label=label)
        bottoms += np.asarray(values)
    ax.set_xticks(range(len(RUN_ORDER)), [RUN_LABELS[run] for run in RUN_ORDER], rotation=25, ha="right")
    ax.set_ylabel("mean finalization share")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES / "trace_finalization_share.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_structure(structure_rows: list[dict[str, Any]]) -> None:
    by_key = {(str(row["run"]), str(row["dataset"])): row for row in structure_rows}
    datasets = ["mbpp", "structeval_t"]
    x = np.arange(len(RUN_ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    width = 0.36
    for dataset_i, dataset in enumerate(datasets):
        offset = -width / 2 if dataset_i == 0 else width / 2
        scores = [value(by_key.get((run, dataset), {}), "structure_first_mean") for run in RUN_ORDER]
        eligible = [value(by_key.get((run, dataset), {}), "eligible_ratio") for run in RUN_ORDER]
        color = "#3977A8" if dataset == "mbpp" else "#D17A34"
        axes[0].bar(x + offset, [np.nan if item is None else item for item in scores], width=width, color=color, label=DATASET_LABELS[dataset])
        axes[1].bar(x + offset, [np.nan if item is None else item for item in eligible], width=width, color=color)
    axes[0].set_title("Structure-first score on eligible traces")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].set_title("Eligible trace ratio")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x, [RUN_LABELS[run] for run in RUN_ORDER], rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "structure_first_diagnostics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def copy_snapshot_inputs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUMMARY_CSV, DATA / "pilot_summary.csv")
    shutil.copy2(TRACE_CSV, DATA / "trace_metrics.csv")
    shutil.copy2(STRUCTURE_CSV, DATA / "structure_first_metrics.csv")
    trace_sources = {
        "trace_qwen_structeval.png": ROOT / "output" / "visualization_output" / "qwen3_4b" / "structeval_t" / "structeval-t-100549_heatmap.png",
        "trace_illada_structeval.png": ROOT / "output" / "visualization_output" / "illada_best" / "structeval_t" / "structeval-t-100549_heatmap.png",
        "trace_dreamreasoner_structeval.png": ROOT / "output" / "visualization_output" / "dreamreasoner_best" / "structeval_t" / "structeval-t-100549_heatmap.png",
    }
    for target, source in trace_sources.items():
        if source.is_file():
            shutil.copy2(source, FIGURES / target)


def hardware_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    result: list[list[str]] = []
    for run in RUN_ORDER:
        run_rows = [row for row in rows if row["run"] == run]
        if not run_rows:
            continue
        commits = sorted({str(row.get("code_commit") or "-")[:8] for row in run_rows})
        torch_versions = sorted(
            {
                str(row.get("run_metadata", {}).get("torch_version"))
                for row in run_rows
                if row.get("run_metadata", {}).get("torch_version")
            }
        )
        result.append(
            [
                RUN_LABELS[run],
                CHECKPOINTS[run],
                str(run_rows[0]["config"]),
                str(run_rows[0].get("gpu") or "-"),
                "BF16" if run != "qwen3_4b" else "not recorded",
                ", ".join(torch_versions) or "-",
                ", ".join(commits),
            ]
        )
    return result


def coverage_table(rows: list[dict[str, Any]]) -> str:
    indexed = row_index(rows)
    output = []
    for run in RUN_ORDER:
        cells = [RUN_LABELS[run]]
        for dataset in DATASET_ORDER:
            row = indexed.get((run, dataset))
            if row is None:
                cells.append("-")
                continue
            selected = int(row["n_samples"] or 0)
            success = success_count(row)
            oom = int(row["status_counts"].get("oom", 0))
            cells.append(f"{success}/{selected}" + (f"; OOM {oom}" if oom else ""))
        output.append(cells)
    return md_table(["Run", *(DATASET_LABELS[d] for d in DATASET_ORDER)], output)


def complete_measurement_table(rows: list[dict[str, Any]]) -> str:
    output = []
    for row in rows:
        energy = value(row, "energy_per_sample_j")
        q = value(row, "q")
        q_per_kj = 1000 * q / energy if q is not None and energy else None
        output.append(
            [
                RUN_LABELS.get(str(row["run"]), str(row["run"])),
                DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"])),
                int(row["n_samples"] or 0),
                success_count(row),
                fmt(q, 4),
                fmt(value(row, "tps"), 2),
                fmt(value(row, "time_per_sample_s"), 3),
                fmt(energy, 1),
                fmt(value(row, "eps"), 1),
                fmt(value(row, "peak_vram_gib"), 2),
                fmt(q_per_kj, 4),
                pct(value(row, "valid_rate")),
                pct(value(row, "complete_rate")),
                str(row.get("timing_source") or "-"),
            ]
        )
    return md_table(
        ["Run", "Dataset", "N", "Success", "q", "TPS", "s/sample", "J/sample", "Mean W", "VRAM GiB", "q/kJ", "Valid", "Complete", "Timing"],
        output,
    )


TASK_METRICS = {
    "gsm8k": ["q", "valid_rate", "complete_rate"],
    "mbpp": ["q", "pass_at_1", "valid_rate", "complete_rate", "executable_rate", "structure_first_eligible_ratio", "structure_first_score"],
    "structeval_t": [
        "q",
        "valid_rate",
        "complete_rate",
        "format_valid_rate",
        "complete_correct_rate",
        "official_render_score",
        "official_key_validation_score",
        "field_completion_rate",
        "content_progress",
        "structure_progress",
        "structure_first_eligible_ratio",
        "structure_first_score",
    ],
    "sudoku": [
        "q",
        "exact_solve_rate",
        "valid_rate",
        "complete_rate",
        "blank_cell_accuracy",
        "cell_accuracy",
        "given_preservation_rate",
        "completion_rate",
        "constraint_satisfaction_rate",
        "conflict_rate",
        "blank_cell_accuracy_easy",
        "blank_cell_accuracy_hard",
    ],
    "ruler": [
        "q",
        "valid_rate",
        "complete_rate",
        "accuracy_context_8192",
        "accuracy_niah_context_8192",
        "accuracy_multi_hop_context_8192",
        "accuracy_aggregation_context_8192",
        "accuracy_front_context_8192",
        "accuracy_middle_context_8192",
        "accuracy_back_context_8192",
        "position_robustness_context_8192",
        "accuracy_context_40960",
        "context_retention",
    ],
    "hellobench": [
        "q",
        "valid_rate",
        "complete_rate",
        "objective_quality_score",
        "output_word_count",
        "length_ratio",
        "length_compliance_rate",
        "seq_rep_4",
        "repeated_segment_fraction",
        "major_issue_free_rate",
        "high_repetition_issue_rate",
        "repeated_segment_loop_issue_rate",
        "objective_quality_2000_words",
        "objective_quality_4000_words",
        "mean_output_words_2000_words",
        "mean_output_words_4000_words",
        "sample_count_2000_words",
        "sample_count_4000_words",
    ],
}


def task_metric_table(rows: list[dict[str, Any]], dataset: str) -> str:
    relevant = [row for row in rows if row["dataset"] == dataset]
    metrics = TASK_METRICS[dataset]
    output = []
    for row in relevant:
        output.append(
            [
                RUN_LABELS.get(str(row["run"]), str(row["run"])),
                *(
                    pct(value(row, metric), 1)
                    if metric.endswith("_rate") or metric.endswith("_ratio")
                    else fmt(value(row, metric), 4)
                    for metric in metrics
                ),
            ]
        )
    return md_table(["Run", *metrics], output)


def ar_relative_table(rows: list[dict[str, Any]]) -> str:
    indexed = row_index(rows)
    output = []
    for dataset in ("gsm8k", "mbpp", "structeval_t", "sudoku"):
        baseline = indexed.get(("qwen3_4b", dataset))
        if baseline is None:
            continue
        for run in RUN_ORDER[1:]:
            row = indexed.get((run, dataset))
            if row is None:
                continue
            q_base = value(baseline, "q")
            q_model = value(row, "q")
            time_base = value(baseline, "time_per_sample_s")
            time_model = value(row, "time_per_sample_s")
            energy_base = value(baseline, "energy_per_sample_j")
            energy_model = value(row, "energy_per_sample_j")
            output.append(
                [
                    DATASET_LABELS[dataset],
                    RUN_LABELS[run],
                    fmt(q_model / q_base if q_model is not None and q_base else None, 3),
                    fmt(time_model / time_base if time_model and time_base else None, 3),
                    fmt(energy_model / energy_base if energy_model and energy_base else None, 3),
                    fmt(value(row, "peak_vram_gib") / value(baseline, "peak_vram_gib") if value(row, "peak_vram_gib") and value(baseline, "peak_vram_gib") else None, 3),
                ]
            )
    return md_table(["Dataset", "dLLM run", "q / AR q", "latency / AR", "energy / AR", "VRAM / AR"], output)


def best_fast_table(rows: list[dict[str, Any]]) -> str:
    output = []
    for row in matched_rows(rows):
        output.append(
            [
                row["family"],
                DATASET_LABELS[row["dataset"]],
                fmt(row["best_q"], 4),
                fmt(row["fast_q"], 4),
                fmt(row["q_delta"], 4),
                fmt(row["latency_speedup"], 3),
                fmt(row["energy_reduction"], 3),
                fmt(row["tps_speedup"], 3),
                fmt(row["vram_delta"], 3),
            ]
        )
    return md_table(["Family", "Dataset", "Best q", "Fast q", "Delta q", "Latency speedup", "Energy reduction", "TPS ratio", "VRAM delta GiB"], output)


def trace_table(trace_rows: list[dict[str, Any]]) -> str:
    order = {name: index for index, name in enumerate(RUN_ORDER)}
    dataset_order = {name: index for index, name in enumerate(DATASET_ORDER)}
    selected = sorted(
        trace_rows,
        key=lambda row: (dataset_order.get(str(row["dataset"]), 999), order.get(str(row["run"]), 999)),
    )
    return md_table(
        ["Run", "Dataset", "Trace N", "Mean TPF", "Peak TPF", "Early", "Middle", "Late", "tau32", "tau64"],
        [
            [
                RUN_LABELS.get(str(row["run"]), str(row["run"])),
                DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"])),
                fmt(value(row, "trace_samples"), 0),
                fmt(value(row, "mean_tpf"), 3),
                fmt(value(row, "peak_tpf"), 3),
                pct(value(row, "early"), 1),
                pct(value(row, "middle"), 1),
                pct(value(row, "late"), 1),
                fmt(value(row, "tau_32"), 3),
                fmt(value(row, "tau_64"), 3),
            ]
            for row in selected
        ],
    )


def structure_table(rows: list[dict[str, Any]]) -> str:
    return md_table(
        ["Run", "Dataset", "Selected", "Trace N", "Eligible N", "Eligible ratio", "Structure-first", "95% CI"],
        [
            [
                RUN_LABELS.get(str(row["run"]), str(row["run"])),
                DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"])),
                fmt(value(row, "selected_samples"), 0),
                fmt(value(row, "trace_samples"), 0),
                fmt(value(row, "eligible_samples"), 0),
                pct(value(row, "eligible_ratio"), 1),
                fmt(value(row, "structure_first_mean"), 3),
                f"[{fmt(value(row, 'ci_low'), 3)}, {fmt(value(row, 'ci_high'), 3)}]",
            ]
            for row in rows
        ],
    )


def auxiliary_long_table(rows: list[dict[str, Any]]) -> str:
    output = []
    for row in rows:
        for metric, metric_value in sorted(row.get("aux", {}).items()):
            output.append(
                [
                    RUN_LABELS.get(str(row["run"]), str(row["run"])),
                    DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"])),
                    metric,
                    fmt(metric_value, 6),
                ]
            )
    return md_table(["Run", "Dataset", "Auxiliary metric", "Value"], output)


def build_report(rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], structure_rows: list[dict[str, Any]]) -> str:
    sections = [
        "# dLLM Benchmark Technical Data Report",
        "",
        f"Data snapshot generated {date.today().isoformat()} from the locally imported RTX 4090 pilot outputs. This document contains measurement tables, task diagnostics, matched comparisons, and trace data only.",
        "",
        "Tracked machine-readable snapshots: [primary/system metrics](data/pilot_summary.csv), [trace metrics](data/trace_metrics.csv), and [structure-first metrics](data/structure_first_metrics.csv).",
        "",
        "## 1. Run Metadata",
        "",
        md_table(["Run", "Checkpoint", "Config", "GPU", "Precision", "Torch", "Code commit(s)"], hardware_rows(rows)),
        "",
        "All current rows use one RTX 4090. Rows were produced by multiple code commits, as shown above; no confidence interval from repeated hardware runs is available. `compute_per_sample_tflops` and CPS are null in every imported row because deferred compute replay has not been run.",
        "",
        "## 2. Coverage, Success, And OOM",
        "",
        coverage_table(rows),
        "",
        "![Generation coverage and OOM matrix](figures/coverage_matrix.png)",
        "",
        "Figure 1. Successful generations over selected samples. DreamReasoner StructEval-T has 55 successes and 45 OOMs. iLLaDA RULER has 30 OOMs for both profiles. Blank cells were not run or not imported.",
        "",
        "## 3. Complete Primary And Resource Measurements",
        "",
        complete_measurement_table(rows),
        "",
        "`Mean W` is EPS (`J/s`). The legacy CSV field named `score_per_energy` is `q / EPS`, so it is not Score/J. The `q/kJ` column above is recomputed as `1000 * q / (J/sample)` and is the energy-per-sample efficiency value used in this report.",
        "",
        "![Primary quality matrix](figures/quality_matrix.png)",
        "",
        "Figure 2. Task-specific primary score. Values are comparable across runs within one dataset only.",
        "",
        "![Resource matrices](figures/resource_matrices.png)",
        "",
        "Figure 3. TPS, wall time, joules per sample, and peak VRAM. TPS, time, and energy use logarithmic color normalization. Missing/OOM-only cells are gray.",
        "",
        "## 4. Quality-Cost Coordinates",
        "",
        "![Quality versus latency](figures/quality_vs_latency.png)",
        "",
        "Figure 4. Primary score against measured seconds per sample; x axes are logarithmic. HelloBench includes Qwen n=20 and iLLaDA Best n=1 and is shown as coverage, not a matched estimate.",
        "",
        "![Quality versus energy](figures/quality_vs_energy.png)",
        "",
        "Figure 5. Primary score against measured joules per sample; x axes are logarithmic. OOM-only cells have no cost coordinate.",
        "",
        "### 4.1 AR-Relative Ratios",
        "",
        ar_relative_table(rows),
        "",
        "Ratios use Qwen3-4B as the denominator on the same task. `latency / AR`, `energy / AR`, and `VRAM / AR` are costs, so lower is better. This is a pilot reference, not an equal-parameter or equal-training-compute claim.",
        "",
        "## 5. Matched Best/Fast Comparisons",
        "",
        best_fast_table(rows),
        "",
        "![Best/Fast trade-offs](figures/best_fast_tradeoffs.png)",
        "",
        "Figure 6. Matched profile deltas on rows with timing and energy. A latency or energy ratio above 1 favors Fast. `Delta q` is Fast minus Best. iLLaDA RULER is excluded from ratios because both rows are OOM-only.",
        "",
        "## 6. Parse Validity And Completion",
        "",
        "![Validity and completion](figures/validity_completion.png)",
        "",
        "Figure 7. Parser validity and completion by task. These are independent of semantic correctness and expose outputs that never reached the requested answer format.",
        "",
        "## 7. Task-Level Diagnostics",
    ]
    for index, dataset in enumerate(DATASET_ORDER, start=1):
        sections.extend(
            [
                "",
                f"### 7.{index} {DATASET_LABELS[dataset]}",
                "",
                task_metric_table(rows, dataset),
            ]
        )
        if dataset == "gsm8k":
            sections.extend(["", "Observed rows: Qwen q=0.59; DreamReasoner Best/Fast q=0.51/0.57; iLLaDA Best/Fast q=0.15/0.21. DreamReasoner Fast is 1.49x faster and uses 1.39x less energy than DreamReasoner Best while increasing q by 0.06."])
        elif dataset == "mbpp":
            sections.extend(["", "Observed rows: pass@1 is 0.06 for Qwen, 0/0 for iLLaDA Best/Fast, and 0.01/0.02 for DreamReasoner Best/Fast. Structure-first scores must be read together with their eligible ratios in Section 9."])
        elif dataset == "structeval_t":
            sections.extend(["", "Observed rows: Qwen final score is 0.5803. iLLaDA Best/Fast are 0.0600/0.0560 with zero complete-correct outputs. DreamReasoner Best/Fast are 0.0193/0.0384 and each has 45 OOMs; their timing/resource means cover the 55 successful samples."])
        elif dataset == "sudoku":
            sections.extend(["", "All exact-solve rates are zero. These rows come from the pre-repair prompt/parser snapshot: Qwen's nonzero blank-cell score (0.0080) and DreamReasoner's 0.0007 must remain audit data, not solving claims. iLLaDA never produced a parseable grid."])
        elif dataset == "ruler":
            sections.extend(["", "Qwen's imported run contains 60 rows across the historical 8,192 and 40,960 context points: accuracy is 0.60 at 8,192 and 0 at 40,960, producing q=0.30. The current formal protocol has since changed to the shared 8,192 point only. Both iLLaDA profiles are 30/30 OOM."])
        elif dataset == "hellobench":
            sections.extend(["", "Coverage is unmatched: Qwen has 20 samples across 2K/4K targets; iLLaDA Best has one 2K sample. The iLLaDA row took 1,340.49 s and 601.41 kJ, generated 2,277 words, and triggered both high-repetition and repeated-segment-loop flags."])
    sections.extend(
        [
            "",
            "## 8. Trace Parallelism",
            "",
            trace_table(trace_rows),
            "",
            "![Trace parallelism](figures/trace_parallelism.png)",
            "",
            "Figure 8. Mean accepted tokens per forward and tau64. iLLaDA Best/Fast are exactly 1/2 TPF because the configured schedule commits one/two tokens per denoising forward; these bars describe the sampler budget, not emergent parallelism.",
            "",
            "![Trace finalization share](figures/trace_finalization_share.png)",
            "",
            "Figure 9. Mean fraction of final token states first reached in the early, middle, and late thirds of generation, averaged across trace-bearing task rows.",
            "",
            "## 9. Structure-First Trace Diagnostics",
            "",
            structure_table(structure_rows),
            "",
            "![Structure-first diagnostics](figures/structure_first_diagnostics.png)",
            "",
            "Figure 10. Structure-first score is conditional on eligible traces. The lower panel is required for interpretation: for StructEval-T the eligible ratio is 61% for Qwen, 26% for either iLLaDA profile, and 4-5% for DreamReasoner.",
            "",
            "## 10. Representative StructEval-T Trace Heatmaps",
            "",
            "| Qwen3-4B AR | iLLaDA Best | DreamReasoner Best |",
            "|---|---|---|",
            "| ![Qwen trace](figures/trace_qwen_structeval.png) | ![iLLaDA trace](figures/trace_illada_structeval.png) | ![DreamReasoner trace](figures/trace_dreamreasoner_structeval.png) |",
            "",
            "Figure 11. Forward step versus token position for shared sample `structeval-t-100549`. These plots expose stabilization/revision patterns only; final task scores and parser validity remain separate measurements.",
            "",
            "## 11. Missing Measurement Cells",
            "",
            md_table(
                ["Missing cell", "State in this snapshot"],
                [
                    ["DiffusionGemma vs Gemma 4 26B-A4B", "No A100 generation/score/resource rows imported"],
                    ["Qwen3-8B", "Adapter/config present; no result rows imported"],
                    ["W1 API", "Adapter/config present; no validated result rows imported"],
                    ["Compute/CPS/TFLOPs", "No deferred compute replay in any imported row"],
                    ["DreamReasoner RULER/HelloBench", "No scored rows imported"],
                    ["iLLaDA Fast HelloBench", "No scored row imported"],
                    ["Repeated-run uncertainty", "No repeated same-hardware trials"],
                ],
            ),
            "",
            "## 12. Complete Auxiliary-Metric Ledger",
            "",
            auxiliary_long_table(rows),
        ]
    )
    return "\n".join(sections) + "\n"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_summary_rows()
    trace_rows = load_trace_rows()
    structure_rows = load_structure_rows()
    _style()
    plot_coverage(rows)
    plot_quality(rows)
    plot_resources(rows)
    plot_quality_cost(rows, "time_per_sample_s", "seconds/sample", "quality_vs_latency.png")
    plot_quality_cost(rows, "energy_per_sample_j", "joules/sample", "quality_vs_energy.png")
    plot_best_fast(rows)
    plot_validity(rows)
    plot_trace(trace_rows)
    plot_structure(structure_rows)
    copy_snapshot_inputs()
    REPORT.write_text(build_report(rows, trace_rows, structure_rows), encoding="utf-8")
    print(f"report: {REPORT}")
    print(f"figures: {len(list(FIGURES.glob('*.png')))}")
    print(f"summary rows: {len(rows)}")


if __name__ == "__main__":
    main()
