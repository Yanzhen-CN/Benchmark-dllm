"""Render platform-selected charts through the benchmark visualization entry."""

from __future__ import annotations

import json
import math
import re
import base64
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


COLORS = (
    "#0f766e", "#d97706", "#2563eb", "#dc2626", "#64748b",
    "#16a34a", "#9333ea", "#0891b2", "#ca8a04", "#be123c",
)

MODEL_LABELS = {
    "diffusiongemma": "DiffusionGemma",
    "dreamreasoner": "DreamReasoner",
    "gemma": "Gemma",
    "illada": "iLLaDA",
    "illada_vargen": "iLLaDA + VarGen",
    "llada2_1": "LLaDA2.1-mini",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}
VARIANT_LABELS = {
    "ar-baseline": "AR baseline",
    "best": "Best",
    "dflash": "dFlash",
    "fast": "Fast",
    "official": "Official",
    "p1": "P1",
    "p2": "P2",
    "p4": "P4",
    "p8": "P8",
    "qmode": "Quality",
    "smode": "Speedy",
}
DATASET_LABELS = {
    "gsm8k": "GSM8K",
    "hellobench": "HelloBench",
    "hellobench_2k": "HelloBench 2K",
    "hellobench_4k": "HelloBench 4K",
    "mbpp": "MBPP",
    "ruler": "RULER",
    "structeval_t": "StructEval-T",
    "sudoku4": "Sudoku 4x4",
    "sudoku4_1shot": "Sudoku 4x4 (1-shot)",
    "sudoku4_thinking": "Sudoku 4x4 (thinking)",
    "sudoku9": "Sudoku 9x9",
    "sudoku9_1shot": "Sudoku 9x9 (1-shot)",
    "sudoku9_thinking": "Sudoku 9x9 (thinking)",
}
METRIC_LABELS = {
    "accepted_tokens_per_forward": "Accepted tokens / forward",
    "accepted_tokens_per_sample": "Accepted tokens / sample",
    "accepted_token_tps": "Accepted-token TPS",
    "accepted_tps": "Accepted TPS",
    "compute_per_accepted_token": "Compute / accepted token",
    "compute_per_second": "Compute / second",
    "compute_tflops": "Compute (TFLOPs)",
    "energy_per_sample": "Energy / sample",
    "eps": "Average power",
    "peak_vram_gb": "Peak VRAM",
    "primary_score": "Primary score",
    "time_per_accepted_token": "Time / accepted token",
    "time_per_sample": "Time / sample",
}


def _configure_font() -> bool:
    for family in (
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Arial Unicode MS",
    ):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = family
        plt.rcParams["axes.unicode_minus"] = False
        return True
    return False


HAS_CJK_FONT = _configure_font()


def _display_label(value: Any) -> str:
    text = str(value or "")
    if text in DATASET_LABELS:
        return DATASET_LABELS[text]
    if text in METRIC_LABELS:
        return METRIC_LABELS[text]
    metric_match = re.search(r"\(([A-Za-z][A-Za-z0-9_]*)\)\s*$", text)
    if metric_match and metric_match.group(1) in METRIC_LABELS:
        return METRIC_LABELS[metric_match.group(1)]
    for model in sorted(MODEL_LABELS, key=len, reverse=True):
        if text == model:
            return MODEL_LABELS[model]
        for separator in ("/", "_"):
            prefix = f"{model}{separator}"
            if text.startswith(prefix):
                variant = text[len(prefix):]
                variant_label = VARIANT_LABELS.get(variant, variant)
                return f"{MODEL_LABELS[model]} ({variant_label})"
    return text


def _title(spec: dict[str, Any], path: Path, fallback: str) -> str:
    value = str(spec.get("title") or fallback)
    if HAS_CJK_FONT or value.isascii():
        return value
    section = path.parent.name
    stem = path.stem
    if section == "score":
        return "Overall score"
    if section == "score_detail":
        return "Metric details"
    if section == "performance_adjusted":
        return "Adjusted score" if stem.startswith("adjusted") else "Original score"
    if section == "performance" and stem.startswith("raw_"):
        return "Performance"
    if section == "profiling":
        return "Profiling"
    return fallback


def _plotly_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict) and value.get("bdata") and value.get("dtype"):
        raw = base64.b64decode(str(value["bdata"]))
        array = np.frombuffer(raw, dtype=np.dtype(str(value["dtype"])))
        shape = value.get("shape")
        if shape:
            dimensions = tuple(int(part.strip()) for part in str(shape).split(",") if part.strip())
            if dimensions:
                array = array.reshape(dimensions)
        return array.tolist()
    return []


def _matplotlib_color(value: Any, fallback: str) -> Any:
    if isinstance(value, list):
        return [_matplotlib_color(item, fallback) for item in value]
    if not isinstance(value, str):
        return fallback
    match = re.fullmatch(
        r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)(?:\s*,\s*([0-9.]+))?\s*\)",
        value,
    )
    if not match:
        return value
    red, green, blue = (float(match.group(index)) / 255.0 for index in (1, 2, 3))
    if match.group(4) is None:
        return red, green, blue
    return red, green, blue, float(match.group(4))


def _trace_label(trace: dict[str, Any], index: int, axis_key: str) -> str:
    raw = trace.get("name") or trace.get("legendgroup") or f"series {index + 1}"
    label = _display_label(raw)
    if axis_key == "x3" and not trace.get("name"):
        dash = (trace.get("line") or {}).get("dash")
        suffix = "effective context" if dash == "dash" else "input length"
        return f"{label} / {suffix}"
    return label


def _finish(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="#f8f6ef")
    plt.close(fig)
    return str(path)


def _radar(spec: dict[str, Any], path: Path) -> str:
    rows = spec.get("rows") or []
    dimensions = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    by_model = {
        model: {
            str(row["dataset"]): float(row["value"])
            for row in rows
            if str(row["model"]) == model and row.get("value") is not None
        }
        for model in models
    }
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.8, 6.6), subplot_kw={"polar": True})
    for index, model in enumerate(models):
        values = [by_model[model].get(name, 0.0) for name in dimensions]
        values += values[:1]
        color = COLORS[index % len(COLORS)]
        ax.plot(angles, values, color=color, linewidth=2.2, marker="o", label=_display_label(model))
        ax.fill(angles, values, color=color, alpha=float(spec.get("fill_opacity", 0.12)))
    ax.set_xticks(angles[:-1], [_display_label(name) for name in dimensions])
    if spec.get("scale_mode") == "fixed":
        ax.set_ylim(0, 1)
    ax.set_title(_title(spec, path, "Overall score"), pad=24, weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.06, 1.02), frameon=False)
    ax.grid(alpha=0.28)
    return _finish(fig, path)


def _bar(spec: dict[str, Any], path: Path) -> str:
    rows = spec.get("rows") or []
    facet_key = str(spec.get("facet_key") or "metric")
    category_key = str(spec.get("category_key") or "model")
    value_key = str(spec.get("value_key") or "value")
    facets = list(dict.fromkeys(str(row.get(facet_key, "")) for row in rows)) or [""]
    columns = min(2, len(facets))
    rows_count = math.ceil(len(facets) / columns)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5.8 * columns, 3.35 * rows_count),
        squeeze=False,
    )
    for panel_index, (panel, facet) in enumerate(zip(axes.flat, facets)):
        values = [row for row in rows if str(row.get(facet_key, "")) == facet]
        labels = [_display_label(row.get(category_key, "")) for row in values]
        numbers = [float(row[value_key]) for row in values]
        order = np.arange(len(labels))
        panel.barh(order, numbers, color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
        if panel_index % columns == 0:
            panel.set_yticks(order, labels, fontsize=8)
        else:
            panel.set_yticks(order, [])
        panel.invert_yaxis()
        panel.set_title(_display_label(facet), fontsize=10, pad=10)
        panel.grid(axis="x", alpha=0.22)
        maximum = max(numbers, default=0.0)
        panel.set_xlim(0, maximum * 1.22 if maximum > 0 else 1.0)
        for y, value in zip(order, numbers):
            panel.text(value + maximum * 0.018, y, f"{value:.4g}", va="center", fontsize=8)
    for panel in list(axes.flat)[len(facets):]:
        panel.set_visible(False)
    fig.suptitle(_title(spec, path, "Comparison"), fontsize=15, weight="bold")
    fig.subplots_adjust(
        left=0.24,
        right=0.97,
        top=0.92,
        bottom=0.07,
        wspace=0.28,
        hspace=0.52,
    )
    return _finish(fig, path)


def _plotly(spec: dict[str, Any], path: Path) -> str:
    figure = spec.get("figure") or {}
    traces = figure.get("data") or []
    layout = figure.get("layout") or {}
    panel_keys = list(dict.fromkeys(
        (str(trace.get("xaxis") or "x"), str(trace.get("yaxis") or "y"))
        for trace in traces
        if trace.get("type") != "scatterpolar"
    ))
    if not panel_keys and any(trace.get("type") == "scatterpolar" for trace in traces):
        radar_rows = []
        for trace in traces:
            theta = list(trace.get("theta") or [])
            values = list(trace.get("r") or [])
            for label, value in zip(theta[:-1] or theta, values[:-1] or values):
                radar_rows.append({"model": trace.get("name", "series"), "dataset": label, "value": value})
        return _radar({"rows": radar_rows, "title": (layout.get("title") or {}).get("text", "Overall score")}, path)

    panel_keys = panel_keys or [("x", "y")]
    columns = min(2, len(panel_keys))
    rows_count = math.ceil(len(panel_keys) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(6.4 * columns, 4.0 * rows_count), squeeze=False)
    subplot_titles = [str(item.get("text", "")) for item in layout.get("annotations", []) if item.get("text")]
    for panel_index, (panel, key) in enumerate(zip(axes.flat, panel_keys)):
        selected = [trace for trace in traces if (str(trace.get("xaxis") or "x"), str(trace.get("yaxis") or "y")) == key]
        bars = [trace for trace in selected if trace.get("type") == "bar"]
        lines = [trace for trace in selected if trace.get("type") in {"scatter", "scattergl"}]
        if bars:
            horizontal = any(trace.get("orientation") == "h" for trace in bars)
            stacked = layout.get("barmode") == "stack"
            categories = [
                _display_label(value)
                for value in _plotly_array(bars[0].get("y" if horizontal else "x"))
            ]
            positions = np.arange(len(categories))
            width = 0.8 / max(1, len(bars))
            cumulative = np.zeros(len(categories))
            for index, trace in enumerate(bars):
                values = np.asarray(
                    _plotly_array(trace.get("x" if horizontal else "y")),
                    dtype=float,
                )
                label = _trace_label(trace, index, key[0])
                color = _matplotlib_color(
                    (trace.get("marker") or {}).get("color"),
                    COLORS[index % len(COLORS)],
                )
                if horizontal:
                    offset = cumulative if stacked else None
                    panel.barh(positions if stacked else positions + (index - (len(bars) - 1) / 2) * width, values, height=0.72 if stacked else width, left=offset, label=label, color=color)
                else:
                    offset = cumulative if stacked else None
                    panel.bar(positions if stacked else positions + (index - (len(bars) - 1) / 2) * width, values, width=0.72 if stacked else width, bottom=offset, label=label, color=color)
                if stacked:
                    cumulative += values
            if horizontal:
                panel.set_yticks(positions, categories)
            else:
                panel.set_xticks(positions, categories, rotation=20, ha="right")
        for index, trace in enumerate(lines):
            line = trace.get("line") or {}
            panel.plot(
                _plotly_array(trace.get("x")), _plotly_array(trace.get("y")),
                marker="o" if "markers" in str(trace.get("mode") or "") else None,
                linewidth=1.8,
                linestyle="--" if line.get("dash") == "dash" else "-",
                color=_matplotlib_color(line.get("color"), COLORS[index % len(COLORS)]),
                label=_trace_label(trace, index, key[0]),
            )
        panel.set_title(_display_label(subplot_titles[panel_index]) if panel_index < len(subplot_titles) else "")
        panel.grid(alpha=0.22)
        if len(selected) > 1:
            if len(panel_keys) == 1:
                panel.legend(
                    frameon=False,
                    fontsize=8,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                )
            else:
                panel.legend(
                    frameon=False,
                    fontsize=7,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.14),
                    ncol=min(4, len(selected)),
                )
    for panel in list(axes.flat)[len(panel_keys):]:
        panel.set_visible(False)
    title = layout.get("title") or {}
    explicit_title = title.get("text") if isinstance(title, dict) else title
    fig.suptitle(_title({**spec, "title": explicit_title or spec.get("title")}, path, "Comparison"), fontsize=15, weight="bold")
    fig.tight_layout(h_pad=5.0, w_pad=3.0)
    return _finish(fig, path)


def render_platform_chart(spec_path: str | Path) -> str:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    path = Path(spec["output_path"])
    kind = str(spec.get("kind") or "plotly")
    if kind == "radar":
        return _radar(spec, path)
    if kind == "bar":
        return _bar(spec, path)
    if kind == "plotly":
        return _plotly(spec, path)
    raise ValueError(f"unsupported platform chart kind: {kind}")
