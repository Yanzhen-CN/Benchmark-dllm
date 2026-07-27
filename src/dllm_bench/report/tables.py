"""Section 3.4's two result tables: the raw results table (one row per
Dataset x Model x Config) and the AR-relative converted results table (needs
an AR baseline run to compare against).
"""

from __future__ import annotations

from typing import Any

from ..metrics.quality_resource import (
    energy_priority_score,
    resource_equivalent_quality,
    resource_ratio,
    speed_ratio,
    time_priority_score,
)

RAW_COLUMNS = [
    "Dataset",
    "Model",
    "Config",
    "q",
    "TPS",
    "EPS",
    "CPS",
    "Peak VRAM",
    "Score/J",
    "Score/TFLOP",
    "Status",
    "Timing source",
]

CONVERTED_COLUMNS = [
    "Dataset",
    "Model",
    "Config",
    "r_speed",
    "r_energy",
    "Q_speed",
    "Q_energy",
    "Speed-priority",
    "Energy-priority",
]


def raw_results_row(summary: dict[str, Any]) -> dict[str, Any]:
    """``summary`` is a run_summary_to_dict()-shaped dict (or an equivalent
    :class:`~dllm_bench.runner.orchestrator.RunSummary`-like mapping)."""
    status_counts = summary["status_counts"]
    dominant_status = max(status_counts, key=status_counts.get) if status_counts else "unknown"
    status_label = dominant_status if len(status_counts) == 1 else f"{dominant_status}*"

    return {
        "Dataset": summary["dataset_name"],
        "Model": summary["model_name"],
        "Config": summary["config_name"],
        "q": summary["q"],
        "TPS": summary.get("tps"),
        "EPS": summary.get("eps"),
        "CPS": summary.get("cps"),
        "Peak VRAM": summary["peak_vram_gb"],
        "Score/J": summary["score_per_energy"],
        "Score/TFLOP": summary["score_per_compute"],
        "Status": status_label,
        "Timing source": summary.get("timing_source", "unavailable"),
    }


def compute_converted_row(
    model_summary: dict[str, Any], baseline_summary: dict[str, Any]
) -> dict[str, Any]:
    """One row of the converted-results table, comparing ``model_summary``
    against the AR baseline's ``baseline_summary`` (section 3.3)."""
    q = model_summary["q"]
    row: dict[str, Any] = {
        "Dataset": model_summary["dataset_name"],
        "Model": model_summary["model_name"],
        "Config": model_summary["config_name"],
        "r_speed": None,
        "r_energy": None,
        "Q_speed": None,
        "Q_energy": None,
        "Speed-priority": None,
        "Energy-priority": None,
    }

    model_tps = model_summary.get("tps")
    baseline_tps = baseline_summary.get("tps")
    q_ar = baseline_summary["q"]
    if model_tps and baseline_tps:
        r_speed = speed_ratio(model_tps, baseline_tps)
        row["r_speed"] = r_speed
        row["Q_speed"] = resource_equivalent_quality(q, r_speed, q_ar=q_ar)

    model_eps = model_summary.get("eps")
    baseline_eps = baseline_summary.get("eps")
    if model_eps and baseline_eps:
        r_energy = resource_ratio(baseline_eps, model_eps)
        row["r_energy"] = r_energy
        row["Q_energy"] = resource_equivalent_quality(q, r_energy, q_ar=q_ar)

    if row["Q_speed"] is not None and row["Q_energy"] is not None:
        row["Speed-priority"] = time_priority_score(row["Q_speed"], row["Q_energy"])
        row["Energy-priority"] = energy_priority_score(row["Q_speed"], row["Q_energy"])

    return row


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _render_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    formatted_rows = [[_format_cell(row.get(col)) for col in columns] for row in rows]
    widths = [
        max(len(col), *(len(r[i]) for r in formatted_rows)) for i, col in enumerate(columns)
    ]
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    separator = "-+-".join("-" * w for w in widths)
    body = "\n".join(
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)) for r in formatted_rows
    )
    return f"{header}\n{separator}\n{body}"


def render_raw_results_table(rows: list[dict[str, Any]]) -> str:
    return _render_table(RAW_COLUMNS, rows)


def render_converted_results_table(rows: list[dict[str, Any]]) -> str:
    return _render_table(CONVERTED_COLUMNS, rows)
