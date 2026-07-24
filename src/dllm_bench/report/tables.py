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
    time_priority_score,
)

RAW_COLUMNS = [
    "Dataset",
    "Model",
    "Config",
    "q",
    "Time/sample",
    "Energy/sample",
    "Compute/sample",
    "Peak VRAM",
    "Score/J",
    "Score/TFLOP",
    "Status",
]

CONVERTED_COLUMNS = [
    "Dataset",
    "Model",
    "Config",
    "r_time",
    "r_energy",
    "Q_time",
    "Q_energy",
    "Time-priority",
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
        "Time/sample": summary["time_per_sample"],
        "Energy/sample": summary["energy_per_sample"],
        "Compute/sample": summary["compute_per_sample"],
        "Peak VRAM": summary["peak_vram_gb"],
        "Score/J": summary["score_per_energy"],
        "Score/TFLOP": summary["score_per_compute"],
        "Status": status_label,
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
        "r_time": None,
        "r_energy": None,
        "Q_time": None,
        "Q_energy": None,
        "Time-priority": None,
        "Energy-priority": None,
    }

    model_time = model_summary["time_per_sample"]
    baseline_time = baseline_summary["time_per_sample"]
    if model_time and baseline_time:
        r_time = resource_ratio(baseline_time, model_time)
        row["r_time"] = r_time
        row["Q_time"] = resource_equivalent_quality(q, r_time)

    model_energy = model_summary["energy_per_sample"]
    baseline_energy = baseline_summary["energy_per_sample"]
    if model_energy and baseline_energy:
        r_energy = resource_ratio(baseline_energy, model_energy)
        row["r_energy"] = r_energy
        row["Q_energy"] = resource_equivalent_quality(q, r_energy)

    if row["Q_time"] is not None and row["Q_energy"] is not None:
        row["Time-priority"] = time_priority_score(row["Q_time"], row["Q_energy"])
        row["Energy-priority"] = energy_priority_score(row["Q_time"], row["Q_energy"])

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
