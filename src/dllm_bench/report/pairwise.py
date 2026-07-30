"""Isolated pairwise quality/resource sensitivity analysis.

This module is intentionally not imported by the ordinary report path.  Each
result is directional (A relative to B) and is never combined into a global
leaderboard.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..metrics.quality_resource import resource_equivalent_quality, scenario_score


class PairwiseCompatibilityError(ValueError):
    """Raised when two summaries do not describe the same benchmark protocol."""


@dataclass(frozen=True)
class PairwiseOptions:
    beta: float = 100.0
    gamma: float = 50.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 100.0:
            raise ValueError("beta must be in [0, 100]")
        if not 0.0 <= self.gamma <= 100.0:
            raise ValueError("gamma must be in [0, 100]")


PAIRWISE_COLUMNS = [
    "Direction",
    "Dataset",
    "q",
    "Seconds/Sample",
    "Energy/Sample",
    "r_speed",
    "r_energy",
    "Q_speed",
    "Q_energy",
    "Q_beta_gamma",
]


def summary_label(summary: dict[str, Any]) -> str:
    return f"{summary['model_name']}/{summary['config_name']}"


def _require_equal(
    model_summary: dict[str, Any], base_summary: dict[str, Any], key: str, label: str
) -> Any:
    model_value = model_summary.get("scoring_metadata", {}).get(key)
    base_value = base_summary.get("scoring_metadata", {}).get(key)
    if model_value is None or base_value is None:
        raise PairwiseCompatibilityError(
            f"{label} is missing; rerun run_score.py with the current code before conversion"
        )
    if model_value != base_value:
        raise PairwiseCompatibilityError(
            f"{label} differs between {summary_label(model_summary)} and "
            f"{summary_label(base_summary)}"
        )
    return model_value


def validate_pairwise_compatibility(
    model_summary: dict[str, Any], base_summary: dict[str, Any]
) -> dict[str, Any]:
    """Validate the design document's same-sample/protocol boundary."""
    if model_summary.get("dataset_name") != base_summary.get("dataset_name"):
        raise PairwiseCompatibilityError("pairwise rows must use the same dataset")

    matched = {
        "sample_set_hash": _require_equal(
            model_summary, base_summary, "sample_set_hash", "sample set hash"
        ),
        "dataset_revision": _require_equal(
            model_summary, base_summary, "dataset_revision", "dataset revision"
        ),
        "prompt_protocol_revision": _require_equal(
            model_summary, base_summary, "prompt_protocol_revision", "prompt protocol"
        ),
        "generation_protocol_revision": _require_equal(
            model_summary,
            base_summary,
            "generation_protocol_revision",
            "prompt/output-budget generation protocol",
        ),
        "expected_sample_count": _require_equal(
            model_summary, base_summary, "expected_sample_count", "sample count"
        ),
    }
    model_boundary = model_summary.get("run_metadata", {}).get("measurement_protocol")
    base_boundary = base_summary.get("run_metadata", {}).get("measurement_protocol")
    if not model_boundary or not base_boundary:
        raise PairwiseCompatibilityError("measurement protocol is missing")
    if model_boundary != base_boundary:
        raise PairwiseCompatibilityError("measurement boundaries differ")
    matched["measurement_protocol"] = model_boundary
    return matched


def compute_pairwise_row(
    model_summary: dict[str, Any],
    base_summary: dict[str, Any],
    *,
    beta: float,
    gamma: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute directional A-relative-to-B values using per-sample resources."""
    options = PairwiseOptions(beta=beta, gamma=gamma)
    protocol = validate_pairwise_compatibility(model_summary, base_summary)
    if model_summary.get("timing_source") != "measured" or base_summary.get(
        "timing_source"
    ) != "measured":
        raise PairwiseCompatibilityError("both timing sources must be measured")

    q = float(model_summary["q"])
    model_time = model_summary.get("time_per_sample")
    base_time = base_summary.get("time_per_sample")
    if not model_time or not base_time:
        raise PairwiseCompatibilityError("Seconds/Sample is missing or non-positive")
    r_speed = float(base_time) / float(model_time)
    q_speed = resource_equivalent_quality(q, r_speed, beta=options.beta)

    model_energy = model_summary.get("energy_per_sample")
    base_energy = base_summary.get("energy_per_sample")
    r_energy = q_energy = combined = None
    if model_energy and base_energy:
        r_energy = float(base_energy) / float(model_energy)
        q_energy = resource_equivalent_quality(q, r_energy, beta=options.beta)
        combined = scenario_score(q_speed, q_energy, gamma=options.gamma)

    direction = f"{summary_label(model_summary)} relative to {summary_label(base_summary)}"
    row = {
        "Direction": direction,
        "Dataset": model_summary["dataset_name"],
        "q": q,
        "Seconds/Sample": float(model_time),
        "Energy/Sample": float(model_energy) if model_energy is not None else None,
        "r_speed": r_speed,
        "r_energy": r_energy,
        "Q_speed": q_speed,
        "Q_energy": q_energy,
        "Q_beta_gamma": combined,
    }
    metadata = {
        "direction": direction,
        "beta": options.beta,
        "gamma": options.gamma,
        "gamma_definition": "energy weight percent; speed weight is 100-gamma",
        "sample_set_hash": protocol["sample_set_hash"],
        "sample_count": protocol["expected_sample_count"],
        "speed_track": "Seconds/Sample_B divided by Seconds/Sample_A",
        "energy_track": "Energy/Sample_B divided by Energy/Sample_A",
        "measurement_protocol": protocol["measurement_protocol"],
        "heuristic_warning": (
            "Pairwise sensitivity analysis only; q remains the primary result. "
            "Do not compare this value across different pairings or use it as a leaderboard."
        ),
    }
    return row, metadata


def render_pairwise_table(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    header = [
        f"Direction: {metadata['direction']}",
        f"beta={metadata['beta']}; gamma={metadata['gamma']} "
        f"({metadata['gamma_definition']})",
        f"sample_set_hash={metadata['sample_set_hash']}; N={metadata['sample_count']}",
        f"speed_track={metadata['speed_track']}",
        f"energy_track={metadata['energy_track']}",
        f"WARNING: {metadata['heuristic_warning']}",
        "",
    ]
    columns = PAIRWISE_COLUMNS
    values = [cell(row.get(column)) for column in columns]
    widths = [max(len(column), len(value)) for column, value in zip(columns, values)]
    header.append(" | ".join(c.ljust(w) for c, w in zip(columns, widths)))
    header.append("-+-".join("-" * w for w in widths))
    header.append(" | ".join(v.ljust(w) for v, w in zip(values, widths)))
    return "\n".join(header)


def plot_pairwise(row: dict[str, Any], metadata: dict[str, Any], out_path: str | Path) -> None:
    values = [("q", row["q"]), ("Q speed", row["Q_speed"])]
    if row.get("Q_energy") is not None:
        values.extend(
            [("Q energy", row["Q_energy"]), ("Q beta/gamma", row["Q_beta_gamma"])]
        )
    labels, heights = zip(*values)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, heights)
    ax.set_ylabel("Directional sensitivity value")
    ax.set_title(
        f"{metadata['direction']}\nbeta={metadata['beta']}, gamma={metadata['gamma']}"
    )
    ax.axhline(row["q"], color="black", linewidth=1, linestyle="--", label="measured q")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_pairwise_outputs(
    row: dict[str, Any], metadata: dict[str, Any], out_dir: str | Path
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "pairwise.txt"
    csv_path = out_dir / "pairwise.csv"
    metadata_path = out_dir / "metadata.json"
    plot_path = out_dir / "pairwise.png"
    table_path.write_text(render_pairwise_table(row, metadata) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRWISE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_pairwise(row, metadata, plot_path)
    return [table_path, csv_path, metadata_path, plot_path]
