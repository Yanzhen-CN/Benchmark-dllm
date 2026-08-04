"""Design-document section 3.4 measured-only result table helpers."""

from __future__ import annotations

from typing import Any


RAW_COLUMNS = [
    "Dataset",
    "Model",
    "Config",
    "N",
    "Sample Set",
    "q",
    "Primary Aux",
    "Answer Start Ratio",
    "Answer Detect Rate",
    "Tps",
    "Seconds/Sample",
    "Energy/Sample",
    "Average Power",
    "Peak VRAM",
    "Hardware",
    "Status",
    "Timing source",
]

PRIMARY_AUX_KEYS = {
    "gsm8k": ("valid_rate",),
    "mbpp": ("executable_rate",),
    "structeval_t": ("official_render_score", "official_key_validation_score"),
    "sudoku4": ("blank_cell_accuracy", "given_preservation_rate", "legal_completion"),
    "sudoku4_thinking": ("blank_cell_accuracy", "given_preservation_rate", "legal_completion"),
    "sudoku9": ("blank_cell_accuracy", "given_preservation_rate", "legal_completion"),
    "sudoku9_thinking": ("blank_cell_accuracy", "given_preservation_rate", "legal_completion"),
    "ruler": ("all_answers_match",),
    "hellobench": (
        "minimum_length_success_rate",
        "long_output_success_rate",
        "degeneration_free_rate",
        "objective_style_score",
    ),
}


def _primary_aux(summary: dict[str, Any]) -> str | None:
    aux = summary.get("aux", {})
    values = [
        f"{key}={aux[key]:.4g}"
        for key in PRIMARY_AUX_KEYS.get(summary.get("dataset_name"), ())
        if isinstance(aux.get(key), (int, float))
    ]
    return ", ".join(values) or None


def _answer_start_ratio(aux: dict[str, Any]) -> float | None:
    for key in ("answer_start_ratio_mean", "answer_start_ratio", "answer_start_char_ratio"):
        value = aux.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def raw_results_row(summary: dict[str, Any]) -> dict[str, Any]:
    status_counts = summary["status_counts"]
    dominant_status = max(status_counts, key=status_counts.get) if status_counts else "unknown"
    status_label = dominant_status if len(status_counts) == 1 else f"{dominant_status}*"
    aux = summary.get("aux", {})
    scoring_metadata = summary.get("scoring_metadata", {})
    cuda_devices = summary.get("run_metadata", {}).get("cuda_devices") or []
    return {
        "Dataset": summary["dataset_name"],
        "Model": summary["model_name"],
        "Config": summary["config_name"],
        "N": summary.get("n_samples"),
        "Sample Set": scoring_metadata.get("sample_set_hash"),
        "q": summary["q"],
        "Primary Aux": _primary_aux(summary),
        "Answer Start Ratio": _answer_start_ratio(aux),
        "Answer Detect Rate": aux.get("answer_region_detected_rate"),
        "Tps": summary.get("tps"),
        "Seconds/Sample": summary.get("time_per_sample"),
        "Energy/Sample": summary.get("energy_per_sample"),
        "Average Power": summary.get("eps"),
        "Peak VRAM": summary["peak_vram_gb"],
        "Hardware": ", ".join(str(value) for value in cuda_devices) or None,
        "Status": status_label,
        "Timing source": summary.get("timing_source", "unavailable"),
        "Score per Unit Energy": summary.get("score_per_energy"),
        "Primary Metric": scoring_metadata.get("primary_metric"),
        "Aux": aux,
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _render_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    formatted_rows = [[_format_cell(row.get(column)) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in formatted_rows))
        for index, column in enumerate(columns)
    ]
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    separator = "-+-".join("-" * width for width in widths)
    body = "\n".join(
        " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in formatted_rows
    )
    return f"{header}\n{separator}\n{body}"


def render_raw_results_table(rows: list[dict[str, Any]]) -> str:
    return _render_table(RAW_COLUMNS, rows)
