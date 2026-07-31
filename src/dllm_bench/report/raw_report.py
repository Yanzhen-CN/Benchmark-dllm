"""Measured-only tables and charts for the first-pass report."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ..runner.output_layout import run_id
from .tables import RAW_COLUMNS, raw_results_row, render_raw_results_table


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def _comparison_key(summary: dict[str, Any]) -> tuple[str, str]:
    devices = summary.get("run_metadata", {}).get("cuda_devices") or []
    hardware = ", ".join(str(value) for value in devices) or "unreported-hardware"
    sample_hash = summary.get("scoring_metadata", {}).get("sample_set_hash") or "unreported-samples"
    return hardware, sample_hash


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [*RAW_COLUMNS, "Primary Metric", "Score per Unit Energy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _attach_trace_summary(
    summary: dict[str, Any], row: dict[str, Any], output_root: Path
) -> None:
    path = (
        output_root
        / "visualization_output"
        / run_id(summary["model_name"], summary["config_name"])
        / summary["dataset_name"]
        / "dataset_trace_summary.json"
    )
    if not path.exists():
        return
    trace_summary = json.loads(path.read_text(encoding="utf-8"))
    row["Trace Summary"] = trace_summary
    row["Mean TPF"] = trace_summary.get("mean_tpf", {}).get("mean")
    row["Peak TPF"] = trace_summary.get("peak_tpf", {}).get("mean")
    row["Revised Position Share"] = (
        trace_summary.get("draft_volatility", {})
        .get("revised_position_share", {})
        .get("mean")
    )


def _remove_stale_report_plots(out_dir: Path) -> None:
    for filename in (
        "quality_tps.png",
        "quality_seconds_per_sample.png",
        "quality_energy_per_sample.png",
        "score_per_unit_energy.png",
        "p1_vs_p2_quality.png",
        "p1_vs_p2_tps.png",
        "task4_tpf_profile.png",
        "task4_certainty.png",
        "task4_tpf_vs_tps.png",
        "task4_parallelism_signature.png",
        "task4_draft_volatility.png",
        "task4_update_geometry.png",
        "task4_visible_draft_correction.png",
        "task4_confidence_dynamics.png",
        "task4_forward_yield.png",
        "dflash_speculative_acceptance.png",
        "task4_commit_tau_windows.png",
        "task4_finalization_share.png",
        "task4_style_coverage.png",
        "task4_structure_first.png",
        "task4_sudoku_revision.png",
        "parallelism_ablation.png",
        "parallelism_generation_dynamics.png",
        "parallelism_structure_first.png",
        "parallelism_quality_latency.png",
        "answer_region_diagnostics.png",
    ):
        (out_dir / filename).unlink(missing_ok=True)


def _nested_mean(trace_summary: dict[str, Any], *path: str) -> Any:
    value: Any = trace_summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _write_trace_metrics(rows: list[dict[str, Any]], path: Path) -> bool:
    fields = [
        "Model",
        "Config",
        "N",
        "Mean TPF",
        "Peak TPF",
        "Commit Tau (32)",
        "Mean Finalization Run",
        "Busiest 10% Finalization Share",
        "P90 Final-Stable Progress",
        "Revised Position Share",
    ]
    metrics = []
    for row in rows:
        trace = row.get("Trace Summary") or {}
        if not trace:
            continue
        metrics.append(
            {
                "Model": row.get("Model"),
                "Config": row.get("Config"),
                "N": row.get("N"),
                "Mean TPF": row.get("Mean TPF"),
                "Peak TPF": row.get("Peak TPF"),
                "Commit Tau (32)": _nested_mean(
                    trace, "commit_order_tau", "32"
                ),
                "Mean Finalization Run": _nested_mean(
                    trace, "update_geometry", "mean_finalization_run_length"
                ),
                "Busiest 10% Finalization Share": _nested_mean(
                    trace,
                    "parallelism_signature",
                    "busiest_10pct_finalization_share",
                ),
                "P90 Final-Stable Progress": _nested_mean(
                    trace, "final_stable_progress", "p90"
                ),
                "Revised Position Share": row.get("Revised Position Share"),
            }
        )
    if not metrics:
        return False
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    return True


def write_raw_report(summaries: list[dict[str, Any]], report_root: str | Path) -> list[Path]:
    """Write raw artifacts and only compare rows with matching samples/hardware."""
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    output_root = report_root.parent
    rows = []
    for summary in summaries:
        row = raw_results_row(summary)
        _attach_trace_summary(summary, row, output_root)
        rows.append(row)
    written: list[Path] = []

    table_path = report_root / "raw_results.txt"
    table_path.write_text(render_raw_results_table(rows) + "\n", encoding="utf-8")
    written.append(table_path)
    csv_path = report_root / "raw_results.csv"
    _write_csv(rows, csv_path)
    written.append(csv_path)
    details_path = report_root / "raw_summary_details.json"
    details_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written.append(details_path)

    dataset_names = sorted({summary["dataset_name"] for summary in summaries})
    for dataset_name in dataset_names:
        if dataset_name == "ruler_context_probe":
            continue
        dataset_summaries = [
            summary for summary in summaries if summary["dataset_name"] == dataset_name
        ]
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for summary in dataset_summaries:
            groups.setdefault(_comparison_key(summary), []).append(summary)
        multiple_groups = len(groups) > 1
        for (hardware, sample_hash), group_summaries in groups.items():
            out_dir = report_root / dataset_name
            if multiple_groups:
                out_dir = out_dir / f"{_slug(hardware)}__samples-{sample_hash[:12]}"
            out_dir.mkdir(parents=True, exist_ok=True)
            group_rows = []
            for summary in group_summaries:
                row = raw_results_row(summary)
                _attach_trace_summary(summary, row, output_root)
                group_rows.append(row)
            group_metadata = {
                "dataset": dataset_name,
                "hardware": hardware,
                "sample_set_hash": sample_hash,
                "n_by_row": {
                    f"{summary['model_name']}/{summary['config_name']}": summary.get("n_samples")
                    for summary in group_summaries
                },
                "note": (
                    "Only rows with the same dataset, sample-set hash, and reported hardware "
                    "are placed in this chart directory. Values remain raw measurements."
                ),
            }
            metadata_path = out_dir / "comparison_scope.json"
            metadata_path.write_text(
                json.dumps(group_metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written.append(metadata_path)

            _remove_stale_report_plots(out_dir)
            trace_metrics_path = out_dir / "trace_metrics.csv"
            trace_metrics_path.unlink(missing_ok=True)
            if _write_trace_metrics(group_rows, trace_metrics_path):
                written.append(trace_metrics_path)
            (out_dir / "task4_tpf_vs_tps.csv").unlink(missing_ok=True)
    return written
