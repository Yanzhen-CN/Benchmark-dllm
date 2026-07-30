"""Measured-only tables and charts for the first-pass report."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ..runner.output_layout import run_id
from .plots import (
    plot_answer_region_diagnostics,
    plot_best_vs_fast,
    plot_quality_vs_resource,
    plot_score_per_unit,
    plot_sudoku_revision_diagnostics,
    plot_task4_curve_overlay,
    plot_task4_draft_volatility,
    plot_task4_finalization_share,
    plot_task4_parallelism_signature,
    plot_task4_style_coverage,
    plot_task4_structure_first,
    plot_task4_tau_windows,
    plot_tpf_vs_tps,
)
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

            for key, filename in (
                ("Tps", "quality_tps.png"),
                ("Seconds/Sample", "quality_seconds_per_sample.png"),
                ("Energy/Sample", "quality_energy_per_sample.png"),
            ):
                path = out_dir / filename
                plot_quality_vs_resource(group_rows, key, str(path))
                if path.exists():
                    written.append(path)
            energy_path = out_dir / "score_per_unit_energy.png"
            plot_score_per_unit(group_rows, "Score per Unit Energy", str(energy_path))
            if energy_path.exists():
                written.append(energy_path)
            answer_path = out_dir / "answer_region_diagnostics.png"
            plot_answer_region_diagnostics(group_rows, str(answer_path))
            if answer_path.exists():
                written.append(answer_path)
            for metric, filename in (
                ("q", "best_vs_fast_quality.png"),
                ("Tps", "best_vs_fast_tps.png"),
            ):
                path = out_dir / filename
                plot_best_vs_fast(group_rows, metric, str(path))
                if path.exists():
                    written.append(path)

            task4_specs = (
                (
                    "task4_tpf_profile.png",
                    lambda path: plot_task4_curve_overlay(
                        group_rows,
                        "tpf",
                        str(path),
                        xlabel="Normalized Forward Progress",
                        ylabel="Mean TPF (token/forward)",
                    ),
                ),
                (
                    "task4_certainty.png",
                    lambda path: plot_task4_curve_overlay(
                        group_rows,
                        "certainty",
                        str(path),
                        xlabel="Accepted Ratio",
                        ylabel="Remaining-token Certainty",
                    ),
                ),
                ("task4_tpf_vs_tps.png", lambda path: plot_tpf_vs_tps(group_rows, str(path))),
                (
                    "task4_parallelism_signature.png",
                    lambda path: plot_task4_parallelism_signature(
                        group_rows, str(path)
                    ),
                ),
                (
                    "task4_draft_volatility.png",
                    lambda path: plot_task4_draft_volatility(group_rows, str(path)),
                ),
                (
                    "task4_commit_tau_windows.png",
                    lambda path: plot_task4_tau_windows(group_rows, str(path)),
                ),
                (
                    "task4_finalization_share.png",
                    lambda path: plot_task4_finalization_share(group_rows, str(path)),
                ),
                (
                    "task4_style_coverage.png",
                    lambda path: plot_task4_style_coverage(group_rows, str(path)),
                ),
                (
                    "task4_structure_first.png",
                    lambda path: plot_task4_structure_first(group_rows, str(path)),
                ),
            )
            for filename, plotter in task4_specs:
                path = out_dir / filename
                plotter(path)
                if path.exists():
                    written.append(path)
            if dataset_name in {"sudoku9", "sudoku9_thinking"}:
                path = out_dir / "task4_sudoku_revision.png"
                plot_sudoku_revision_diagnostics(group_rows, str(path))
                if path.exists():
                    written.append(path)

            tpf_rows = [
                {
                    "Model": row["Model"],
                    "Config": row["Config"],
                    "N": row.get("N"),
                    "Mean TPF": row.get("Mean TPF"),
                    "Tps": row.get("Tps"),
                }
                for row in group_rows
                if row.get("Mean TPF") is not None
            ]
            if tpf_rows:
                path = out_dir / "task4_tpf_vs_tps.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(tpf_rows[0]))
                    writer.writeheader()
                    writer.writerows(tpf_rows)
                written.append(path)
    return written
