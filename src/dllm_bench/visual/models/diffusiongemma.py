"""DiffusionGemma-specific cross-variant trace visualizations.

All step axes are real integer forward indices within a 256-token block.
There is no normalized-time interpolation.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ...datasets.base import Sample
from ...interfaces import GenerationResult, TraceStep
from ..base import ModelVisual

DEFAULT_BLOCK_LENGTH = 256
VARIANT_ORDER = ("official", "SC2", "SC05", "SC0", "EB2", "EB05", "Lg2", "Lg05")


def _ordered(names: Any) -> list[str]:
    names = list(names)
    return [name for name in VARIANT_ORDER if name in names] + sorted(
        set(names) - set(VARIANT_ORDER)
    )


def _block_index(step: TraceStep, previous: int, block_length: int) -> int:
    entropy = step.entropy_by_position or {}
    if entropy:
        counts = Counter(int(position) // block_length for position in entropy)
        return counts.most_common(1)[0][0]
    if step.committed_positions:
        return max(int(position) for position in step.committed_positions) // block_length
    return previous


def _blocks(
    trace: list[TraceStep], block_length: int
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    local_step: dict[int, int] = defaultdict(int)
    previous = 0
    for step in trace:
        block = _block_index(step, previous, block_length)
        previous = block
        local_step[block] += 1
        result[block].append(local_step[block])
    return dict(result)


def _metrics(
    records: list[tuple[Sample, GenerationResult]], block_length: int
) -> dict[str, float | int]:
    usable = [generation for _, generation in records if generation.trace]
    forwards = [
        generation.num_forward_passes or len(generation.trace)
        for generation in usable
    ]
    tokens = [generation.final_valid_length for generation in usable]
    per_block = [
        len(steps)
        for generation in usable
        for steps in _blocks(generation.trace, block_length).values()
        if steps
    ]
    return {
        "samples": len(usable),
        "mean_forward": float(np.mean(forwards)) if forwards else float("nan"),
        "median_forward": float(np.median(forwards)) if forwards else float("nan"),
        "mean_output_tokens": float(np.mean(tokens)) if tokens else float("nan"),
        "mean_forward_per_observed_block": (
            float(np.mean(per_block)) if per_block else float("nan")
        ),
        "weighted_tokens_per_forward": (
            float(sum(tokens) / sum(forwards)) if forwards and sum(forwards) else float("nan")
        ),
    }


def _plot_efficiency(
    dataset_name: str, summaries: dict[str, dict[str, Any]], path: Path
) -> None:
    variants = _ordered(summaries)
    fields = (
        ("mean_forward", "Mean total forward / sample"),
        ("mean_forward_per_observed_block", "Mean forward / observed block"),
        ("weighted_tokens_per_forward", "Weighted tokens / forward"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (field, title) in zip(axes, fields):
        values = [float(summaries[variant][field]) for variant in variants]
        bars = ax.bar(variants, values, color="#27896d")
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_title(title, weight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"DiffusionGemma | {dataset_name} | forward efficiency",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _cross_dataset(comparison_root: Path) -> str | None:
    summaries = []
    for path in sorted(comparison_root.glob("*/model_visual_summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if not summaries:
        return None
    datasets = [summary["dataset"] for summary in summaries]
    variants = _ordered(
        {
            variant
            for summary in summaries
            for variant in summary.get("variants", {})
        }
    )
    fields = (
        ("mean_forward", "Mean total forward / sample"),
        ("mean_forward_per_observed_block", "Mean forward / observed block"),
        ("weighted_tokens_per_forward", "Weighted tokens / forward"),
    )
    csv_path = comparison_root / "cross_dataset_forward.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "variant", *(field for field, _ in fields)])
        for summary in summaries:
            for variant, metrics in summary.get("variants", {}).items():
                writer.writerow(
                    [
                        summary["dataset"],
                        variant,
                        *(metrics.get(field) for field, _ in fields),
                    ]
                )
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), constrained_layout=True)
    for ax, (field, title) in zip(axes, fields):
        matrix = np.full((len(variants), len(datasets)), np.nan)
        for column, summary in enumerate(summaries):
            for row, variant in enumerate(variants):
                value = summary.get("variants", {}).get(variant, {}).get(field)
                if value is not None:
                    matrix[row, column] = float(value)
        image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(datasets)), datasets, rotation=35, ha="right")
        ax.set_yticks(range(len(variants)), variants)
        ax.set_title(title, weight="bold")
        for row in range(len(variants)):
            for column in range(len(datasets)):
                if math.isfinite(matrix[row, column]):
                    ax.text(
                        column,
                        row,
                        f"{matrix[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle(
        "DiffusionGemma cross-dataset forward comparison\n"
        "Raw observed means; each metric uses one shared panel scale",
        fontsize=15,
        weight="bold",
    )
    output_path = comparison_root / "cross_dataset_forward.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_model_specific_visualization(
    *,
    dataset_name: str,
    records_by_variant: dict[str, list[tuple[Sample, GenerationResult]]],
    out_dir: Path,
    seed: int = 42,
    block_length: int | None = None,
    figures: set[str] | None = None,
) -> dict[str, str]:
    """Render only DiffusionGemma-specific forward-efficiency diagnostics.

    Common trace selection, metrics, CSV files, and figures are owned by
    ``visual.public.trace_comparison`` and dispatched by ``visual.__init__``.
    """
    del seed
    block_length = block_length or DEFAULT_BLOCK_LENGTH
    figures = figures or {"all"}
    unsupported = figures.difference({"all", "forward"})
    if unsupported:
        requested = ", ".join(sorted(unsupported))
        raise ValueError(
            f"DiffusionGemma does not own public figure(s): {requested}; "
            "run run_visualization.py for shared trace outputs"
        )
    variants = _ordered(
        variant for variant, records in records_by_variant.items() if records
    )
    if not variants:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "entropy_by_real_forward_step.png",
        "entropy_by_real_forward_step.csv",
        "position_step_entropy.png",
        "trace_comparison.png",
        "trace_comparison.csv",
    ):
        (out_dir / stale).unlink(missing_ok=True)

    summaries = {
        variant: _metrics(records_by_variant[variant], block_length)
        for variant in variants
    }
    efficiency_path = out_dir / "forward_efficiency.png"
    _plot_efficiency(dataset_name, summaries, efficiency_path)

    summary = {
        "model": "diffusiongemma",
        "dataset": dataset_name,
        "block_length": block_length,
        "scope": "model_specific_forward_efficiency_only",
        "variants": summaries,
    }
    summary_path = out_dir / "model_visual_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cross_dataset_path = _cross_dataset(out_dir.parent)
    written = {
        "summary": str(summary_path),
        "forward_efficiency": str(efficiency_path),
    }
    if cross_dataset_path:
        written["cross_dataset_forward"] = cross_dataset_path
    return written


def render_model_comparison_visualization(
    *,
    dataset_name: str,
    records_by_variant: dict[str, list[tuple[Sample, GenerationResult]]],
    out_dir: Path,
    seed: int = 42,
    block_length: int | None = None,
    figures: set[str] | None = None,
) -> dict[str, str]:
    """Compose the same public suite as every model plus DG-only figures."""
    from ..public.trace_comparison import render_trace_comparison

    figures = figures or {"all"}
    allowed = {"all", "trace", "state", "convergence", "yield", "forward"}
    unsupported = figures.difference(allowed)
    if unsupported:
        requested = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported DiffusionGemma figure(s): {requested}")

    written: dict[str, str] = {}
    public_figures = figures.intersection({"trace", "state", "convergence", "yield"})
    if "all" in figures:
        public_figures = {"all"}
    if public_figures:
        written.update(
            render_trace_comparison(
                model_name="diffusiongemma",
                dataset_name=dataset_name,
                records_by_variant=records_by_variant,
                out_dir=out_dir,
                figures=public_figures,
            )
        )
    if "all" in figures or "forward" in figures:
        written.update(
            _render_model_specific_visualization(
                dataset_name=dataset_name,
                records_by_variant=records_by_variant,
                out_dir=out_dir,
                seed=seed,
                block_length=block_length,
                figures={"forward"},
            )
        )
    return written


MODEL_VISUAL = ModelVisual(
    model_name="diffusiongemma",
    render_comparison=render_model_comparison_visualization,
)


def main(argv: list[str] | None = None) -> int:
    return MODEL_VISUAL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
