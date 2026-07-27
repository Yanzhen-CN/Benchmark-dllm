"""Dataset-level trace aggregation and plots required by design section 4.2."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datasets.base import Sample
from ..interfaces import GenerationResult
from ..metrics.certainty import build_certainty_curve
from ..metrics.commit_order import aggregate_commit_order, commit_order_tau_windows
from ..metrics.stats_utils import BinnedPoint, aggregate_curve_by_bins, summarize
from ..metrics.trace_parallelism import (
    compute_final_stable_steps,
    effective_tokens_per_forward,
    finalization_share,
    mean_peak_effective_tokens_per_forward,
    normalized_forward_progress,
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _plot(points: list[BinnedPoint], path: Path, ylabel: str) -> None:
    if not points:
        return
    x = [point.bin_center for point in points]
    y = [point.stats.median for point in points]
    low = [point.stats.ci_low for point in points]
    high = [point.stats.ci_high for point in points]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, low, high, alpha=0.2)
    ax.set_xlabel("Normalized Forward Progress")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_dataset_trace_summary(
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, list[BinnedPoint]]]:
    usable = [(sample, result) for sample, result in records if result.trace]
    summary: dict[str, Any] = {"dataset": dataset_name, "trace_samples": len(usable)}
    curves: dict[str, list[BinnedPoint]] = {}
    if not usable:
        return summary, curves

    tpf_samples: list[list[tuple[float, float]]] = []
    certainty_samples: list[list[tuple[float, float]]] = []
    means: list[float] = []
    peaks: list[float] = []
    shares: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    taus = []

    for _, result in usable:
        length = result.final_valid_length or len(result.trace[-1].token_ids)
        if length <= 0:
            continue
        sequences = [step.token_ids[:length] for step in result.trace]
        stable = compute_final_stable_steps(sequences)
        counts = effective_tokens_per_forward(stable, len(sequences))
        tpf_samples.append([
            (normalized_forward_progress(i, len(sequences)), float(counts[i]))
            for i in range(len(sequences))
        ])
        mean_tpf, peak_tpf = mean_peak_effective_tokens_per_forward(stable, len(sequences))
        means.append(mean_tpf)
        peaks.append(float(peak_tpf))
        for stage, value in finalization_share(stable, len(sequences)).items():
            shares[stage].append(value)
        taus.append(commit_order_tau_windows(list(range(len(stable))), stable))
        certainty = build_certainty_curve(result.trace, length)
        certainty_samples.append([
            (normalized_forward_progress(i, len(certainty)), point[1])
            for i, point in enumerate(certainty)
        ])

    if not means:
        return summary, curves
    curves["tpf"] = aggregate_curve_by_bins(tpf_samples, seed=seed)
    curves["certainty"] = aggregate_curve_by_bins(certainty_samples, seed=seed)
    summary["mean_tpf"] = _jsonable(summarize(means, seed=seed))
    summary["peak_tpf"] = _jsonable(summarize(peaks, seed=seed))
    summary["finalization_share"] = {
        stage: _jsonable(summarize(values, seed=seed)) for stage, values in shares.items()
    }
    summary["commit_order_tau"] = _jsonable(aggregate_commit_order(taus, seed=seed))
    return summary, curves


def render_dataset_trace_report(
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    out_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, curves = build_dataset_trace_summary(dataset_name, records, seed=seed)
    summary_path = out / "dataset_trace_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    written = {"summary": str(summary_path)}
    for name, ylabel in (("tpf", "Tokens per Forward"), ("certainty", "Certainty")):
        path = out / f"dataset_{name}.png"
        _plot(curves.get(name, []), path, ylabel)
        if path.exists():
            written[name] = str(path)
    return written
