"""Equal-sample aggregation helpers for Task 4 metrics."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass
class SummaryStats:
    mean: float
    median: float
    iqr_low: float
    iqr_high: float
    ci_low: float
    ci_high: float
    n: int


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = q * (len(sorted_values) - 1)
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = index - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    count = len(values)
    resample_means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(count)] for _ in range(count)]
        resample_means.append(statistics.fmean(resample))
    resample_means.sort()
    alpha = 1 - confidence
    return (
        _percentile(resample_means, alpha / 2),
        _percentile(resample_means, 1 - alpha / 2),
    )


def summarize(
    values: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> SummaryStats:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    sorted_values = sorted(values)
    ci_low, ci_high = bootstrap_ci(
        values, n_resamples=n_resamples, confidence=confidence, seed=seed
    )
    return SummaryStats(
        mean=statistics.fmean(values),
        median=statistics.median(values),
        iqr_low=_percentile(sorted_values, 0.25),
        iqr_high=_percentile(sorted_values, 0.75),
        ci_low=ci_low,
        ci_high=ci_high,
        n=len(values),
    )


@dataclass
class BinnedPoint:
    bin_center: float
    stats: SummaryStats


def aggregate_curve_by_bins(
    samples_xy: list[list[tuple[float, float]]],
    n_bins: int = 20,
    seed: int = 42,
) -> list[BinnedPoint]:
    """Bin within each sample, then aggregate one within-bin mean per sample.

    A long trace therefore cannot outweigh a short trace merely because it has
    more checkpoints in a normalized-progress bin. Values outside ``[0, 1]``
    are clamped; bins with no contributing sample are omitted.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bin_values: list[list[float]] = [[] for _ in range(n_bins)]
    for sample_points in samples_xy:
        sample_bins: list[list[float]] = [[] for _ in range(n_bins)]
        for x, y in sample_points:
            clamped = min(1.0, max(0.0, x))
            bin_index = min(int(clamped * n_bins), n_bins - 1)
            sample_bins[bin_index].append(y)
        for index, values in enumerate(sample_bins):
            if values:
                bin_values[index].append(statistics.fmean(values))

    return [
        BinnedPoint(
            bin_center=(index + 0.5) / n_bins,
            stats=summarize(values, seed=seed),
        )
        for index, values in enumerate(bin_values)
        if values
    ]
