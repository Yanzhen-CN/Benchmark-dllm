"""Shared aggregation helpers reused across Part 4 metrics.

Every Part 4 metric (4.2 parallelism, 4.3 strategy score, 4.4 commit-order,
4.5 certainty) reports the same shape of summary: Mean/Median, IQR, and a
Bootstrap 95% CI, aggregated per ``Model x Config x Dataset``. This module
implements that shape once — including the "dataset-level average curve"
shape (Appendix C / design doc 4.2.1 & 4.2.4): different samples have
different forward counts, so a curve can't just be averaged index-by-index;
instead every sample's (x, y) points are binned by a fixed x-grid over
[0, 1] and each bin reports Median + Bootstrap 95% CI across whatever points
(from however many samples) landed in it.
"""

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
    """Linear-interpolation percentile, q in [0, 1]."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile-method bootstrap CI for the mean of ``values``."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    resample_means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        resample_means.append(statistics.fmean(resample))
    resample_means.sort()
    alpha = 1 - confidence
    lo = _percentile(resample_means, alpha / 2)
    hi = _percentile(resample_means, 1 - alpha / 2)
    return lo, hi


def summarize(
    values: list[float], n_resamples: int = 2000, confidence: float = 0.95, seed: int = 42
) -> SummaryStats:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    sorted_values = sorted(values)
    ci_low, ci_high = bootstrap_ci(values, n_resamples=n_resamples, confidence=confidence, seed=seed)
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
    """Bins the union of every sample's (x, y) points into `n_bins`
    fixed-width bins over x in [0, 1] (Appendix C: "曲线按固定 progress bins
    汇总"), and within each non-empty bin reports Median + Bootstrap 95% CI
    across every (sample, point) that landed there — a sample can contribute
    multiple points to one bin, or none, depending on its own forward count;
    that's fine, this is a dataset-level average, not a per-sample one.

    x values outside [0, 1] are clamped into the nearest edge bin. Empty
    bins (no sample had a point there) are omitted, not zero-filled.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bin_values: list[list[float]] = [[] for _ in range(n_bins)]
    for sample_points in samples_xy:
        for x, y in sample_points:
            clamped = min(1.0, max(0.0, x))
            bin_index = min(int(clamped * n_bins), n_bins - 1)
            bin_values[bin_index].append(y)

    results: list[BinnedPoint] = []
    for i, values in enumerate(bin_values):
        if not values:
            continue
        bin_center = (i + 0.5) / n_bins
        results.append(BinnedPoint(bin_center=bin_center, stats=summarize(values, seed=seed)))
    return results
