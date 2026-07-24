"""Shared aggregation helpers reused across Part 4 metrics.

Every Part 4 metric (4.2 parallelism, 4.3 strategy score, 4.4 commit-order,
4.5 certainty) reports the same shape of summary: Mean/Median, IQR, and a
Bootstrap 95% CI, aggregated per ``Model x Config x Dataset``. This module
implements that shape once.
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
