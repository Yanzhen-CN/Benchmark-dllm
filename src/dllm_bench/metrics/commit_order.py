"""Section 4.4: Commit-order tau.

Kendall's tau_b between final valid token position and final_stable_step,
computed per sample over sliding windows of size 4/8/16/32/64, then
aggregated (Mean, Median, IQR, Bootstrap 95% CI) per ``Model x Config x
Dataset``.

Implemented without a SciPy dependency (windows are at most 64 tokens, so the
O(n^2) pairwise comparison is negligible) to keep this pure-math module
importable without the optional ``hf``/heavier extras installed.
"""

from __future__ import annotations

from collections import Counter

from .stats_utils import SummaryStats, summarize

DEFAULT_WINDOW_SIZES = (4, 8, 16, 32, 64)


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Kendall's tau_b. Returns 0.0 in the degenerate all-tied case (matches
    the "no discernible order" reading rather than propagating NaN)."""
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 2:
        raise ValueError("need at least 2 observations")

    concordant_minus_discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sign = dx * dy
            if sign > 0:
                concordant_minus_discordant += 1
            elif sign < 0:
                concordant_minus_discordant -= 1

    n0 = n * (n - 1) // 2
    n1 = sum(t * (t - 1) // 2 for t in Counter(x).values())
    n2 = sum(t * (t - 1) // 2 for t in Counter(y).values())

    denom = ((n0 - n1) * (n0 - n2)) ** 0.5
    if denom == 0:
        return 0.0
    return concordant_minus_discordant / denom


def commit_order_tau_windows(
    positions: list[int],
    final_stable_steps: list[int],
    window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
) -> dict[int, list[float]]:
    """Per-window-size tau_b values for one sample (non-overlapping windows,
    trailing partial window dropped)."""
    if len(positions) != len(final_stable_steps):
        raise ValueError("positions and final_stable_steps must have the same length")

    n = len(positions)
    result: dict[int, list[float]] = {}
    for window in window_sizes:
        if window > n:
            continue
        taus = []
        for start in range(0, n - window + 1, window):
            end = start + window
            taus.append(
                kendall_tau_b(positions[start:end], final_stable_steps[start:end])
            )
        result[window] = taus
    return result


def aggregate_commit_order(
    per_sample_window_taus: list[dict[int, list[float]]], seed: int = 42
) -> dict[int, SummaryStats]:
    """Flatten every sample's per-window tau values by window size and summarize."""
    by_window: dict[int, list[float]] = {}
    for sample_result in per_sample_window_taus:
        for window, taus in sample_result.items():
            by_window.setdefault(window, []).extend(taus)
    return {window: summarize(values, seed=seed) for window, values in by_window.items() if values}
