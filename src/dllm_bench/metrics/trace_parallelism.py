"""Section 4.2: Forward Effective Parallelism.

    final_stable_step(i) = min{t : x_i^(t) = x_i^(T) and for all u >= t, x_i^(u) = x_i^(T)}
    EffectiveTokensPerForward(t) = |{i : final_stable_step(i) = t}|

Reported alongside Mean/Peak Effective Tokens per Forward and Early/Middle/
Late Finalization Share, on a Normalized Forward Progress x-axis (Appendix C:
"AR decode is a ~1 token/forward parallelism reference").
"""

from __future__ import annotations

from .stats_utils import SummaryStats, summarize


def compute_final_stable_steps(token_id_sequences: list[list[int]]) -> list[int]:
    """``token_id_sequences[t][i]`` is the token id at position i after forward t.

    Shorter early lists are allowed (AR decoding grows one position per
    forward); a not-yet-created position is treated like a mask. Returns, for
    each final position i, the earliest forward from which its final token is
    present and never changes again.
    """
    total_steps = len(token_id_sequences)
    if total_steps == 0:
        return []
    n_positions = len(token_id_sequences[-1])
    final_values = token_id_sequences[-1]

    stable_steps = []
    for i in range(n_positions):
        stable_from = total_steps - 1
        for t in range(total_steps - 1, -1, -1):
            current = token_id_sequences[t][i] if i < len(token_id_sequences[t]) else None
            if current == final_values[i]:
                stable_from = t
            else:
                break
        stable_steps.append(stable_from)
    return stable_steps


def effective_tokens_per_forward(final_stable_steps: list[int], num_steps: int) -> dict[int, int]:
    counts = {t: 0 for t in range(num_steps)}
    for t in final_stable_steps:
        counts[t] = counts.get(t, 0) + 1
    return counts


def normalized_forward_progress(step_index: int, num_steps: int) -> float:
    if num_steps <= 1:
        return 0.0
    return step_index / (num_steps - 1)


def finalization_share(final_stable_steps: list[int], num_steps: int) -> dict[str, float]:
    """Fraction of tokens whose final_stable_step falls in the early/middle/late
    third of Normalized Forward Progress ([0, 1/3), [1/3, 2/3), [2/3, 1])."""
    if not final_stable_steps:
        raise ValueError("final_stable_steps must be non-empty")
    counts = {"early": 0, "middle": 0, "late": 0}
    for t in final_stable_steps:
        progress = normalized_forward_progress(t, num_steps)
        if progress < 1 / 3:
            counts["early"] += 1
        elif progress < 2 / 3:
            counts["middle"] += 1
        else:
            counts["late"] += 1
    n = len(final_stable_steps)
    return {k: v / n for k, v in counts.items()}


def mean_peak_effective_tokens_per_forward(
    final_stable_steps: list[int], num_steps: int
) -> tuple[float, int]:
    counts = effective_tokens_per_forward(final_stable_steps, num_steps)
    values = list(counts.values())
    mean_val = sum(values) / len(values) if values else 0.0
    peak_val = max(values) if values else 0
    return mean_val, peak_val


def summarize_effective_tokens_per_forward(
    per_sample_counts: list[list[int]], seed: int = 42
) -> SummaryStats:
    """Aggregate mean-effective-tokens-per-forward across samples with
    Mean/Median/IQR/Bootstrap 95% CI (design doc 4.2)."""
    per_sample_means = [
        sum(counts) / len(counts) if counts else 0.0 for counts in per_sample_counts
    ]
    return summarize(per_sample_means, seed=seed)
