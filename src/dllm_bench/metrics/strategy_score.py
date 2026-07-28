"""Structure-vs-content generation preference for StructEval-T and MBPP.

The dataset modules first classify observable output features as framework
structure or substantive content and produce progress curves over generation
checkpoints.  This module converts the *first-formation mass* of those curves
to a pairwise order score analogous to a Kendall/Mann-Whitney concordance:

``StructureFirst = P(T_structure < T_content) + 0.5 P(T_structure = T_content)``

The result is in ``[0, 1]``: 1 means all observed framework formed before the
content, 0 means the reverse, and 0.5 means tied or order-balanced formation.
The AUC helpers remain available for plotting the two progress curves, but AUC
difference is no longer the preference score.

A sample only enters the preference score if both final progress values reach
0.5; otherwise its curves are kept but the score is N/A (``None``).
"""

from __future__ import annotations

from .stats_utils import SummaryStats, summarize

ELIGIBILITY_THRESHOLD = 0.5


def normalized_progress_series(scores: list[float], eps: float = 1e-9) -> list[float]:
    if not scores:
        raise ValueError("scores must be non-empty")
    final_score = max(scores[-1], eps)
    return [min(s / final_score, 1.0) for s in scores]


def auc_trapezoid(normalized_series: list[float]) -> float:
    """Trapezoidal-rule AUC over progress points assumed evenly spaced on [0, 1]."""
    n = len(normalized_series)
    if n == 0:
        raise ValueError("normalized_series must be non-empty")
    if n == 1:
        return normalized_series[0]
    step = 1.0 / (n - 1)
    total = 0.0
    for i in range(n - 1):
        total += step * (normalized_series[i] + normalized_series[i + 1]) / 2.0
    return total


def is_eligible(final_form_progress: float, final_content_progress: float) -> bool:
    return (
        final_form_progress >= ELIGIBILITY_THRESHOLD
        and final_content_progress >= ELIGIBILITY_THRESHOLD
    )


def first_formation_distribution(scores: list[float]) -> list[tuple[float, float]]:
    """Return ``(normalized_time, newly_formed_mass)`` pairs.

    A cumulative envelope records first formation and prevents later revision
    or temporary parse regressions from counting the same feature twice.
    """
    normalized = normalized_progress_series(scores)
    checkpoint_count = len(normalized)
    previous = 0.0
    envelope = 0.0
    distribution: list[tuple[float, float]] = []
    for index, value in enumerate(normalized):
        envelope = max(envelope, value)
        newly_formed = max(0.0, envelope - previous)
        if newly_formed > 0.0:
            time = index / (checkpoint_count - 1) if checkpoint_count > 1 else 0.0
            distribution.append((time, newly_formed))
        previous = envelope
    return distribution


def pairwise_structure_first_score(
    form_scores: list[float], content_scores: list[float]
) -> float:
    """Pair every formed structure unit with every formed content unit."""
    form_distribution = first_formation_distribution(form_scores)
    content_distribution = first_formation_distribution(content_scores)
    form_mass = sum(mass for _, mass in form_distribution)
    content_mass = sum(mass for _, mass in content_distribution)
    denominator = form_mass * content_mass
    if denominator <= 0.0:
        return 0.5

    structure_before = 0.0
    tied = 0.0
    for form_time, form_weight in form_distribution:
        for content_time, content_weight in content_distribution:
            pair_weight = form_weight * content_weight
            if form_time < content_time:
                structure_before += pair_weight
            elif form_time == content_time:
                tied += pair_weight
    score = (structure_before + 0.5 * tied) / denominator
    return max(0.0, min(1.0, score))


def strategy_score(
    form_scores: list[float],
    content_scores: list[float],
    final_form_progress: float | None = None,
    final_content_progress: float | None = None,
) -> float | None:
    """Return the Structure-First Score in ``[0, 1]`` or ``None``.

    ``final_form_progress``/``final_content_progress`` default to the raw
    (non-normalized) final entries of each series when not given explicitly.
    """
    if not form_scores or not content_scores:
        raise ValueError("form_scores and content_scores must be non-empty")

    ffp = final_form_progress if final_form_progress is not None else form_scores[-1]
    fcp = final_content_progress if final_content_progress is not None else content_scores[-1]
    if not is_eligible(ffp, fcp):
        return None
    return pairwise_structure_first_score(form_scores, content_scores)


def summarize_strategy_scores(scores: list[float | None], seed: int = 42) -> tuple[SummaryStats, float]:
    """Returns (summary over eligible scores, eligible_sample_ratio)."""
    eligible = [s for s in scores if s is not None]
    eligible_ratio = len(eligible) / len(scores) if scores else 0.0
    return summarize(eligible, seed=seed), eligible_ratio
