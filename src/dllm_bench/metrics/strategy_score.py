"""Section 4.3 / Appendix A.4: Structure/Constraint-vs-Content formation strategy.

    NormalizedProgress_m(t) = min(Score_m(t) / max(Score_m(T), eps), 1)
    AUC_m = trapezoid integral of NormalizedProgress_m over normalized progress p in [0, 1]
    StrategyScore = clip(50 * (1 + AUC_form - AUC_content), 0, 100)

``AUC_form`` is the Structure-progress AUC for StructEval-T or the
Constraint-progress AUC for IFEval (design doc 4.3, last paragraph).
A sample only enters the strategy score if both final progress values reach
0.5 (Appendix A.4); otherwise its curves are kept but the score is N/A,
represented here as ``None``.
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


def strategy_score(
    form_scores: list[float],
    content_scores: list[float],
    final_form_progress: float | None = None,
    final_content_progress: float | None = None,
) -> float | None:
    """Returns the 0-100 strategy score, or ``None`` ("N/A") if the sample is
    ineligible (final form/content progress below 0.5, Appendix A.4).

    ``final_form_progress``/``final_content_progress`` default to the raw
    (non-normalized) final entries of each series when not given explicitly.
    """
    form_norm = normalized_progress_series(form_scores)
    content_norm = normalized_progress_series(content_scores)

    ffp = final_form_progress if final_form_progress is not None else form_scores[-1]
    fcp = final_content_progress if final_content_progress is not None else content_scores[-1]
    if not is_eligible(ffp, fcp):
        return None

    auc_form = auc_trapezoid(form_norm)
    auc_content = auc_trapezoid(content_norm)
    raw = 50.0 * (1 + auc_form - auc_content)
    return min(100.0, max(0.0, raw))


def summarize_strategy_scores(scores: list[float | None], seed: int = 42) -> tuple[SummaryStats, float]:
    """Returns (summary over eligible scores, eligible_sample_ratio)."""
    eligible = [s for s in scores if s is not None]
    eligible_ratio = len(eligible) / len(scores) if scores else 0.0
    return summarize(eligible, seed=seed), eligible_ratio
