"""Section 4.5: Remaining-token Certainty.

    H_i(t) = -sum_v p_i^(t)(v) log p_i^(t)(v) / log(|V|)
    Certainty(t) = 1 - mean_{i in R_t} H_i(t)
    AcceptedRatio(t) = |A_t| / FinalValidLength

Main plot is Certainty(t) against AcceptedRatio(t); Top-1 Confidence is
reported as an auxiliary curve.
"""

from __future__ import annotations

import math
import statistics

from ..interfaces import PositionState, TraceStep


def normalized_entropy(probs: list[float], vocab_size: int) -> float:
    """H_i(t) for one still-masked position, given its output distribution.

    Utility for adapters that have direct access to per-position logits/probs
    (e.g. a real diffusion model's denoiser head) and want to compute the
    scalar entropy to store in ``TraceStep.entropy_by_position``.
    """
    if vocab_size <= 1:
        return 0.0
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy / math.log(vocab_size)


def certainty(entropies: list[float]) -> float:
    """Certainty(t) over the remaining (still-masked) positions R_t.

    Defined as 1.0 when R_t is empty (nothing left uncertain)."""
    if not entropies:
        return 1.0
    return 1 - statistics.fmean(entropies)


def accepted_ratio(num_accepted: int, final_valid_length: int) -> float:
    if final_valid_length <= 0:
        raise ValueError("final_valid_length must be positive")
    return num_accepted / final_valid_length


def build_certainty_curve(
    trace: list[TraceStep], final_valid_length: int
) -> list[tuple[float, float, float]]:
    """Per-step (AcceptedRatio, Certainty, mean Top-1 Confidence) triples.

    Positions still masked at a step but lacking a recorded entropy (backend
    did not expose a distribution for them) are excluded from that step's
    Certainty average rather than assumed maximally uncertain.
    """
    curve = []
    for step in trace:
        num_accepted = sum(1 for s in step.position_states if s == PositionState.ACCEPTED)
        ratio = accepted_ratio(num_accepted, final_valid_length)

        entropies = list(step.entropy_by_position.values()) if step.entropy_by_position else []
        cert = certainty(entropies)

        top1_values = (
            list(step.top1_confidence_by_position.values())
            if step.top1_confidence_by_position
            else []
        )
        top1_mean = statistics.fmean(top1_values) if top1_values else 1.0

        curve.append((ratio, cert, top1_mean))
    return curve
