"""Section 2.1/2.2: length-retention ratios for long-input and long-output.

    Context Retention               = Score_{1.0x context} / Score_{0.5x context}
    LongOutput Quality Retention    = HelloEval_{4K} / HelloEval_{2K}
"""

from __future__ import annotations


def context_retention(score_1x_context: float, score_0_5x_context: float) -> float:
    if score_0_5x_context <= 0:
        raise ValueError("score_0_5x_context must be positive")
    return score_1x_context / score_0_5x_context


def long_output_quality_retention(helloeval_4k: float, helloeval_2k: float) -> float:
    if helloeval_2k <= 0:
        raise ValueError("helloeval_2k must be positive")
    return helloeval_4k / helloeval_2k
