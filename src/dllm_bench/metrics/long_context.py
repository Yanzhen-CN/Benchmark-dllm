"""Section 2.1/2.2: length-retention ratios for long-input and long-output.

    Context Retention               = Score_{model max} / Score_{common 8K}
    LongOutput Quality Retention    = HelloEval_{4K} / HelloEval_{2K}
"""

from __future__ import annotations


def context_retention(score_model_max: float, score_common: float) -> float:
    if score_common <= 0:
        raise ValueError("score_common must be positive")
    return score_model_max / score_common


def long_output_quality_retention(helloeval_4k: float, helloeval_2k: float) -> float:
    if helloeval_2k <= 0:
        raise ValueError("helloeval_2k must be positive")
    return helloeval_4k / helloeval_2k
