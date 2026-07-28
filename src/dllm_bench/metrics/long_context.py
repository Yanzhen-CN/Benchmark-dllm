"""Section 2.1/2.2: length-retention ratios for long-input and long-output.

    Context Retention               = Score_{model max} / Score_{common 8K}
    LongOutput Quality Retention    = ObjectiveQuality_{4K} / ObjectiveQuality_{2K}
"""

from __future__ import annotations


def context_retention(score_model_max: float, score_common: float) -> float:
    if score_common <= 0:
        raise ValueError("score_common must be positive")
    return score_model_max / score_common


def long_output_quality_retention(quality_4k: float, quality_2k: float) -> float:
    if quality_2k <= 0:
        raise ValueError("quality_2k must be positive")
    return quality_4k / quality_2k
