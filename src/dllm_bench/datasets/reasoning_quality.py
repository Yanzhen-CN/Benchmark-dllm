from __future__ import annotations

import re

from dllm_bench.datasets.answer_region import AnswerRegion
from dllm_bench.datasets.hellobench import repeated_segment_fraction, seq_rep_n


_LOGIC_GROUPS = (
    ("row", "行"),
    ("column", "col", "列"),
    ("box", "block", "subgrid", "宫"),
    ("candidate", "possible", "候选"),
    ("eliminate", "exclude", "conflict", "排除", "冲突"),
    ("therefore", "thus", "must", "only", "所以", "因此", "只能"),
)


def sudoku_reasoning_surface_metrics(
    output_text: str,
    answer_region: AnswerRegion,
) -> dict[str, float]:
    """Measure readable, progressive Sudoku reasoning without claiming correctness."""
    reasoning = output_text[: answer_region.start_char] if answer_region.detected else output_text
    reasoning = re.sub(r"</?(?:think|analysis)>", " ", reasoning, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]|\d+", reasoning)
    lowered = reasoning.lower()
    repetition_4 = seq_rep_n(reasoning, n=4)
    repeated_fraction = repeated_segment_fraction(reasoning)
    anti_repetition = max(0.0, 1.0 - max(repetition_4, repeated_fraction))

    covered_groups = sum(
        any(keyword in lowered for keyword in group)
        for group in _LOGIC_GROUPS
    )
    logic_coverage = covered_groups / len(_LOGIC_GROUPS)
    numbered_steps = len(re.findall(r"(?m)^\s*(?:step\s*)?\d+[.)：:]", reasoning, re.IGNORECASE))
    assignment_steps = len(
        re.findall(
            r"(?:\br\s*\d+\s*c\s*\d+\b|\bcell\s*\(?\s*\d+\s*[,x]\s*\d+\s*\)?)"
            r"[^\n]{0,40}?(?:=|is|must be|只能是|为)\s*[1-9]",
            reasoning,
            re.IGNORECASE,
        )
    )
    progression = min(1.0, (numbered_steps + assignment_steps) / 6.0)

    quality = 0.45 * anti_repetition + 0.35 * logic_coverage + 0.20 * progression
    if len(words) < 10:
        quality *= 0.25
    elif len(words) < 30:
        quality *= 0.65
    if not answer_region.detected:
        quality *= 0.50
    if answer_region.unclosed_thinking:
        quality *= 0.70

    return {
        "reasoning_surface_quality_score": quality,
        "reasoning_word_count": float(len(words)),
        "reasoning_seq_rep_4": repetition_4,
        "reasoning_repeated_segment_fraction": repeated_fraction,
        "reasoning_logic_coverage": logic_coverage,
        "reasoning_progression_score": progression,
        "reasoning_answer_transition_rate": float(answer_region.detected),
        "reasoning_unclosed_rate": float(answer_region.unclosed_thinking),
    }
