"""Pinned, minimal ports of upstream primary-metric functions.

Only task-primary behavior belongs here. Benchmark-specific answer-position,
trace, style, and audit metrics must stay in their dataset modules/``aux``.

Sources:
- lm-evaluation-harness ``gsm8k-cot.yaml`` and ``RegexFilter`` at
  f4d4b3de3ee6741a7151a9fe74945ee515262f4c
- dllm-reasoning/d1 ``eval/sudoku.py::validate_sudoku`` at
  6f5abf5ca8a58c6e08bbf06d412ad260dca6dbd3
- HKUNLP/diffusion-vs-ar ``src/llmtuner/tuner/core/metric.py::compute_acc``
  at 6743981a4ba42062c95279e590f3991de3985581
- NVIDIA/RULER ``scripts/eval/synthetic/constants.py::string_match_all`` at
  c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
"""

from __future__ import annotations

import re
from collections.abc import Sequence


GSM8K_FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
GSM8K_EXACT_IGNORES = (r",", r"\$", r"(?s).*#### ", r"\.$")
def gsm8k_flexible_extract(text: str) -> str | None:
    """lm-eval RegexFilter(group_select=-1), returning ``None`` for fallback."""
    matches = GSM8K_FLEXIBLE_PATTERN.findall(text)
    if not matches:
        return None
    selected = [part for part in matches[-1] if part]
    return selected[0].strip() if selected else None


def gsm8k_exact_match(prediction: str | None, reference: str) -> bool:
    """lm-eval exact_match with gsm8k_cot's configured ignore regexes."""
    if prediction is None:
        return False
    prediction_text = prediction
    reference_text = reference
    for pattern in GSM8K_EXACT_IGNORES:
        prediction_text = re.sub(pattern, "", prediction_text)
        reference_text = re.sub(pattern, "", reference_text)
    return prediction_text.lower() == reference_text.lower()


def d1_sudoku_blank_cell_accuracy(
    prediction: str | None, puzzle: str, solution: str
) -> float:
    """d1 ``validate_sudoku`` accuracy, without its diagnostic ``print`` calls."""
    empty_indices = [index for index in range(16) if puzzle[index] == "0"]
    empty_cells = len(empty_indices)
    if prediction is None or len(prediction) == 0:
        return 0.0
    if len(prediction) < 16:
        prediction = prediction + "0" * (16 - len(prediction))
    elif len(prediction) > 16:
        prediction = prediction[:16]
    correct_cells = sum(
        1 for index in empty_indices if prediction[index] == solution[index]
    )
    return correct_cells / empty_cells if empty_cells > 0 else 0.0


def ye_sudoku_sequence_accuracy(
    prediction: str | None, reference: str
) -> float:
    """Ye et al. Sudoku ``compute_acc`` after answer-region normalization.

    The upstream evaluator compares the complete predicted action sequence to
    the complete label sequence and emits one boolean per puzzle. Our answer
    adapter first removes presentation-only separators, so this minimal port
    receives canonical 81-digit strings rather than tokenizer-specific spaces.
    """
    if prediction is None:
        return 0.0
    return float(prediction == reference)


def ruler_string_match_all(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """NVIDIA RULER's official percentage score, including 2-decimal rounding."""
    score = sum(
        sum(1.0 if ref.lower() in prediction.lower() else 0.0 for ref in refs)
        / len(refs)
        for prediction, refs in zip(predictions, references, strict=True)
    ) / len(predictions) * 100
    return round(score, 2)
