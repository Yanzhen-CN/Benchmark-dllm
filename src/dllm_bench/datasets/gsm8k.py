"""GSM8K: Accuracy, plus valid-answer-rate and complete-output-rate (section 1).

Answer extraction follows the common GSM8K convention: a gold-style
``#### <number>`` marker if the model emits one, otherwise the last number
that appears in the response.
"""

from __future__ import annotations

import re

from .base import Dataset, Sample, ScoreResult

_GOLD_MARKER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?%?")


def extract_final_number(text: str) -> float | None:
    gold_match = _GOLD_MARKER_RE.search(text)
    if gold_match:
        return _to_float(gold_match.group(1))

    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return _to_float(numbers[-1])


def _to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def _looks_complete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "####" in stripped:
        return True
    return stripped[-1] in ".!?\"')" or stripped[-1].isdigit()


class GSM8KDataset(Dataset):
    name = "gsm8k"

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples = samples or []

    def load_samples(self, n: int | None = None) -> list[Sample]:
        return self._samples[:n] if n is not None else list(self._samples)

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        predicted = extract_final_number(output_text)
        valid = predicted is not None
        correct = valid and abs(predicted - float(sample.reference)) < 1e-4
        return ScoreResult(
            primary_score=1.0 if correct else 0.0,
            valid=valid,
            complete=_looks_complete(output_text),
        )
