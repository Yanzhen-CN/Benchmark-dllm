"""Shared abstractions for the six formal scorers plus legacy IFEval support.

Every concrete dataset (``gsm8k.py``, ``mbpp.py``, ``structeval_t.py``,
``ifeval.py``, ``sudoku.py``, ``ruler.py``, ``hellobench.py``) exposes a
``Dataset`` subclass whose ``score(sample, output_text)`` returns a
``ScoreResult`` with a primary task score in ``[0, 1]`` (section 1's "统一到
[0,1]" requirement for the quality-resource metrics in Part 3) plus whatever
auxiliary metrics that dataset's row in section 1's table calls for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    sample_id: str
    prompt: str
    reference: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    primary_score: float
    """Main task score, normalized to [0, 1] (accuracy, pass@1, complete-correct
    rate, etc. — whatever section 1 lists as that dataset's primary metric)."""
    aux: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    """Whether the output could be parsed/interpreted at all (e.g. an answer
    was extractable, code was syntactically parseable)."""
    complete: bool = True
    """Whether the output looks like it finished (not cut off mid-token/mid-
    structure) rather than truncated."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.primary_score <= 1.0:
            raise ValueError(f"primary_score must be in [0, 1], got {self.primary_score}")


class Dataset(ABC):
    name: str

    @abstractmethod
    def load_samples(self, n: int | None = None) -> list[Sample]:
        """Load (or synthesize) up to ``n`` samples. ``n=None`` loads all."""

    @abstractmethod
    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        """Score one generated output against its sample's reference."""

    def aggregate(self, results: list[ScoreResult]) -> dict[str, float]:
        """Default aggregation: mean primary score + mean of every aux key
        that appears on every result, plus valid/complete rates."""
        if not results:
            raise ValueError("cannot aggregate an empty result list")
        n = len(results)
        summary = {
            f"{self.name}_score": sum(r.primary_score for r in results) / n,
            "valid_rate": sum(1 for r in results if r.valid) / n,
            "complete_rate": sum(1 for r in results if r.complete) / n,
        }
        aux_keys = set.intersection(*(set(r.aux) for r in results)) if results else set()
        for key in aux_keys:
            summary[key] = sum(r.aux[key] for r in results) / n
        return summary

    def aggregate_records(
        self, samples: list[Sample], results: list[ScoreResult]
    ) -> dict[str, float]:
        """Aggregate with sample metadata available for stratified datasets."""
        if len(samples) != len(results):
            raise ValueError("samples and results must have the same length")
        return self.aggregate(results)
