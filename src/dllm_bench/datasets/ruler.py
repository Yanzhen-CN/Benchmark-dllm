"""RULER: Accuracy, plus Context Retention, Position Robustness, Completion,
Truncation (section 1 / 2.1).

Context Retention reuses :func:`dllm_bench.metrics.long_context.context_retention`
across a model's 0.5x/1.0x max-context runs; Completion/Truncation are derived
from each run's :class:`~dllm_bench.interfaces.RunStatus` at the orchestrator
level (a run that hit ``RunStatus.TRUNCATED`` did not finish naturally), not
computed here. This module owns per-sample accuracy and the Position
Robustness aggregate across front/middle/back placements.

:func:`build_niah_sample` is a **synthetic placeholder** generator (no
internet/dataset-file access assumed) standing in for the official RULER
NIAH/Multi-hop/Aggregation task templates — swap it for the real task bank
before running formal numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from .base import Dataset, Sample, ScoreResult

TaskType = Literal["niah", "multi_hop", "aggregation"]
Position = Literal["front", "middle", "back"]


@dataclass
class RulerReference:
    task_type: TaskType
    position: Position
    required_answers: list[str]
    context_length: int = 0


class RulerDataset(Dataset):
    name = "ruler"

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples = samples or []

    def load_samples(self, n: int | None = None) -> list[Sample]:
        return self._samples[:n] if n is not None else list(self._samples)

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: RulerReference = sample.reference
        lowered = output_text.lower()
        hits = [a for a in ref.required_answers if a.lower() in lowered]
        exact = bool(ref.required_answers) and len(hits) == len(ref.required_answers)
        partial_rate = len(hits) / len(ref.required_answers) if ref.required_answers else 1.0
        return ScoreResult(
            primary_score=1.0 if exact else 0.0,
            aux={"partial_match_rate": partial_rate},
            valid=True,
            complete=bool(output_text.strip()),
        )


def position_robustness(scores_by_position: dict[str, float]) -> float:
    """worst-position score / best-position score, in [0, 1] (1 = fully robust)."""
    if not scores_by_position:
        raise ValueError("scores_by_position must be non-empty")
    values = list(scores_by_position.values())
    best = max(values)
    if best == 0:
        return 0.0
    return min(values) / best


_FILLER_SENTENCES = [
    "The weather in the valley stayed mild throughout the autumn season.",
    "Researchers catalogued dozens of species along the river basin.",
    "The committee reviewed last quarter's budget without major changes.",
    "A new bridge connects the two districts on either side of the canal.",
    "Local markets reported steady demand for seasonal produce this year.",
]


def build_niah_sample(
    sample_id: str,
    needle_value: str,
    position: Position,
    num_filler_sentences: int,
    seed: int = 42,
) -> Sample:
    """Synthetic Needle-In-A-Haystack sample: one needle sentence placed at
    ``position`` among ``num_filler_sentences`` unrelated filler sentences."""
    rng = random.Random(seed)
    fillers = [rng.choice(_FILLER_SENTENCES) for _ in range(num_filler_sentences)]
    needle_sentence = f"The secret code is {needle_value}."

    if position == "front":
        sentences = [needle_sentence, *fillers]
    elif position == "back":
        sentences = [*fillers, needle_sentence]
    else:
        mid = len(fillers) // 2
        sentences = [*fillers[:mid], needle_sentence, *fillers[mid:]]

    context = " ".join(sentences)
    prompt = f"{context}\n\nQuestion: What is the secret code?"
    return Sample(
        sample_id=sample_id,
        prompt=prompt,
        reference=RulerReference(
            task_type="niah",
            position=position,
            required_answers=[needle_value],
            context_length=len(context.split()),
        ),
    )
