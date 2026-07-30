"""RULER-inspired controlled long-context diagnostic.

The formal matrix uses one controlled 4096-token input for every model.
A separate one-sample dataset probes half of each model's declared context
without mixing that capacity result into formal quality/resource aggregates.
Completion/Truncation are derived
from each run's :class:`~dllm_bench.interfaces.RunStatus` at the orchestrator
level (a run that hit ``RunStatus.TRUNCATED`` did not finish naturally), not
computed here. This module owns per-sample accuracy and the Position
Robustness aggregate across front/middle/back placements.

The task families and ``string_match_all`` scorer follow RULER's design, but
the explicit front/middle/back strata and reduced three-task bank are this
project's controlled diagnostic rather than NVIDIA's official 13-task suite.
Prepared prompts use a whitespace proxy; every local model adapter performs
last-mile fitting with that checkpoint's chat template and tokenizer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from .base import Dataset, Sample, ScoreResult

TaskType = Literal["niah", "multi_hop", "aggregation"]
Position = Literal["front", "middle", "back"]

RULER_UPSTREAM_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
RULER_DIAGNOSTIC_REVISION = "ruler-inspired-controlled-v2"
DEFAULT_CONTEXT_WINDOWS = (8192, 32768, 40960, 262144)


@dataclass
class RulerReference:
    task_type: TaskType
    position: Position
    required_answers: list[str]
    context_length: int = 0


class RulerDataset(Dataset):
    name = "ruler"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        context_windows: list[int] | tuple[int, ...] = DEFAULT_CONTEXT_WINDOWS,
        samples_per_context_window_position: int = 10,
        max_output_tokens: int = 64,
        seed: int = 42,
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        self._context_windows = tuple(int(value) for value in context_windows)
        self._samples_per_group = int(samples_per_context_window_position)
        self._max_output_tokens = int(max_output_tokens)
        self._seed = int(seed)

    def load_samples(self, n: int | None = None) -> list[Sample]:
        samples = (
            list(self._samples)
            if self._samples is not None
            else generate_ruler_bank(
                self._context_windows,
                self._samples_per_group,
                self._max_output_tokens,
                self._seed,
            )
        )
        return samples[:n] if n is not None else samples

    def preparation_signature(self) -> dict[str, object]:
        """Resolved inputs that must invalidate an existing prepared bank."""
        return {
            "protocol_revision": RULER_DIAGNOSTIC_REVISION,
            "context_windows": self._context_windows,
            "samples_per_context_window_position": self._samples_per_group,
            "max_output_tokens": self._max_output_tokens,
            "seed": self._seed,
        }

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: RulerReference = sample.reference
        lowered = output_text.lower()
        hits = [a for a in ref.required_answers if a.lower() in lowered]
        exact = bool(ref.required_answers) and len(hits) == len(ref.required_answers)
        partial_rate = len(hits) / len(ref.required_answers) if ref.required_answers else 1.0
        return ScoreResult(
            # NVIDIA RULER's official string_match_all gives fractional credit
            # for each required reference found in the prediction.
            primary_score=partial_rate,
            aux={"all_answers_match": 1.0 if exact else 0.0},
            valid=True,
            complete=bool(output_text.strip()),
        )

    def aggregate_records(
        self, samples: list[Sample], results: list[ScoreResult]
    ) -> dict[str, float]:
        summary = super().aggregate_records(samples, results)
        summary["ruler_string_match_all"] = summary[f"{self.name}_score"]
        grouped: dict[tuple[int, str, str], list[float]] = {}
        for sample, result in zip(samples, results):
            window = int(sample.meta.get("context_window_tokens", sample.reference.context_length))
            key = (window, sample.reference.task_type, sample.reference.position)
            grouped.setdefault(key, []).append(result.primary_score)

        windows = sorted({key[0] for key in grouped})
        for window in windows:
            window_scores = [
                score for (group_window, _, _), scores in grouped.items()
                if group_window == window for score in scores
            ]
            summary[f"accuracy_context_{window}"] = sum(window_scores) / len(window_scores)
            for task in sorted({key[1] for key in grouped if key[0] == window}):
                task_scores = [
                    score for (group_window, group_task, _), scores in grouped.items()
                    if group_window == window and group_task == task for score in scores
                ]
                summary[f"accuracy_{task}_context_{window}"] = sum(task_scores) / len(task_scores)
            for position in sorted({key[2] for key in grouped if key[0] == window}):
                position_scores = [
                    score for (group_window, _, group_position), scores in grouped.items()
                    if group_window == window and group_position == position for score in scores
                ]
                summary[f"accuracy_{position}_context_{window}"] = (
                    sum(position_scores) / len(position_scores)
                )
            position_scores = {
                position: summary[f"accuracy_{position}_context_{window}"]
                for position in sorted({key[2] for key in grouped if key[0] == window})
            }
            summary[f"position_robustness_context_{window}"] = position_robustness(
                position_scores
            )

        if len(windows) == 2:
            common_score = summary[f"accuracy_context_{windows[0]}"]
            max_score = summary[f"accuracy_context_{windows[1]}"]
            summary["context_retention"] = max_score / common_score if common_score > 0 else 0.0
        return summary


class RulerContextProbeDataset(RulerDataset):
    """One oversized source prompt, fitted per model to its half-context target."""

    name = "ruler_context_probe"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        source_input_tokens: int = 131072,
        max_output_tokens: int = 64,
        seed: int = 42,
    ) -> None:
        self._probe_samples = list(samples) if samples is not None else None
        self._source_input_tokens = int(source_input_tokens)
        self._max_output_tokens = int(max_output_tokens)
        self._seed = int(seed)

    def load_samples(self, n: int | None = None) -> list[Sample]:
        if self._probe_samples is not None:
            samples = list(self._probe_samples)
        else:
            rng = random.Random(self._seed)
            answer = f"P{rng.randrange(10**11, 10**12)}"
            samples = [
                _make_generated_sample(
                    "ruler-context-probe-000",
                    "niah",
                    "middle",
                    [answer],
                    f"The hidden access code is {answer}.",
                    "What is the hidden access code? Answer with only the code.",
                    self._source_input_tokens + self._max_output_tokens,
                    self._max_output_tokens,
                )
            ]
        return samples[:n] if n is not None else samples

    def preparation_signature(self) -> dict[str, object]:
        return {
            "source_input_tokens": self._source_input_tokens,
            "max_output_tokens": self._max_output_tokens,
            "seed": self._seed,
        }


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


def _position_for_index(index: int, task_offset: int) -> Position:
    return ("front", "middle", "back")[(index + task_offset) % 3]  # type: ignore[return-value]


def _place_payload(filler: str, payload: str, position: Position) -> str:
    if position == "front":
        return f"{payload}\n{filler}"
    if position == "back":
        return f"{filler}\n{payload}"
    midpoint = len(filler) // 2
    boundary = filler.find(" ", midpoint)
    boundary = midpoint if boundary < 0 else boundary
    return f"{filler[:boundary]}\n{payload}\n{filler[boundary:]}"


def _fit_prompt(payload: str, question: str, target_words: int, position: Position) -> str:
    """Create a deterministic prompt using a whitespace length proxy.

    Native token counts vary by model tokenizer. The requested window and
    input target are retained in metadata for last-mile fitting before timing.
    """
    # Official RULER keeps an answer prefix after the question so generation
    # starts in answer mode instead of spending budget on explanation/refusal.
    answer_prefix = "Answer:"
    fixed_words = len(payload.split()) + len(question.split()) + 1
    filler = ("background " * max(0, target_words - fixed_words)).strip()
    return (
        f"{_place_payload(filler, payload, position)}\n\n"
        f"{question}\n{answer_prefix}"
    )


def _make_generated_sample(
    sample_id: str,
    task_type: TaskType,
    position: Position,
    answers: list[str],
    payload: str,
    question: str,
    window: int,
    max_output_tokens: int,
) -> Sample:
    target_input = window - max_output_tokens
    return Sample(
        sample_id=sample_id,
        prompt=_fit_prompt(payload, question, target_input, position),
        reference=RulerReference(task_type, position, answers, target_input),
        meta={
            "source": "dllm-bench RULER-inspired controlled diagnostic",
            "source_revision": RULER_DIAGNOSTIC_REVISION,
            "upstream_reference": "NVIDIA/RULER",
            "upstream_revision": RULER_UPSTREAM_REVISION,
            "official_ruler_compatible": False,
            "context_window_tokens": window,
            "target_input_tokens": target_input,
            "prepared_length_unit": "whitespace_proxy",
            "runtime_length_unit": "model_tokenizer_after_chat_template",
        },
    )


def generate_ruler_bank(
    context_windows: tuple[int, ...] | list[int],
    samples_per_context_window_position: int = 10,
    max_output_tokens: int = 64,
    seed: int = 42,
) -> list[Sample]:
    """Generate the complete configured RULER bank during preparation.

    Task-specific position offsets mean 10 samples per task produces exactly
    10 samples in every ``window x position`` cell (30 per context window).
    """
    count = int(samples_per_context_window_position)
    windows = sorted(set(int(value) for value in context_windows))
    if count <= 0:
        raise ValueError("RULER sample count must be positive")
    if not windows or any(window <= max_output_tokens for window in windows):
        raise ValueError("every RULER context window must exceed max_output_tokens")
    rng = random.Random(seed)
    samples: list[Sample] = []
    for window in windows:
        for task_offset, task_type in enumerate(("niah", "multi_hop", "aggregation")):
            for index in range(count):
                position = _position_for_index(index, task_offset)
                sample_id = f"ruler-{task_type}-{window}-{index:03d}"
                if task_type == "niah":
                    answer = f"R{rng.randrange(10**11, 10**12)}"
                    payload = f"The hidden access code is {answer}."
                    question = "What is the hidden access code? Answer with only the code."
                    answers = [answer]
                elif task_type == "multi_hop":
                    answer = f"V{rng.randrange(100000, 999999)}"
                    variables = [f"node_{index}_{letter}" for letter in "abcd"]
                    payload = "\n".join([
                        f"{variables[0]} stores {answer}.",
                        *[
                            f"{variables[offset]} copies {variables[offset - 1]}."
                            for offset in range(1, len(variables))
                        ],
                    ])
                    question = (
                        f"What value is ultimately stored in {variables[-1]}? "
                        "Answer with only the value."
                    )
                    answers = [answer]
                else:
                    answers = [f"common_{index}_{letter}" for letter in "xyz"]
                    distractors = [f"rare_{index}_{letter}" for letter in "abcdefghijkl"]
                    words = distractors + answers * 4
                    rng.shuffle(words)
                    payload = "Word list: " + ", ".join(words) + "."
                    question = "Which three words occur most often? Return all three words."
                samples.append(_make_generated_sample(
                    sample_id, task_type, position, answers, payload, question,
                    window, max_output_tokens,
                ))
    return samples


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
