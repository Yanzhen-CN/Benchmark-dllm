"""d1-compatible 4x4 Sudoku evaluation for general dLLM checkpoints.

The source, zero-shot prompt, answer tags, and blank-cell metric follow the
official d1 evaluation.  This suite freezes a deterministic 100-row subset of
d1's 500-row test CSV and additionally reports strict whole-puzzle validity so
partial cell accuracy is never mistaken for a solved Sudoku.
"""

from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .base import Dataset, Sample, ScoreResult
from .remote import ensure_download


SUDOKU4_SOURCE_REVISION = "6f5abf5ca8a58c6e08bbf06d412ad260dca6dbd3"
SUDOKU4_PROTOCOL_REVISION = "direct-copy-fill-raw-4x4-v5"
SUDOKU4_REASONING_PROTOCOL_REVISION = "d1-zero-shot-4x4-v1"
SUDOKU4_SOURCE_SHA256 = "ef86c7c28ebef88484d85fda59b3909a7b621241aa1abf36343437dbc4a3ffb6"
SUDOKU4_SOURCE_URL = (
    "https://raw.githubusercontent.com/dllm-reasoning/d1/"
    f"{SUDOKU4_SOURCE_REVISION}/dataset/4x4_test_sudoku.csv"
)

SUDOKU4_SYSTEM_PROMPT = """Solve this 4x4 Sudoku puzzle: {puzzle}, where '0' represents an empty cell.
Directly output the COMPLETE 16-character string answer. Copy the puzzle to the output and replace every 0 with the correct digit.
Your output must be exactly 16 digits using only 1-4 and nothing else."""

SUDOKU4_REASONING_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>"""

_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|\Z)", re.DOTALL | re.IGNORECASE)
_D1_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_COMPACT_RE = re.compile(r"(?<![0-9])([1-4]{16})(?![0-9])")


@dataclass
class Sudoku4Reference:
    puzzle: str
    solution: str


def _reasoning_enabled(configured: bool | None = None) -> bool:
    if configured is not None:
        return bool(configured)
    return os.environ.get("DLLM_BENCH_ENABLE_REASONING", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def format_sudoku4_prompt(
    puzzle: str, enable_reasoning: bool | None = None
) -> str:
    if _reasoning_enabled(enable_reasoning):
        return (
            f"{SUDOKU4_REASONING_PROMPT}\n\n"
            f"Solve the following Sudoku puzzle: {puzzle}\n"
        )
    return SUDOKU4_SYSTEM_PROMPT.format(puzzle=puzzle)


def extract_sudoku4_answer(text: str) -> tuple[str | None, bool, bool]:
    """Use d1's answer-tag precedence with a tolerant final-16 fallback."""
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        payload = re.sub(r"\s", "", matches[-1].group(1))
        complete = "</answer>" in text[matches[-1].start() :].lower()
        if re.fullmatch(r"[1-4]{16}", payload):
            return payload, True, complete
        compact = _COMPACT_RE.findall(payload)
        return (compact[-1] if compact else None), True, complete
    compact = _COMPACT_RE.findall(text)
    return (compact[-1] if compact else None), False, False


def _extract_d1_answer(text: str) -> str | None:
    """Mirror d1's case-sensitive, closed-tag answer parser."""
    matches = _D1_ANSWER_RE.findall(text)
    return re.sub(r"\s", "", matches[-1].strip()) if matches else None


def sudoku4_blank_cell_accuracy(
    prediction: str | None, puzzle: str, solution: str
) -> float:
    blank_indices = [index for index, value in enumerate(puzzle) if value == "0"]
    if not blank_indices:
        return float(prediction == solution)
    if prediction is None or not prediction:
        return 0.0
    # d1 pads short payloads and truncates long payloads before scoring only
    # the originally blank cells. Preserve this behavior for its named metric.
    prediction = (prediction + "0" * 16)[:16]
    return sum(prediction[index] == solution[index] for index in blank_indices) / len(
        blank_indices
    )


def is_valid_sudoku4(solution: str | None, puzzle: str) -> bool:
    if solution is None or not re.fullmatch(r"[1-4]{16}", solution):
        return False
    if any(clue != "0" and solution[index] != clue for index, clue in enumerate(puzzle)):
        return False
    grid = [solution[row * 4 : (row + 1) * 4] for row in range(4)]
    required = set("1234")
    if any(set(row) != required for row in grid):
        return False
    if any({grid[row][column] for row in range(4)} != required for column in range(4)):
        return False
    return all(
        {
            grid[row][column]
            for row in range(box_row, box_row + 2)
            for column in range(box_column, box_column + 2)
        }
        == required
        for box_row in (0, 2)
        for box_column in (0, 2)
    )


class Sudoku4Dataset(Dataset):
    name = "sudoku4"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        sample_count: int = 100,
        seed: int = 42,
        enable_reasoning: bool | None = None,
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        self._sample_count = int(sample_count)
        self._seed = int(seed)
        self._enable_reasoning = _reasoning_enabled(enable_reasoning)

    def load_samples(self, n: int | None = None) -> list[Sample]:
        if self._samples is not None:
            samples = list(self._samples)
        else:
            source = ensure_download(
                "sudoku4",
                "4x4_test_sudoku.csv",
                url=SUDOKU4_SOURCE_URL,
                sha256=SUDOKU4_SOURCE_SHA256,
            )
            samples = _load_d1_samples(
                source, enable_reasoning=self._enable_reasoning
            )
            rng = random.Random(self._seed)
            rng.shuffle(samples)
            if self._sample_count <= 0 or self._sample_count > len(samples):
                raise ValueError(
                    f"sudoku4 sample_count must be in [1, {len(samples)}]"
                )
            samples = [
                replace(
                    sample,
                    meta={
                        **sample.meta,
                        "formal_subset": True,
                        "formal_subset_seed": self._seed,
                    },
                )
                for sample in samples[: self._sample_count]
            ]
        return samples[:n] if n is not None else samples

    def preparation_signature(self) -> dict[str, object]:
        return {
            "source_revision": SUDOKU4_SOURCE_REVISION,
            "protocol_revision": (
                SUDOKU4_REASONING_PROTOCOL_REVISION
                if self._enable_reasoning
                else SUDOKU4_PROTOCOL_REVISION
            ),
            "enable_reasoning": self._enable_reasoning,
            "source_sha256": SUDOKU4_SOURCE_SHA256,
            "sample_count": self._sample_count,
            "seed": self._seed,
        }

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        reference: Sudoku4Reference = sample.reference
        d1_prediction = _extract_d1_answer(output_text)
        prediction, marker_present, marker_complete = extract_sudoku4_answer(output_text)
        if (
            not self._enable_reasoning
            and re.fullmatch(r"[1-4]{16}", output_text.strip())
        ):
            d1_prediction = output_text.strip()
        cell_accuracy = sudoku4_blank_cell_accuracy(
            d1_prediction, reference.puzzle, reference.solution
        )
        puzzle_success = is_valid_sudoku4(prediction, reference.puzzle)
        exact = prediction == reference.solution
        format_valid = prediction is not None
        clue_indices = [
            index for index, value in enumerate(reference.puzzle) if value != "0"
        ]
        clue_rate = (
            sum(
                prediction is not None
                and prediction[index] == reference.puzzle[index]
                for index in clue_indices
            )
            / len(clue_indices)
            if clue_indices
            else 1.0
        )
        return ScoreResult(
            # d1's published evaluator averages correctness over original blanks.
            primary_score=cell_accuracy,
            aux={
                "d1_blank_cell_accuracy": cell_accuracy,
                "puzzle_success_rate": float(puzzle_success),
                "reference_exact_match": float(exact),
                "given_preservation_rate": clue_rate,
                "answer_format_valid": float(format_valid),
                "answer_marker_present_rate": float(marker_present),
                "answer_marker_complete_rate": float(marker_complete),
            },
            valid=format_valid,
            complete=format_valid,
        )

    def aggregate(self, results: list[ScoreResult]) -> dict[str, float]:
        summary = super().aggregate(results)
        summary["d1_blank_cell_accuracy"] = summary["sudoku4_score"]
        summary["puzzle_success_rate"] = sum(
            result.aux["puzzle_success_rate"] for result in results
        ) / len(results)
        return summary


def _load_d1_samples(
    path: Path, *, enable_reasoning: bool | None = None
) -> list[Sample]:
    reasoning = _reasoning_enabled(enable_reasoning)
    protocol_revision = (
        SUDOKU4_REASONING_PROTOCOL_REVISION
        if reasoning
        else SUDOKU4_PROTOCOL_REVISION
    )
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        if rows.fieldnames != ["Puzzle", "Solution"]:
            raise ValueError(f"unexpected d1 Sudoku header: {rows.fieldnames!r}")
        for index, row in enumerate(rows):
            puzzle = str(row["Puzzle"])
            solution = str(row["Solution"])
            if not re.fullmatch(r"[0-4]{16}", puzzle):
                raise ValueError(f"invalid d1 puzzle at row {index}")
            if not re.fullmatch(r"[1-4]{16}", solution):
                raise ValueError(f"invalid d1 solution at row {index}")
            if puzzle.count("0") != 8:
                raise ValueError(f"d1 row {index} does not contain exactly 8 blanks")
            samples.append(
                Sample(
                    sample_id=f"sudoku4-d1-{index:04d}",
                    prompt=format_sudoku4_prompt(
                        puzzle, enable_reasoning=reasoning
                    ),
                    reference=Sudoku4Reference(puzzle, solution),
                    meta={
                        "source": "dllm-reasoning/d1",
                        "source_revision": SUDOKU4_SOURCE_REVISION,
                        "source_index": index,
                        "protocol_revision": protocol_revision,
                        "prompt_protocol": (
                            "d1 official zero-shot reasoning"
                            if reasoning
                            else "direct raw copy-and-fill"
                        ),
                        "enable_reasoning": reasoning,
                        "blank_count": 8,
                        "difficulty_stratified": False,
                    },
                )
            )
    if len(samples) != 500:
        raise ValueError(f"expected 500 d1 Sudoku rows, found {len(samples)}")
    return samples
