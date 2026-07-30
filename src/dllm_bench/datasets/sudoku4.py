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

from .answer_region import (
    AnswerRegion,
    aggregate_answer_position_metrics,
    empty_answer_region,
    locate_digit_answer,
    position_aux,
    scored_payload_aux,
    trace_position_aux,
)
from .base import Dataset, Sample, ScoreResult
from .official_metrics import (
    d1_sudoku_blank_cell_accuracy,
)
from .remote import ensure_download
from ..interfaces import GenerationResult


SUDOKU4_SOURCE_REVISION = "6f5abf5ca8a58c6e08bbf06d412ad260dca6dbd3"
SUDOKU4_PROTOCOL_REVISION = "direct-copy-fill-raw-4x4-v6"
SUDOKU4_REASONING_PROTOCOL_REVISION = "d1-zero-shot-4x4-v1"
SUDOKU4_SOURCE_SHA256 = "ef86c7c28ebef88484d85fda59b3909a7b621241aa1abf36343437dbc4a3ffb6"
SUDOKU4_SOURCE_URL = (
    "https://raw.githubusercontent.com/dllm-reasoning/d1/"
    f"{SUDOKU4_SOURCE_REVISION}/dataset/4x4_test_sudoku.csv"
)

SUDOKU4_SYSTEM_PROMPT = """Solve this 4x4 Sudoku puzzle: {puzzle}, where '0' represents an empty cell and the characters are ordered row by row, from left to right and top to bottom.
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


def locate_sudoku4_answer(text: str, *, enable_reasoning: bool) -> AnswerRegion:
    """Locate the final 16-cell submission without scoring copied puzzles.

    The d1 blank-cell metric remains the primary metric.  This extraction
    adapter handles general instruction checkpoints that may reason despite a
    direct-answer prompt: the last complete answer wins, while partial credit
    is allowed only inside an explicit ``<answer>`` block.
    """
    region = locate_digit_answer(
        text,
        expected_length=16,
        allowed_digits="1234",
        marker_pairs=(("<answer>", "</answer>"),),
        minimum_partial_length=16,
        marker_minimum_partial_length=1,
    )
    if region.detected or region.method != "not_found":
        return region

    # Some checkpoints finish with four labelled rows instead of a compact
    # string. Take the last complete four-row block; row labels are excluded.
    row_candidates: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        digits = re.findall(r"[1-4]", line)
        row_match = re.search(
            r"(?i)\brow\s*[0-4]?\s*[:=-]?\s*([1-4](?:[ \t,|/-]*[1-4]){3})\s*$",
            line.rstrip("\r\n"),
        )
        if row_match:
            row_digits = "".join(re.findall(r"[1-4]", row_match.group(1)))
            row_candidates.append(
                (offset + row_match.start(1), offset + row_match.end(1), row_digits)
            )
        elif len(digits) == 4 and not re.search(r"[A-Za-z]", line):
            first = next(index for index, char in enumerate(line) if char in "1234")
            last = max(index for index, char in enumerate(line) if char in "1234") + 1
            row_candidates.append((offset + first, offset + last, "".join(digits)))
        offset += len(line)
    if len(row_candidates) >= 4:
        for end_index in range(len(row_candidates), 3, -1):
            rows = row_candidates[end_index - 4 : end_index]
            cue = text[max(0, rows[0][0] - 200) : rows[0][0]].lower()
            if not re.search(r"(?:final\s+grid|final\s+answer|solution\s+is)", cue):
                continue
            return AnswerRegion(
                text="".join(row[2] for row in rows),
                start_char=rows[0][0],
                end_char=rows[-1][1],
                detected=True,
                method="final_four_row_grid",
            )
    return empty_answer_region(text, "sudoku4_answer_not_found")


def sudoku4_blank_cell_accuracy(
    prediction: str | None, puzzle: str, solution: str
) -> float:
    return d1_sudoku_blank_cell_accuracy(prediction, puzzle, solution)


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

    def scoring_signature(self) -> dict[str, object]:
        return {
            "upstream": "dllm-reasoning/d1:eval/sudoku.py::validate_sudoku",
            "upstream_revision": SUDOKU4_SOURCE_REVISION,
            "metric": "blank_cell_accuracy",
            "answer_extraction": "final-submission-adapter-v1",
            "direct_track_adapter": not self._enable_reasoning,
        }

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        reference: Sudoku4Reference = sample.reference
        region = locate_sudoku4_answer(
            output_text, enable_reasoning=self._enable_reasoning
        )
        d1_prediction = region.text if region.detected else None
        prediction = (
            (d1_prediction + "0" * 16)[:16] if d1_prediction else None
        )
        marker_present = "<answer>" in output_text.lower()
        marker_complete = "</answer>" in output_text.lower()
        cell_accuracy = sudoku4_blank_cell_accuracy(
            d1_prediction, reference.puzzle, reference.solution
        )
        puzzle_success = is_valid_sudoku4(prediction, reference.puzzle)
        exact = prediction == reference.solution
        source_payload = (
            output_text[region.start_char : region.end_char].strip()
            if region.detected
            else ""
        )
        format_valid = bool(re.fullmatch(r"[1-4]{16}", source_payload))
        direct_clean = bool(re.fullmatch(r"[1-4]{16}", output_text.strip()))
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
        aux = {
            "d1_blank_cell_accuracy": cell_accuracy,
            "puzzle_success_rate": float(puzzle_success),
            "reference_exact_match": float(exact),
            "given_preservation_rate": clue_rate,
            "strict_16_digit_format_rate": float(format_valid),
            "direct_answer_instruction_following_rate": float(
                marker_complete and format_valid
                if self._enable_reasoning
                else direct_clean
            ),
            "answer_marker_present_rate": float(marker_present),
            "answer_marker_complete_rate": float(marker_complete),
        }
        result = ScoreResult(
            # d1's published evaluator averages correctness over original blanks.
            primary_score=cell_accuracy,
            aux=aux,
            valid=d1_prediction is not None,
            complete=bool(d1_prediction and len(d1_prediction) >= 16),
        )
        result.aux.update(position_aux(region, output_text))
        result.aux.update(scored_payload_aux(d1_prediction or ""))
        return result

    def score_generation(
        self, sample: Sample, generation: GenerationResult
    ) -> ScoreResult:
        result = self.score(sample, generation.output_text)
        region = locate_sudoku4_answer(
            generation.output_text, enable_reasoning=self._enable_reasoning
        )
        result.aux.update(trace_position_aux(region, generation.trace))
        result.aux.update(self.trace_aux_metrics(sample, generation.trace))
        return result

    def aggregate(self, results: list[ScoreResult]) -> dict[str, float]:
        summary = super().aggregate(results)
        summary["d1_blank_cell_accuracy"] = summary[f"{self.name}_score"]
        summary["puzzle_success_rate"] = sum(
            result.aux["puzzle_success_rate"] for result in results
        ) / len(results)
        summary.update(aggregate_answer_position_metrics(results))
        return summary


class Sudoku4ThinkingDataset(Sudoku4Dataset):
    """The same 100-row Sudoku4 set with the original reasoning prompt."""

    name = "sudoku4_thinking"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        sample_count: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__(
            samples=samples,
            sample_count=sample_count,
            seed=seed,
            enable_reasoning=True,
        )


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
