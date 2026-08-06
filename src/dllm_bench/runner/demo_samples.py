"""Tiny hand-authored sample sets, one per dataset, so the CLI's ``--demo``
flag can run the full generate -> score -> measure -> report pipeline
end-to-end without needing real dataset files or network access — this is
what the mock-adapter smoke test in the project README exercises.

Not a substitute for the real section-1 datasets: swap these for the actual
GSM8K/MBPP/... loaders before using this harness for real numbers.
"""

from __future__ import annotations

from ..datasets.base import Sample
from ..datasets.hellobench import HelloBenchReference
from ..datasets.mbpp import MbppSample
from ..datasets.ruler import build_niah_sample
from ..datasets.structeval_t import StructEvalSchema
from ..datasets.sudoku9 import SudokuReference
from ..datasets.sudoku4 import Sudoku4Reference, format_sudoku4_prompt

_EASY_SUDOKU_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def _sudoku_puzzle_text() -> str:
    import copy

    puzzle = copy.deepcopy(_EASY_SUDOKU_SOLUTION)
    for r, c in [(0, 0), (1, 1), (2, 2)]:
        puzzle[r][c] = 0
    return "\n".join(" ".join(str(v) for v in row) for row in puzzle), puzzle


def build_demo_samples(dataset_name: str, n: int = 5) -> list[Sample]:
    builder = _BUILDERS.get(dataset_name)
    if builder is None:
        raise ValueError(f"no demo sample builder for dataset {dataset_name!r}")
    return builder(n)


def _gsm8k_samples(n: int) -> list[Sample]:
    samples = []
    for i in range(n):
        a, b = i + 2, i + 3
        samples.append(
            Sample(
                sample_id=f"gsm8k-demo-{i}",
                prompt=f"If a store has {a} apples and receives {b} more, how many apples does it have in total?",
                reference=float(a + b),
            )
        )
    return samples


def _mbpp_samples(n: int) -> list[Sample]:
    samples = []
    for i in range(n):
        samples.append(
            Sample(
                sample_id=f"mbpp-demo-{i}",
                prompt=f"Write a Python function `add{i}(a, b)` that returns the sum of a and b.",
                reference=MbppSample(test_list=[f"assert add{i}(2, 3) == 5", f"assert add{i}(-1, 1) == 0"]),
            )
        )
    return samples


def _structeval_t_samples(n: int) -> list[Sample]:
    samples = []
    for i in range(n):
        samples.append(
            Sample(
                sample_id=f"structeval-demo-{i}",
                prompt=f'Return a JSON object with keys "name" and "age" for person {i}.',
                reference=StructEvalSchema(format="json", required_keys=["name", "age"]),
            )
        )
    return samples


def _sudoku9_samples(n: int) -> list[Sample]:
    text, puzzle = _sudoku_puzzle_text()
    samples = []
    for i in range(n):
        samples.append(
            Sample(
                sample_id=f"sudoku9-demo-{i}",
                prompt=(
                    f"Solve this Sudoku puzzle (0 = blank):\n{text}\n\n"
                    "Directly return the 81 numbers answer. Return ONLY the "
                    "completed grid as one row-major 81-digit string. Your "
                    "entire response must contain exactly 81 digits "
                    "(1-9), with no spaces, labels, explanation, reasoning, or "
                    "other text.\nAnswer (81 digits only):"
                ),
                reference=SudokuReference(puzzle=puzzle, solution=_EASY_SUDOKU_SOLUTION),
            )
        )
    return samples


def _sudoku4_samples(n: int) -> list[Sample]:
    puzzle = "3102200002100320"
    solution = "3142243142131324"
    return [
        Sample(
            sample_id=f"sudoku4-demo-{index}",
            prompt=format_sudoku4_prompt(puzzle),
            reference=Sudoku4Reference(puzzle, solution),
        )
        for index in range(n)
    ]


def _sudoku_trace_samples(n: int) -> list[Sample]:
    from ..datasets.sudoku9 import format_sudoku_trace_prompt

    samples = _sudoku9_samples(n)
    return [
        Sample(
            sample_id=sample.sample_id.replace("sudoku9-demo", "sudoku-trace-demo"),
            prompt=format_sudoku_trace_prompt(
                "".join(str(value) for row in sample.reference.puzzle for value in row)
            ),
            reference=sample.reference,
            meta={**sample.meta, "max_new_tokens": 128},
        )
        for sample in samples
    ]


def _ruler_samples(n: int) -> list[Sample]:
    positions = ["front", "middle", "back"]
    return [
        build_niah_sample(
            sample_id=f"ruler-demo-{i}",
            needle_value=str(1000 + i),
            position=positions[i % len(positions)],
            num_filler_sentences=20,
            seed=42 + i,
        )
        for i in range(n)
    ]


def _hellobench_samples(n: int) -> list[Sample]:
    samples = []
    for i in range(n):
        samples.append(
            Sample(
                sample_id=f"hellobench-demo-{i}",
                prompt=f"Write a short story about topic #{i}.",
                reference=HelloBenchReference(target_length_words=50),
                meta={"max_new_tokens": 128},
            )
        )
    return samples


_BUILDERS = {
    "gsm8k": _gsm8k_samples,
    "mbpp": _mbpp_samples,
    "structeval_t": _structeval_t_samples,
    "sudoku4": _sudoku4_samples,
    "sudoku9": _sudoku9_samples,
    "sudoku_trace": _sudoku_trace_samples,
    "ruler": _ruler_samples,
    "hellobench": _hellobench_samples,
}
