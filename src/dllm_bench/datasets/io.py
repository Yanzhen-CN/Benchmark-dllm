"""Read normalized or common official-style JSON/JSONL sample files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .base import Sample


def _jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one JSON object at a time. A prepared dataset's JSONL (e.g.
    RULER's, spanning context windows up to 262144 tokens) can hold enough
    total text that reading the whole file into one string and then
    `.splitlines()`-ing it (a second full copy, held alongside the first for
    the rest of the loop) meaningfully raises peak memory during loading —
    observed contributing to an OOM-kill several jobs later in the same
    long-running `matrix` process (memory a prior job used isn't always
    handed back to the OS immediately even once Python's own references are
    gone). Reading line-by-line means only the current line's text is ever
    live, and each parsed record can be collected as soon as the caller is
    done with it instead of outliving the whole file's worth of raw text.
    """
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                yield json.loads(line)


def _records(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _jsonl_records(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("samples", data.get("data"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records")
    return data


def _reference(dataset_name: str, record: dict[str, Any]) -> Any:
    raw = record.get("reference", record.get("answer"))
    if dataset_name == "mbpp":
        from .mbpp import MbppSample
        return MbppSample(**raw) if isinstance(raw, dict) else MbppSample(
            test_list=record.get("test_list", []),
            test_setup_code=record.get("test_setup_code", ""),
        )
    if dataset_name == "structeval_t":
        from .structeval_t import StructEvalSchema
        return raw if isinstance(raw, StructEvalSchema) else StructEvalSchema(**raw)
    if dataset_name in {
        "sudoku9", "sudoku9_1shot", "sudoku9_thinking", "sudoku_trace"
    }:
        from .sudoku9 import SudokuReference
        return raw if isinstance(raw, SudokuReference) else SudokuReference(**raw)
    if dataset_name in {"sudoku4", "sudoku4_1shot", "sudoku4_thinking"}:
        from .sudoku4 import Sudoku4Reference
        return raw if isinstance(raw, Sudoku4Reference) else Sudoku4Reference(**raw)
    if dataset_name == "ruler":
        from .ruler import RulerReference
        return raw if isinstance(raw, RulerReference) else RulerReference(**raw)
    if dataset_name == "hellobench":
        from .hellobench import HelloBenchReference
        return raw if isinstance(raw, HelloBenchReference) else HelloBenchReference(**raw)
    return raw


def load_samples_file(path: str | Path, dataset_name: str, n: int | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for index, record in enumerate(_records(path)):
        prompt = record.get("prompt", record.get("question", record.get("text")))
        if prompt is None:
            raise ValueError(f"record {index} in {path} has no prompt/question/text")
        sample_id = str(record.get("sample_id", record.get("id", record.get("task_id", index))))
        samples.append(Sample(
            sample_id=sample_id,
            prompt=str(prompt),
            reference=_reference(dataset_name, record),
            meta=dict(record.get("meta", {})),
        ))
    return samples[:n] if n is not None else samples
