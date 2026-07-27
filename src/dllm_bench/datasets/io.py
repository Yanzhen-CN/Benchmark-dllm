"""Read normalized or common official-style JSON/JSONL sample files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Sample


def _records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
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
    if dataset_name == "sudoku":
        from .sudoku import SudokuReference
        return raw if isinstance(raw, SudokuReference) else SudokuReference(**raw)
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
