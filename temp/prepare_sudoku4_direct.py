#!/usr/bin/env python3
"""Prepare an isolated Sudoku4 set with a minimal direct-answer prompt."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path(__file__).resolve().parent
DATA_PATH = TEMP_ROOT / "data" / "sudoku4_direct_short.jsonl"
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from dllm_bench.datasets.sudoku4 import Sudoku4Dataset

    samples = Sudoku4Dataset(sample_count=100, seed=42).load_samples()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            puzzle = str(sample.reference.puzzle)
            record = {
                "sample_id": sample.sample_id,
                "prompt": (
                    f"Solve this Sudoku: {puzzle}\n"
                    "Directly return your answer with only 16 digits."
                ),
                "reference": dataclasses.asdict(sample.reference),
                "meta": {
                    **sample.meta,
                    "prompt_protocol": "temporary-minimal-direct-return-v7",
                },
            }
            output.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            output.write("\n")
    print(f"Prepared {len(samples)} samples -> {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
