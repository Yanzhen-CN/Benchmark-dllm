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
                    f"Solve this 4x4 Sudoku: {puzzle}\n"
                    "Directly return only the final 16-digit answer using digits 1-4."
                ),
                "reference": dataclasses.asdict(sample.reference),
                "meta": {
                    **sample.meta,
                    "prompt_protocol": "temporary-minimal-direct-16-digits-v3",
                },
            }
            output.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            output.write("\n")
    print(f"Prepared {len(samples)} samples -> {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
