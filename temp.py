#!/usr/bin/env python3
"""Temporary one-off: strip the (potentially huge) "trace" field out of
already-saved HelloBench generation results under output/model_output/.

HelloBench outputs are 2K-4K words, so a diffusion model's per-step trace
for those samples can run into many MB per file. Scoring only ever reads
`output_text` (see datasets/hellobench.py), so the trace isn't needed there
— this only touches on-disk model_output/*/hellobench/*.json files directly
(plain JSON dict edit, no dllm_bench import needed), it does not re-run
generation or scoring.

Reports what it would do, then asks for y/n confirmation before touching
anything. Pass -y/--yes to skip the prompt (e.g. from another script).

Usage:
    python temp.py                # report, then ask y/n
    python temp.py -y             # no prompt, just do it
    python temp.py --root output  # different --output-root
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def hellobench_generation_files(output_root: Path):
    model_output_root = output_root / "model_output"
    if not model_output_root.is_dir():
        raise SystemExit(f"no such directory: {model_output_root}")
    for path in sorted(model_output_root.glob("*/hellobench/*.json")):
        if path.name != "_meta.json":
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="output", help="Output root (default: output)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the y/n prompt and apply immediately")
    args = parser.parse_args()

    output_root = Path(args.root)
    to_clear: list[tuple[Path, dict, int, int]] = []  # (path, data, before_size, after_size)
    total_files = 0
    total_before = 0
    total_after = 0

    for path in hellobench_generation_files(output_root):
        total_files += 1
        before_size = path.stat().st_size
        data = json.loads(path.read_text(encoding="utf-8"))

        if not data.get("trace"):
            total_before += before_size
            total_after += before_size
            continue

        data["trace"] = []
        after_size = len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        to_clear.append((path, data, before_size, after_size))
        total_before += before_size
        total_after += after_size

    print(f"Would clear trace from {len(to_clear)}/{total_files} hellobench generation file(s) under {output_root}/model_output")
    print(f"Size: {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")

    if not to_clear:
        return 0

    if not args.yes:
        answer = input("Clear these traces now? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted — no files changed.")
            return 0

    for path, data, _before, _after in to_clear:
        path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    print(f"Cleared {len(to_clear)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
