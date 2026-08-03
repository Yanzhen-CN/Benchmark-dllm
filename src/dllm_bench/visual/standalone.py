"""Reusable CLI for a model module's fine-grained comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

from ..runner.output_layout import (
    model_comparison_visualization_output_dir,
    resolve_model_output_dir,
)
from ..runner.persistence import load_generation_result


def run_model_visual_cli(
    model_name: str,
    renderer: Callable,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=f"Render fine-grained {model_name} comparison figures."
    )
    parser.add_argument("-d", "--dataset", action="extend", nargs="+", required=True)
    parser.add_argument("-v", "--variant", action="extend", nargs="+", required=True)
    parser.add_argument(
        "--figure",
        action="extend",
        nargs="+",
        choices=("all", "trace", "state", "convergence", "yield", "forward"),
        default=None,
    )
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--block-length", type=int, default=None)
    args = parser.parse_args(argv)

    variants = list(dict.fromkeys(args.variant))
    figures = set(args.figure or ["all"])
    for dataset_name in dict.fromkeys(args.dataset):
        records_by_variant = {}
        for variant in variants:
            source = resolve_model_output_dir(
                args.output_root,
                model_name,
                variant,
                dataset_name,
            )
            records = []
            for path in sorted(source.glob("*.json")):
                if path.name.startswith("_") or path.name == "oom_info.json":
                    continue
                generation = load_generation_result(path)
                sample = SimpleNamespace(sample_id=generation.request.sample_id)
                records.append((sample, generation))
            if records:
                records_by_variant[variant] = records

        output = model_comparison_visualization_output_dir(
            args.output_root,
            model_name,
            dataset_name,
        )
        written = renderer(
            dataset_name=dataset_name,
            records_by_variant=records_by_variant,
            out_dir=Path(output),
            block_length=args.block_length,
            figures=figures,
        )
        for name, path in written.items():
            print(f"[{dataset_name}] {name}: {path}")
    return 0
