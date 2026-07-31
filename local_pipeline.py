"""Shared local dispatcher for score and visualization matrix stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from run_bench import (
    DEFAULT_MATRIX,
    PROJECT_ROOT,
    matrix_model_names,
    normalize_model_names,
)


def build_parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run the {stage} stage locally under .venvs/root. "
            "Model adapters and weights are never loaded."
        )
    )
    parser.add_argument("-m", "--model", action="extend", nargs="+", default=[])
    parser.add_argument(
        "-d", "--dataset", action="extend", nargs="+", default=[],
        help=(
            "Dataset name(s); 'sudoku' selects every sudoku* variant in the "
            "matrix"
        ),
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--demo", dest="real_data", action="store_false")
    data.add_argument("--real-data", dest="real_data", action="store_true")
    parser.set_defaults(real_data=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "-max",
        "--max-new-tokens",
        action="extend",
        nargs="+",
        type=int,
        default=[],
        help=(
            "One or more generation lengths. Multiple values read matching "
            "<output-root>/len<value>/ trees."
        ),
    )
    parser.add_argument(
        "-v",
        "--variant",
        action="extend",
        nargs="+",
        default=[],
        help="Variant subset; space-separate or repeat, e.g. -v p1 p2 p4 p8",
    )
    parser.add_argument("--output-root", default="output")
    if stage == "score":
        resume = parser.add_mutually_exclusive_group()
        resume.add_argument("--resume", dest="resume", action="store_true")
        resume.add_argument("--no-resume", dest="resume", action="store_false")
        parser.set_defaults(resume=True)
    if stage == "visualize":
        parser.add_argument(
            "--n-representative",
            type=int,
            default=0,
            help=(
                "Automatically render this many single-sample traces "
                "(default: 0; prefer --sample-ids for curated examples)"
            ),
        )
        parser.add_argument(
            "--sample-ids",
            default=None,
            help=(
                "Comma-separated curated sample IDs. Matching IDs are rendered "
                "per dataset; dataset-level Task 4 summaries still use all traces."
            ),
        )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(stage: str, argv: Sequence[str] | None = None) -> int:
    if stage not in {"score", "visualize"}:
        raise ValueError(f"unsupported local stage: {stage}")
    args = build_parser(stage).parse_args(argv)
    matrix_path = Path(args.matrix).resolve()
    try:
        models = normalize_model_names(args.model, matrix_model_names(matrix_path))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    max_new_tokens = list(dict.fromkeys(args.max_new_tokens))
    if any(value <= 0 for value in max_new_tokens):
        raise SystemExit("--max-new-tokens must be greater than zero")
    selected_variants = ",".join(
        dict.fromkeys(
            part.strip()
            for value in args.variant
            for part in value.split(",")
            if part.strip()
        )
    )

    print(f"Matrix: {matrix_path}")
    print(f"Local stage: {stage}")
    print(f"Models: {', '.join(models)}")
    for index, model in enumerate(models, start=1):
        command = [
            sys.executable,
            "-m",
            "dllm_bench.cli",
            "matrix",
            "--experiment-config",
            str(matrix_path),
            "--model",
            model,
            "--stage",
            stage,
            "--no-demo" if args.real_data else "--demo",
            "--output-root",
            args.output_root,
        ]
        if stage == "visualize":
            command.extend(["--n-representative", str(args.n_representative)])
            if args.sample_ids:
                command.extend(["--sample-ids", args.sample_ids])
        if stage == "score":
            command.append("--resume" if args.resume else "--no-resume")
        if args.n_samples is not None:
            command.extend(["--n-samples", str(args.n_samples)])
        for length in max_new_tokens:
            command.extend(["--max-new-tokens", str(length)])
        if selected_variants:
            command.extend(["--variants", selected_variants])
        for value in args.dataset:
            for dataset_name in value.split(","):
                if dataset_name.strip():
                    command.extend(["--dataset", dataset_name.strip()])
        print(f"[{index}/{len(models)}] {model}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if stage == "visualize":
        report_roots = (
            [str(Path(args.output_root) / f"len{length}") for length in max_new_tokens]
            if len(max_new_tokens) > 1
            else [args.output_root]
        )
        for report_root in report_roots:
            report_command = [
                sys.executable, "-m", "dllm_bench.cli", "report",
                "--output-root", report_root,
            ]
            for model in models:
                report_command.extend(["--model", model])
            for value in args.dataset:
                for dataset_name in value.split(","):
                    if dataset_name.strip():
                        report_command.extend(["--dataset", dataset_name.strip()])
            print(f"Report: {' '.join(report_command)}", flush=True)
            if not args.dry_run:
                subprocess.run(report_command, cwd=PROJECT_ROOT, check=True)
    return 0
