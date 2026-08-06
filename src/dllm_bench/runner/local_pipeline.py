"""Shared local dispatcher for score and visualization matrix stages."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .matrix import available_matrix_models


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"


def normalize_model_names(values: Sequence[str], available: Sequence[str]) -> list[str]:
    requested = list(
        dict.fromkeys(
            part.strip()
            for value in values
            for part in value.split(",")
            if part.strip()
        )
    )
    unknown = set(requested).difference(available)
    if unknown:
        raise ValueError(
            f"unknown model(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(available)}"
        )
    return requested or list(available)


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
        "-suffix", "--output-suffix", default=None,
        help="Append a suffix to the final dataset output directory",
    )
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
    parser.add_argument(
        "--output-root",
        default="output/model_output",
        help=(
            "Exact generation-output root; model run directories are read "
            "directly below this path"
        ),
    )
    if stage == "score":
        resume = parser.add_mutually_exclusive_group()
        resume.add_argument("--resume", dest="resume", action="store_true")
        resume.add_argument("--no-resume", dest="resume", action="store_false")
        parser.set_defaults(resume=True)
        parser.add_argument(
            "--preview",
            action="store_true",
            help=(
                "Recompute scores in memory and print them to the terminal; "
                "do not create or update score_output files"
            ),
        )
    if stage == "visualize":
        parser.add_argument(
            "--preset",
            choices=(
                "report-assets",
                "profiling-comparison",
                "sudoku-trace-batch",
                "platform-chart",
            ),
            default=None,
            help="Generate one curated aggregate artifact set without model loading.",
        )
        parser.add_argument(
            "--compare-run",
            action="append",
            default=[],
            metavar="MODEL/VARIANT",
            help="Repeat for every model/variant in a Sudoku trace comparison.",
        )
        parser.add_argument(
            "--chart-spec",
            default=None,
            help="JSON chart specification used by the platform-chart preset.",
        )
        parser.add_argument(
            "--scope",
            choices=("all", "sample", "dataset", "comparison"),
            default="all",
            help="Limit rendering to the requested visualization layer.",
        )
        parser.add_argument(
            "--figure",
            action="extend",
            nargs="+",
            default=[],
            choices=("all", "trace", "state", "convergence", "yield", "forward"),
            help=(
                "Visualization subset. Public choices work for every model; "
                "'forward' is currently DiffusionGemma-specific."
            ),
        )
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
        report = parser.add_mutually_exclusive_group()
        report.add_argument("--report", dest="report", action="store_true")
        report.add_argument("--no-report", dest="report", action="store_false")
        parser.set_defaults(report=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(stage: str, argv: Sequence[str] | None = None) -> int:
    if stage not in {"score", "visualize"}:
        raise ValueError(f"unsupported local stage: {stage}")
    args = build_parser(stage).parse_args(argv)
    matrix_path = Path(args.matrix).resolve()
    try:
        models = normalize_model_names(args.model, available_matrix_models(matrix_path))
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
    selected_figures = ",".join(dict.fromkeys(getattr(args, "figure", [])))

    if stage == "visualize" and args.preset == "sudoku-trace-batch":
        if not args.compare_run:
            raise SystemExit(
                "sudoku-trace-batch requires at least one --compare-run value"
            )
        if not args.sample_ids or "," in args.sample_ids:
            raise SystemExit(
                "sudoku-trace-batch requires one --sample-ids value"
            )
        selected_datasets = [
            part.strip()
            for value in args.dataset
            for part in value.split(",")
            if part.strip()
        ]
        if len(selected_datasets) != 1:
            raise SystemExit(
                "sudoku-trace-batch requires exactly one dataset"
            )

        runs: list[tuple[str, str]] = []
        for value in args.compare_run:
            separator = "/" if "/" in value else ":"
            model, found, variant = value.partition(separator)
            if not found or not model.strip() or not variant.strip():
                raise SystemExit(
                    f"invalid --compare-run {value!r}; use MODEL/VARIANT"
                )
            if model.strip() not in models:
                raise SystemExit(f"comparison model {model.strip()!r} was not selected")
            runs.append((model.strip(), variant.strip()))

        dataset_config = selected_datasets[0]
        for index, (model, variant) in enumerate(runs, start=1):
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
                "visualize",
                "--no-demo" if args.real_data else "--demo",
                "--output-root",
                args.output_root,
                "--visualization-scope",
                "sample",
                "--sample-ids",
                args.sample_ids,
                "--variants",
                variant,
                "--dataset",
                dataset_config,
            ]
            if args.output_suffix:
                command.extend(["--output-suffix", args.output_suffix])
            print(
                f"[{index}/{len(runs)}] {model}/{variant}: {' '.join(command)}",
                flush=True,
            )
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return 0

    if stage == "visualize" and args.preset:
        if args.preset == "platform-chart":
            if not args.chart_spec:
                raise SystemExit("platform-chart requires --chart-spec")
            from ..visual.public.platform_chart import render_platform_chart

            print(render_platform_chart(args.chart_spec))
            return 0
        from ..visual.public.targeted import (
            render_profiling_comparison_from_output,
            render_report_assets_from_output,
        )

        benchmark_output = Path(args.output_root).resolve().parent
        selected_datasets = [
            part.strip()
            for value in args.dataset
            for part in value.split(",")
            if part.strip()
        ]
        if args.preset == "report-assets":
            written = render_report_assets_from_output(
                benchmark_output,
                model_names=models,
                dataset_names=selected_datasets,
            )
        else:
            written = render_profiling_comparison_from_output(
                benchmark_output,
                model_names=models,
                dataset_names=selected_datasets,
            )
        for path in written:
            print(path)
        if not written:
            raise SystemExit(
                f"no files were generated for preset {args.preset}; "
                "check the selected models, datasets, and source artifacts"
            )
        return 0

    print(f"Matrix: {matrix_path}")
    print(f"Local stage: {stage}")
    if stage == "score" and args.preview:
        print("Score mode: PREVIEW ONLY (no score files will be written)")
    print(f"Models: {', '.join(models)}")
    child_environment = os.environ.copy()
    if stage == "score" and args.preview:
        child_environment["DLLM_SCORE_PREVIEW"] = "1"
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
        if args.output_suffix:
            command.extend(["--output-suffix", args.output_suffix])
        if stage == "visualize":
            command.extend(["--n-representative", str(args.n_representative)])
            command.extend(["--visualization-scope", args.scope])
            if args.sample_ids:
                command.extend(["--sample-ids", args.sample_ids])
            if selected_figures:
                command.extend(["--figures", selected_figures])
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
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=child_environment,
                check=True,
            )

    if stage == "visualize" and args.report:
        generation_roots = (
            [Path(args.output_root) / f"len{length}" for length in max_new_tokens]
            if len(max_new_tokens) > 1
            else [Path(args.output_root)]
        )
        report_roots = [str(root.parent) for root in generation_roots]
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
                        dataset_name = dataset_name.strip()
                        if args.output_suffix:
                            dataset_name = f"{dataset_name}_{args.output_suffix}"
                        report_command.extend(["--dataset", dataset_name])
            print(f"Report: {' '.join(report_command)}", flush=True)
            if not args.dry_run:
                subprocess.run(report_command, cwd=PROJECT_ROOT, check=True)
    return 0
