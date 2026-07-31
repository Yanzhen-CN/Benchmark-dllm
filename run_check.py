#!/usr/bin/env python3
"""Check output completeness and consistency against an experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from venv_scripts.root import run_in_root_venv


def build_parser() -> argparse.ArgumentParser:
    from run_bench import DEFAULT_MATRIX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", action="extend", nargs="+", default=[])
    parser.add_argument(
        "-d",
        "--dataset",
        action="extend",
        nargs="+",
        default=[],
        help="Dataset name(s); 'sudoku' selects every sudoku* matrix dataset",
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-root", default="output")
    parser.add_argument(
        "--stage",
        choices=("generate", "score", "visualize", "all"),
        default="generate",
        help="Artifact stage to require (default: generated model output)",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("-max", "--max-new-tokens", type=int, default=None)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument(
        "-v", "--variant", action="extend", nargs="+", default=[]
    )
    variants.add_argument("--variants", help="Comma-separated variants")
    parser.add_argument(
        "--require-diagnostics",
        action="store_true",
        help="Require capacity_diagnostic datasets such as ruler_context_probe",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report",
    )
    return parser


def _split(values: Sequence[str]) -> list[str]:
    return list(
        dict.fromkeys(
            part.strip()
            for value in values
            for part in value.split(",")
            if part.strip()
        )
    )


def _format_status_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def _run(argv: Sequence[str] | None = None) -> int:
    from dllm_bench.registry import list_model_variants, load_yaml
    from dllm_bench.runner.check_stage import check_job_variant, serialise_checks
    from dllm_bench.runner.matrix import load_matrix_jobs
    from run_bench import matrix_model_names, normalize_model_names

    args = build_parser().parse_args(argv)
    matrix_path = Path(args.matrix).resolve()
    models = normalize_model_names(args.model, matrix_model_names(matrix_path))
    datasets = _split(args.dataset)
    jobs, seed = load_matrix_jobs(
        matrix_path,
        model_names=models,
        dataset_names=datasets,
    )
    selected_variants = _split(
        args.variant if args.variant else [args.variants or ""]
    )
    rows = []
    for job in jobs:
        available_variants = list_model_variants(job.model_config)
        if selected_variants:
            unknown = set(selected_variants).difference(available_variants)
            if unknown:
                raise SystemExit(
                    f"selected variant(s) are not available for {job.model_name}: "
                    f"{', '.join(sorted(unknown))}; available: "
                    f"{', '.join(available_variants)}"
                )
            variants = selected_variants
        else:
            variants = list(job.variants)
        dataset_config = load_yaml(job.dataset_config)
        diagnostic = dataset_config.get("protocol_type") == "capacity_diagnostic"
        optional = diagnostic and not args.require_diagnostics and not datasets
        for variant in variants:
            rows.append(
                check_job_variant(
                    job,
                    variant,
                    output_root=args.output_root,
                    seed=seed,
                    stage=args.stage,
                    n_samples_override=args.n_samples,
                    max_new_tokens_override=args.max_new_tokens,
                    optional=optional,
                )
            )

    required_failures = [row for row in rows if row.errors and not row.optional]
    if args.json:
        print(
            json.dumps(
                {
                    "matrix": str(matrix_path),
                    "stage": args.stage,
                    "ok": not required_failures,
                    "rows": serialise_checks(rows),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"Matrix: {matrix_path}")
        print(f"Output: {Path(args.output_root).resolve()}")
        print(f"Stage: {args.stage}")
        print(f"Models: {', '.join(models)}")
        for row in rows:
            label = "OPTIONAL" if row.optional else ("OK" if row.ok else "FAIL")
            expected = row.expected_samples if row.expected_samples is not None else "?"
            suffix = (
                f" samples={row.actual_samples}/{expected}"
                f" status={_format_status_counts(row.status_counts)}"
                f" trace={row.trace_samples}"
            )
            if row.score_complete is not None:
                suffix += f" score={'yes' if row.score_complete else 'no'}"
            if row.visualization_present is not None:
                suffix += (
                    " visualization=yes" if row.visualization_present
                    else " visualization=no"
                )
            print(
                f"[{label}] {row.model}/{row.variant}/{row.dataset}{suffix}"
            )
            for message in row.errors:
                print(f"  error: {message}")
            for message in row.warnings:
                print(f"  warning: {message}")
        required_rows = [row for row in rows if not row.optional]
        optional_rows = len(rows) - len(required_rows)
        print(
            f"Summary: {len(required_rows) - len(required_failures)}/"
            f"{len(required_rows)} required rows pass; "
            f"optional rows={optional_rows}"
        )
    return 1 if required_failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if argv is None and not any(value in arguments for value in ("-h", "--help")):
        run_in_root_venv(__file__, arguments)
    return _run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
