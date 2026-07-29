#!/usr/bin/env python3
"""Dispatch the benchmark through one isolated environment per model.

The model list comes from ``configs/experiments/full_matrix.yaml``. Each
selected model is delegated to ``venv_scripts/<model>.py run``. That script
creates its own venv when needed and executes the benchmark with that venv's
Python. This keeps incompatible torch/transformers versions isolated.

    python run_bench.py
    python run_bench.py -m illada
    python run_bench.py -m dreamreasoner
    python run_bench.py -m qwen3_4b
    python run_bench.py -m qwen3_8b
    python run_bench.py -m diffusiongemma
    python run_bench.py -m gemma
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"
DEFAULT_VENV_SCRIPTS_DIR = PROJECT_ROOT / "venv_scripts"


def matrix_model_names(path: str | Path) -> list[str]:
    """Read model config stems without requiring PyYAML before setup."""
    names: list[str] = []
    in_models = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "models:":
            in_models = True
            continue
        if in_models and stripped == "datasets:":
            break
        if in_models:
            match = re.match(r"-\s+config:\s+([^#\s]+)", stripped)
            if match:
                names.append(Path(match.group(1)).stem)
    if not names:
        raise ValueError(f"no models found in matrix: {path}")
    return names


def normalize_model_names(values: Sequence[str], available: Sequence[str]) -> list[str]:
    requested: list[str] = []
    for value in values:
        requested.extend(part.strip() for part in value.split(",") if part.strip())
    requested = list(dict.fromkeys(requested))
    unknown = set(requested).difference(available)
    if unknown:
        raise ValueError(
            f"unknown model(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(available)}"
        )
    return requested or list(available)


def dispatch_model_scripts(
    model_names: Sequence[str],
    *,
    action: str,
    scripts_dir: str | Path = DEFAULT_VENV_SCRIPTS_DIR,
    env_updates: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    scripts_dir = Path(scripts_dir).resolve()
    environment = os.environ.copy()
    environment.update(env_updates or {})

    for index, model_name in enumerate(model_names, start=1):
        script = scripts_dir / f"{model_name}.py"
        if not script.is_file():
            raise FileNotFoundError(f"model script not found: {script}")
        command = [sys.executable, str(script), action]
        print(f"[{index}/{len(model_names)}] {model_name}: {' '.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-m", "--model", action="extend", nargs="+", default=[],
        help="Model name(s); space-separate, repeat, or comma-separate (default: all matrix models)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="extend",
        nargs="+",
        default=[],
        help="Dataset name(s) to run, e.g. -d ruler hellobench (default: all matrix datasets)",
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="Experiment matrix YAML")
    parser.add_argument("--venv-scripts-dir", dest="scripts_dir", default=str(DEFAULT_VENV_SCRIPTS_DIR), help="Directory containing per-model Python environment scripts")
    parser.add_argument("--stage", choices=("generate", "score", "visualize", "all"), default="all")
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--demo", dest="data_source", action="store_const", const="demo", default="demo")
    data.add_argument("--real-data", dest="data_source", action="store_const", const="real")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "--hellobench-length",
        action="append",
        choices=("2k", "4k", "2000", "4000"),
        help="HelloBench output profile; repeat to include both (default: 2k and 4k)",
    )
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument(
        "-v", "--variant",
        help="Run one sampling variant for every selected model, e.g. fast",
    )
    variants.add_argument(
        "--variants",
        help="Comma-separated sampling variants, e.g. best,fast",
    )
    parser.add_argument("--output-root", default="output")
    compute = parser.add_mutually_exclusive_group()
    compute.add_argument("--measure-compute", dest="measure_compute", action="store_true")
    compute.add_argument("--no-measure-compute", dest="measure_compute", action="store_false")
    parser.set_defaults(measure_compute=False)
    metrics = parser.add_mutually_exclusive_group()
    metrics.add_argument("--require-all-metrics", dest="require_all_metrics", action="store_true")
    metrics.add_argument("--allow-missing-metrics", dest="require_all_metrics", action="store_false")
    parser.set_defaults(require_all_metrics=False)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--n-representative", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Print model script commands without running them")
    parser.add_argument("--list-models", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix_path = Path(args.matrix).resolve()
    available = matrix_model_names(matrix_path)
    if args.list_models:
        print("\n".join(available))
        return 0
    try:
        selected = normalize_model_names(args.model, available)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Matrix: {matrix_path}")
    print(f"Models: {', '.join(selected)}")
    if args.dataset:
        print(f"Datasets: {', '.join(args.dataset)}")
    env_updates = {
        "EXPERIMENT_CONFIG": str(matrix_path),
        "DATA_SOURCE": args.data_source,
        "STAGE": args.stage,
        "OUTPUT_ROOT": args.output_root,
        "MEASURE_COMPUTE": "1" if args.measure_compute else "0",
        "REQUIRE_ALL_METRICS": "1" if args.require_all_metrics else "0",
        "RESUME": "1" if args.resume else "0",
    }
    if args.dataset:
        env_updates["DATASETS"] = ",".join(
            dict.fromkeys(
                part.strip()
                for value in args.dataset
                for part in value.split(",")
                if part.strip()
            )
        )
    if args.stage in {"visualize", "all"}:
        env_updates["N_REPRESENTATIVE"] = str(args.n_representative)
    if args.n_samples is not None:
        env_updates["N_SAMPLES"] = str(args.n_samples)
    if args.hellobench_length:
        env_updates["HELLOBENCH_LENGTHS"] = ",".join(args.hellobench_length)
    selected_variants = args.variant or args.variants
    if selected_variants:
        env_updates["MATRIX_VARIANTS"] = selected_variants
    dispatch_model_scripts(
        selected,
        action="run",
        scripts_dir=args.scripts_dir,
        env_updates=env_updates,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
