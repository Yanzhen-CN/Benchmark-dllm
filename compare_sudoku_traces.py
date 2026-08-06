#!/usr/bin/env python3
"""Render matched LLaDA2.1 versus DiffusionGemma Sudoku trace artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from venv_scripts.root import run_in_root_venv


DEFAULT_MODELS = ("llada2_1:qmode", "diffusiongemma:official")


def parse_model_spec(value: str) -> tuple[str, str]:
    model, separator, config = value.partition(":")
    if not separator or not model.strip() or not config.strip():
        raise argparse.ArgumentTypeError("model must use MODEL:CONFIG, e.g. llada2_1:qmode")
    return model.strip(), config.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        default=None,
        help="MODEL:CONFIG pair; repeat for each comparison row",
    )
    parser.add_argument("-d", "--dataset", default="sudoku4_1shot")
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Matched sample ID; default selects the common median-length trace",
    )
    parser.add_argument("--output-root", default="output")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(value in arguments for value in ("-h", "--help")):
        run_in_root_venv(__file__, arguments)

    from dllm_bench.registry import build_dataset
    from dllm_bench.runner.output_layout import (
        resolve_model_output_dir,
        resolve_score_output_dir,
        visualization_output_dir,
    )
    from dllm_bench.runner.persistence import (
        load_generation_result,
        load_score_result,
    )
    from dllm_bench.visual import render_sample_visualization
    from dllm_bench.visual.public.trace_comparison import (
        render_trace_comparison,
        select_common_sample,
    )

    args = build_parser().parse_args(arguments)
    model_specs = args.model or [parse_model_spec(value) for value in DEFAULT_MODELS]
    if len(model_specs) < 2:
        raise SystemExit("at least two --model MODEL:CONFIG pairs are required")

    repository_root = Path(__file__).resolve().parent
    dataset_config = repository_root / "configs" / "datasets" / f"{args.dataset}.yaml"
    if not dataset_config.is_file():
        raise SystemExit(f"dataset config not found: {dataset_config}")
    dataset = build_dataset(str(dataset_config))
    samples = {sample.sample_id: sample for sample in dataset.load_samples()}
    output_root = Path(args.output_root)
    generation_root = output_root / "model_output"

    records_by_label = {}
    model_records = {}
    for model_name, config_name in model_specs:
        source = resolve_model_output_dir(
            generation_root, model_name, config_name, args.dataset
        )
        records = []
        for path in sorted(source.glob("*.json")):
            if path.name.startswith("_") or path.name == "oom_info.json":
                continue
            generation = load_generation_result(path)
            sample = samples.get(generation.request.sample_id)
            if sample is None or not generation.trace:
                continue
            records.append((sample, generation))
        if not records:
            raise SystemExit(f"no traced samples found under {source}")
        label = f"{model_name}/{config_name}"
        records_by_label[label] = records
        model_records[label] = (model_name, config_name)

    if args.sample_id:
        selected = {}
        for label, records in records_by_label.items():
            match = next(
                (record for record in records if record[0].sample_id == args.sample_id),
                None,
            )
            if match is None:
                raise SystemExit(f"sample {args.sample_id} is missing from {label}")
            selected[label] = match
        sample_id = args.sample_id
    else:
        common = select_common_sample(records_by_label)
        if common is None:
            raise SystemExit("the selected model rows have no common traced sample")
        sample_id, selected = common

    comparison_records = {}
    for label, (sample, generation) in selected.items():
        model_name, config_name = model_records[label]
        score_dir = resolve_score_output_dir(
            generation_root, model_name, config_name, args.dataset
        )
        score_path = score_dir / f"{sample_id}.json"
        final_score = (
            load_score_result(score_path).primary_score if score_path.exists() else None
        )
        visual_dir = visualization_output_dir(
            generation_root, model_name, config_name, args.dataset
        )
        written = render_sample_visualization(
            model_name=model_name,
            sample_id=sample_id,
            trace=generation.trace,
            final_valid_length=generation.final_valid_length,
            out_dir=visual_dir,
            final_output_text=generation.output_text,
            final_score=final_score,
            dataset_name=args.dataset,
            sample=sample,
        )
        for name, path in written.items():
            print(f"[{label}] {name}: {path}")
        comparison_records[label] = [(sample, generation)]

    comparison_dir = (
        output_root / "visualization_output" / "sudoku_model_comparison" / args.dataset
    )
    written = render_trace_comparison(
        model_name=" vs ".join(records_by_label),
        dataset_name=args.dataset,
        records_by_variant=comparison_records,
        out_dir=comparison_dir,
        figures={"all"},
    )
    for name, path in written.items():
        print(f"[comparison] {name}: {path}")
    print(f"Matched sample: {sample_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
