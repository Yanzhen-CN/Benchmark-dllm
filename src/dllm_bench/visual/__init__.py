"""Model-aware visualization dispatch over public and private layers."""

from __future__ import annotations

from pathlib import Path

from ..datasets.base import Sample
from ..interfaces import GenerationResult, TraceStep
from .models import load_model_visual
from .public.dataset_trace_report import render_dataset_trace_report
from .public.profiling_report import render_dataset_profiling_report
from .public.trace_report import render_sample_report


def render_sample_visualization(
    *,
    model_name: str,
    sample_id: str,
    trace: list[TraceStep],
    final_valid_length: int,
    out_dir: str | Path,
    final_output_text: str = "",
    final_score: float | None = None,
    dataset_name: str | None = None,
    sample: Sample | None = None,
    block_length: int | None = None,
) -> dict[str, str]:
    visual = load_model_visual(model_name)
    written: dict[str, str] = {}
    if visual.public_sample:
        written.update(
            render_sample_report(
                sample_id=sample_id,
                trace=trace,
                final_valid_length=final_valid_length,
                out_dir=str(out_dir),
                final_output_text=final_output_text,
                final_score=final_score,
                dataset_name=dataset_name,
                sample=sample,
                block_length=block_length,
            )
        )
    if visual.render_sample:
        written.update(
            visual.render_sample(
                sample_id=sample_id,
                trace=trace,
                final_valid_length=final_valid_length,
                out_dir=Path(out_dir),
                dataset_name=dataset_name,
                sample=sample,
                block_length=block_length,
            )
            or {}
        )
    return written


def render_dataset_visualization(
    *,
    model_name: str,
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    out_dir: str | Path,
    seed: int = 42,
    config_name: str | None = None,
    block_length: int | None = None,
) -> dict[str, str]:
    visual = load_model_visual(model_name)
    written: dict[str, str] = {}
    if visual.public_dataset:
        if any(result.trace for _, result in records):
            written.update(
                render_dataset_trace_report(
                    dataset_name,
                    records,
                    out_dir,
                    seed=seed,
                    model_name=model_name,
                    config_name=config_name,
                )
            )
        written.update(
            render_dataset_profiling_report(
                dataset_name,
                records,
                out_dir,
                model_name=model_name,
                config_name=config_name,
            )
        )
    if visual.render_dataset:
        written.update(
            visual.render_dataset(
                dataset_name=dataset_name,
                records=records,
                out_dir=Path(out_dir),
                seed=seed,
                config_name=config_name,
                block_length=block_length,
            )
            or {}
        )
    return written


def render_model_comparison_visualization(
    *,
    model_name: str,
    dataset_name: str,
    records_by_variant: dict[str, list[tuple[Sample, GenerationResult]]],
    out_dir: str | Path,
    seed: int = 42,
    block_length: int | None = None,
    figures: set[str] | None = None,
) -> dict[str, str]:
    visual = load_model_visual(model_name)
    if visual.render_comparison is None:
        return {}
    return visual.render_comparison(
        dataset_name=dataset_name,
        records_by_variant=records_by_variant,
        out_dir=Path(out_dir),
        seed=seed,
        block_length=block_length,
        figures=figures,
    ) or {}
