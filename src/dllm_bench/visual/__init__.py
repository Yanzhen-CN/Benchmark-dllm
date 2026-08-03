"""Model-aware visualization dispatch.

Shared reports are always rendered. A module named after the configured model
may expose optional hooks for model-specific artifacts, so adding a new model
does not require another branch in the CLI.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from ..datasets.base import Sample
from ..interfaces import GenerationResult, TraceStep
from .public.dataset_trace_report import render_dataset_trace_report
from .public.trace_report import render_sample_report
from .base import ModelVisualProfile


def _model_module(model_name: str) -> ModuleType:
    module_name = f"{__name__}.{model_name.replace('-', '_')}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise RuntimeError(
                f"model {model_name!r} must own "
                f"src/dllm_bench/visual/{model_name.replace('-', '_')}.py"
            ) from exc
        raise


def _profile(module: ModuleType) -> ModelVisualProfile:
    profile = getattr(module, "VISUAL_PROFILE", None)
    if not isinstance(profile, ModelVisualProfile):
        raise RuntimeError(f"{module.__name__} must define VISUAL_PROFILE")
    return profile


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
    module = _model_module(model_name)
    profile = _profile(module)
    written = {}
    if profile.shared_sample:
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
    hook = getattr(module, "render_sample_visualization", None)
    if hook:
        written.update(
            hook(
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
    module = _model_module(model_name)
    profile = _profile(module)
    written = {}
    if profile.shared_dataset:
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
    hook = getattr(module, "render_dataset_visualization", None)
    if hook:
        written.update(
            hook(
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
    module = _model_module(model_name)
    profile = _profile(module)
    hook = getattr(module, "render_model_comparison_visualization", None)
    if not profile.cross_variant or not hook:
        return {}
    return hook(
        dataset_name=dataset_name,
        records_by_variant=records_by_variant,
        out_dir=Path(out_dir),
        seed=seed,
        block_length=block_length,
        figures=figures,
    ) or {}
