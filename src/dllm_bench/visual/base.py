"""Declarative ownership for each model's visualization surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ModelVisualProfile:
    """Select the visualization layers maintained by one model module."""

    shared_sample: bool = True
    shared_dataset: bool = True
    cross_variant: bool = False


def public_comparison_renderer(
    model_name: str,
) -> Callable[..., dict[str, str]]:
    """Build a model-owned entry point backed by the public implementation."""

    def render_model_comparison_visualization(
        *,
        dataset_name: str,
        records_by_variant: dict[str, list[tuple[Any, Any]]],
        out_dir: Path,
        seed: int = 42,
        block_length: int | None = None,
        figures: set[str] | None = None,
    ) -> dict[str, str]:
        del seed, block_length
        from .public.trace_comparison import render_trace_comparison

        return render_trace_comparison(
            model_name=model_name,
            dataset_name=dataset_name,
            records_by_variant=records_by_variant,
            out_dir=out_dir,
            figures=figures,
        )

    return render_model_comparison_visualization
