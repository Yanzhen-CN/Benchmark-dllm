"""Contracts shared by public and model-owned visualization modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

VisualRenderer = Callable[..., dict[str, str]]


@dataclass(frozen=True)
class ModelVisual:
    """Declare the public suite and optional private renderers for one model."""

    model_name: str
    public_sample: bool = True
    public_dataset: bool = True
    render_sample: VisualRenderer | None = None
    render_dataset: VisualRenderer | None = None
    render_comparison: VisualRenderer | None = None

    def main(self, argv: Sequence[str] | None = None) -> int:
        """Run this model's fine-grained comparison CLI."""
        if self.render_comparison is None:
            raise RuntimeError(f"{self.model_name} has no comparison renderer")
        from .standalone import run_model_visual_cli

        return run_model_visual_cli(
            self.model_name,
            self.render_comparison,
            argv,
        )


def public_comparison_renderer(model_name: str) -> VisualRenderer:
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


def public_model_visual(model_name: str) -> ModelVisual:
    """Return the default model declaration using every public visual layer."""
    return ModelVisual(
        model_name=model_name,
        render_comparison=public_comparison_renderer(model_name),
    )
