"""Reusable model-agnostic visualization and reporting components."""

from .style import install_public_style

install_public_style()

from .dataset_trace_report import render_dataset_trace_report
from .paper_assets import render_paper_assets
from .raw_report import write_raw_report
from .trace_comparison import render_trace_comparison
from .trace_metrics import (
    build_auxiliary_performance_rows,
    build_auxiliary_performance_summary,
    build_trace_step_rows,
    summarize_profiling,
)
from .trace_report import render_sample_report

__all__ = [
    "render_dataset_trace_report",
    "render_paper_assets",
    "build_auxiliary_performance_rows",
    "build_auxiliary_performance_summary",
    "render_sample_report",
    "render_trace_comparison",
    "build_trace_step_rows",
    "summarize_profiling",
    "write_raw_report",
]
