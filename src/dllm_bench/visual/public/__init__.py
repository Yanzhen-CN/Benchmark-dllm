"""Reusable model-agnostic visualization and reporting components."""

from .style import install_public_style

install_public_style()

from .dataset_trace_report import render_dataset_trace_report
from .paper_assets import render_paper_assets
from .profiling_comparison import (
    ProfilingComparisonSeries,
    build_profiling_comparison_series,
    plot_normalized_step_comparison,
    plot_profiling_totals_comparison,
    plot_stage_share_comparison,
    render_profiling_comparison_report,
    write_profiling_comparison_csv,
)
from .profiling_report import (
    build_dataset_profiling_summary,
    build_stage_profiling_summary,
    plot_stage_profiling,
    render_dataset_profiling_report,
)
from .raw_report import write_raw_report
from .targeted import (
    render_profiling_comparison_from_output,
    render_report_assets_from_output,
)
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
    "ProfilingComparisonSeries",
    "build_profiling_comparison_series",
    "plot_normalized_step_comparison",
    "plot_profiling_totals_comparison",
    "plot_stage_share_comparison",
    "render_profiling_comparison_report",
    "write_profiling_comparison_csv",
    "build_dataset_profiling_summary",
    "build_stage_profiling_summary",
    "plot_stage_profiling",
    "render_dataset_profiling_report",
    "build_auxiliary_performance_rows",
    "build_auxiliary_performance_summary",
    "render_sample_report",
    "render_trace_comparison",
    "build_trace_step_rows",
    "summarize_profiling",
    "write_raw_report",
    "render_profiling_comparison_from_output",
    "render_report_assets_from_output",
]
