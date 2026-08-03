from types import SimpleNamespace

import pytest

from dllm_bench.interfaces import PositionState, TraceStep
from dllm_bench.visual.models.diffusiongemma import (
    _metrics,
    render_model_comparison_visualization,
)


def _generation() -> SimpleNamespace:
    trace = [
        TraceStep(
            forward_index=index,
            token_ids=[1, 2, 3, 4],
            position_states=[PositionState.VISIBLE] * 4,
            committed_positions=[index],
            decoded_text="test",
            entropy_by_position={position: 0.5 for position in range(4)},
        )
        for index in range(2)
    ]
    return SimpleNamespace(
        trace=trace,
        num_forward_passes=2,
        final_valid_length=4,
    )


def test_forward_metrics_use_real_counts_without_normalized_steps():
    records = [(SimpleNamespace(sample_id="sample-1"), _generation())]

    metrics = _metrics(records, block_length=4)

    assert metrics["mean_forward"] == 2
    assert metrics["mean_forward_per_observed_block"] == 2
    assert metrics["weighted_tokens_per_forward"] == 2


def test_forward_figure_is_model_specific_and_writes_summary(tmp_path):
    records = {
        "official": [(SimpleNamespace(sample_id="sample-1"), _generation())]
    }

    written = render_model_comparison_visualization(
        dataset_name="gsm8k",
        records_by_variant=records,
        out_dir=tmp_path,
        block_length=4,
        figures={"forward"},
    )

    assert (tmp_path / "forward_efficiency.png").exists()
    assert (tmp_path / "model_visual_summary.json").exists()
    assert "trace_position_state" not in written


def test_removed_entropy_figure_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported DiffusionGemma figure"):
        render_model_comparison_visualization(
            dataset_name="gsm8k",
            records_by_variant={},
            out_dir=tmp_path,
            figures={"entropy"},
        )
