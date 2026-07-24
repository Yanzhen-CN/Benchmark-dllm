import pytest

from dllm_bench.metrics.long_context import (
    context_retention,
    long_output_quality_retention,
)


def test_context_retention_ratio():
    assert context_retention(0.8, 0.9) == pytest.approx(0.8 / 0.9)


def test_context_retention_rejects_zero_denominator():
    with pytest.raises(ValueError):
        context_retention(0.8, 0.0)


def test_long_output_quality_retention_ratio():
    assert long_output_quality_retention(70.0, 80.0) == pytest.approx(0.875)


def test_long_output_quality_retention_rejects_zero_denominator():
    with pytest.raises(ValueError):
        long_output_quality_retention(70.0, 0.0)
