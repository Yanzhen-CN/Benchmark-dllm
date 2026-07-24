import pytest

from dllm_bench.metrics.stats_utils import bootstrap_ci, summarize


def test_bootstrap_ci_single_value_is_degenerate():
    lo, hi = bootstrap_ci([5.0])
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_is_reproducible_given_same_seed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    lo1, hi1 = bootstrap_ci(values, n_resamples=500, seed=42)
    lo2, hi2 = bootstrap_ci(values, n_resamples=500, seed=42)
    assert (lo1, hi1) == (lo2, hi2)


def test_bootstrap_ci_brackets_the_sample_mean_for_low_variance_data():
    values = [10.0, 10.1, 9.9, 10.05, 9.95]
    lo, hi = bootstrap_ci(values, n_resamples=1000, seed=42)
    assert lo <= sum(values) / len(values) <= hi


def test_summarize_basic_shape():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = summarize(values, seed=42)
    assert stats.n == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.iqr_low <= stats.median <= stats.iqr_high
    assert stats.ci_low <= stats.ci_high


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])
