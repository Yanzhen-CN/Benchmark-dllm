import pytest

from dllm_bench.metrics.stats_utils import aggregate_curve_by_bins, bootstrap_ci, summarize


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


# ---------------------------------------------------------------------------
# aggregate_curve_by_bins (design doc 4.2.1/4.2.4, Appendix C)
# ---------------------------------------------------------------------------

def test_aggregate_curve_by_bins_groups_points_into_the_right_bin():
    # 4 bins over [0,1]: [0,.25) [.25,.5) [.5,.75) [.75,1]
    samples = [[(0.1, 10.0)], [(0.6, 20.0)]]
    result = aggregate_curve_by_bins(samples, n_bins=4)
    assert len(result) == 2
    assert result[0].bin_center == pytest.approx(0.125)
    assert result[0].stats.mean == pytest.approx(10.0)
    assert result[1].bin_center == pytest.approx(0.625)
    assert result[1].stats.mean == pytest.approx(20.0)


def test_aggregate_curve_by_bins_combines_multiple_samples_in_one_bin():
    samples = [[(0.1, 10.0)], [(0.12, 20.0)], [(0.11, 30.0)]]
    result = aggregate_curve_by_bins(samples, n_bins=4)
    assert len(result) == 1
    assert result[0].stats.n == 3
    assert result[0].stats.mean == pytest.approx(20.0)


def test_aggregate_curve_by_bins_gives_each_sample_one_value_per_bin():
    # Multiple checkpoints from one sample are averaged inside the bin first,
    # so a longer trace still contributes exactly one value to that bin.
    samples = [[(0.0, 1.0), (0.1, 3.0), (0.9, 3.0)], [(0.05, 5.0)]]
    result = aggregate_curve_by_bins(samples, n_bins=2)
    assert len(result) == 2
    low_bin = next(p for p in result if p.bin_center < 0.5)
    assert low_bin.stats.n == 2
    assert low_bin.stats.mean == pytest.approx(3.5)


def test_aggregate_curve_by_bins_omits_empty_bins():
    samples = [[(0.9, 1.0)]]
    result = aggregate_curve_by_bins(samples, n_bins=10)
    assert len(result) == 1


def test_aggregate_curve_by_bins_clamps_out_of_range_x():
    samples = [[(-0.5, 1.0), (1.5, 2.0)]]
    result = aggregate_curve_by_bins(samples, n_bins=4)
    bin_centers = [p.bin_center for p in result]
    assert min(bin_centers) == pytest.approx(0.125)
    assert max(bin_centers) == pytest.approx(0.875)


def test_aggregate_curve_by_bins_no_points_is_empty_result():
    assert aggregate_curve_by_bins([], n_bins=4) == []
    assert aggregate_curve_by_bins([[]], n_bins=4) == []


def test_aggregate_curve_by_bins_rejects_non_positive_bins():
    with pytest.raises(ValueError):
        aggregate_curve_by_bins([[(0.5, 1.0)]], n_bins=0)
