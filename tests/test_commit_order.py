import pytest

from dllm_bench.metrics.commit_order import (
    aggregate_commit_order,
    commit_order_tau_windows,
    kendall_tau_b,
)


def test_kendall_tau_b_perfect_agreement():
    x = [0, 1, 2, 3]
    y = [0, 1, 2, 3]
    assert kendall_tau_b(x, y) == pytest.approx(1.0)


def test_kendall_tau_b_perfect_disagreement():
    x = [0, 1, 2, 3]
    y = [3, 2, 1, 0]
    assert kendall_tau_b(x, y) == pytest.approx(-1.0)


def test_kendall_tau_b_matches_scipy_when_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    x = [0, 1, 2, 3, 4, 5]
    y = [1, 0, 2, 4, 3, 5]
    expected = scipy_stats.kendalltau(x, y).statistic
    assert kendall_tau_b(x, y) == pytest.approx(expected)


def test_kendall_tau_b_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        kendall_tau_b([0, 1], [0, 1, 2])


def test_kendall_tau_b_rejects_too_short():
    with pytest.raises(ValueError):
        kendall_tau_b([0], [0])


def test_kendall_tau_b_all_ties_returns_zero():
    assert kendall_tau_b([1, 1, 1], [0, 5, 9]) == pytest.approx(0.0)


def test_commit_order_tau_windows_skips_windows_larger_than_sequence():
    positions = list(range(10))
    final_stable_steps = list(range(10))
    result = commit_order_tau_windows(positions, final_stable_steps, window_sizes=(4, 8, 16))
    assert set(result.keys()) == {4, 8}
    assert 16 not in result


def test_commit_order_tau_windows_perfect_order_gives_tau_one_everywhere():
    positions = list(range(16))
    final_stable_steps = list(range(16))
    result = commit_order_tau_windows(positions, final_stable_steps, window_sizes=(4, 8))
    for taus in result.values():
        for tau in taus:
            assert tau == pytest.approx(1.0)


def test_aggregate_commit_order_flattens_across_samples():
    per_sample = [
        {4: [1.0, 0.5]},
        {4: [0.0]},
        {8: [-1.0]},
    ]
    summary = aggregate_commit_order(per_sample)
    assert summary[4].n == 3
    assert summary[4].mean == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert summary[8].n == 1
