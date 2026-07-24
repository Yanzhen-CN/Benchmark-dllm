import pytest

from dllm_bench.metrics.trace_parallelism import (
    compute_final_stable_steps,
    effective_tokens_per_forward,
    finalization_share,
    mean_peak_effective_tokens_per_forward,
    normalized_forward_progress,
)


def test_compute_final_stable_steps_simple():
    # 3 positions, 3 forward steps. Position 0 settles at step 0, position 1 at
    # step 1 (changes then locks), position 2 flips back and forth then locks late.
    sequences = [
        [10, 99, 5],
        [10, 20, 6],
        [10, 20, 7],
    ]
    assert compute_final_stable_steps(sequences) == [0, 1, 2]


def test_compute_final_stable_steps_detects_late_flip_even_if_value_repeats():
    # position matches final value early, flips away, then returns: must not be
    # reported as stable from the earlier occurrence.
    sequences = [
        [1],
        [2],
        [1],
    ]
    assert compute_final_stable_steps(sequences) == [2]


def test_compute_final_stable_steps_empty():
    assert compute_final_stable_steps([]) == []


def test_effective_tokens_per_forward_counts_by_step():
    counts = effective_tokens_per_forward([0, 0, 1, 2, 2, 2], num_steps=3)
    assert counts == {0: 2, 1: 1, 2: 3}


def test_normalized_forward_progress_endpoints():
    assert normalized_forward_progress(0, num_steps=5) == pytest.approx(0.0)
    assert normalized_forward_progress(4, num_steps=5) == pytest.approx(1.0)
    assert normalized_forward_progress(0, num_steps=1) == pytest.approx(0.0)


def test_finalization_share_buckets_and_sums_to_one():
    # num_steps=9 -> thirds are [0,3), [3,6), [6,9) in step index space (0..8)
    final_stable_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    shares = finalization_share(final_stable_steps, num_steps=9)
    assert shares["early"] == pytest.approx(3 / 9)
    assert shares["middle"] == pytest.approx(3 / 9)
    assert shares["late"] == pytest.approx(3 / 9)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_finalization_share_rejects_empty():
    with pytest.raises(ValueError):
        finalization_share([], num_steps=4)


def test_mean_peak_effective_tokens_per_forward():
    mean_val, peak_val = mean_peak_effective_tokens_per_forward([0, 0, 1], num_steps=2)
    assert mean_val == pytest.approx(1.5)
    assert peak_val == 2
