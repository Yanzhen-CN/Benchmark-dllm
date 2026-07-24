import pytest

from dllm_bench.metrics.quality_resource import (
    energy_priority_score,
    resource_equivalent_quality,
    resource_ratio,
    score_per_compute,
    score_per_unit_energy,
    time_priority_score,
)


def test_score_per_unit_energy():
    assert score_per_unit_energy(0.5, 2.0) == pytest.approx(0.25)


def test_score_per_compute():
    assert score_per_compute(0.8, 4.0) == pytest.approx(0.2)


def test_score_per_unit_energy_rejects_invalid_q():
    with pytest.raises(ValueError):
        score_per_unit_energy(1.5, 2.0)


def test_score_per_unit_energy_rejects_nonpositive_energy():
    with pytest.raises(ValueError):
        score_per_unit_energy(0.5, 0.0)


def test_resource_ratio():
    assert resource_ratio(baseline_value=10.0, model_value=5.0) == pytest.approx(2.0)


def test_resource_equivalent_quality_perfect_score_is_resource_independent():
    # q = 1 -> ideal quality is 1 regardless of resource ratio.
    assert resource_equivalent_quality(1.0, 0.1) == pytest.approx(1.0)
    assert resource_equivalent_quality(1.0, 100.0) == pytest.approx(1.0)


def test_resource_equivalent_quality_matches_hand_computation():
    # Q = 1 - (1 - q)^r
    q, r = 0.6, 2.0
    expected = 1 - (1 - q) ** r
    assert resource_equivalent_quality(q, r) == pytest.approx(expected)


def test_scenario_scores_weight_time_and_energy_oppositely():
    q_time, q_energy = 0.9, 0.4
    fast = time_priority_score(q_time, q_energy)
    green = energy_priority_score(q_time, q_energy)
    # time-priority should sit closer to q_time, energy-priority closer to q_energy.
    assert abs(fast - q_time) < abs(green - q_time)
    assert abs(green - q_energy) < abs(fast - q_energy)


def test_scenario_score_equal_inputs_collapses_to_that_value():
    assert time_priority_score(0.7, 0.7) == pytest.approx(0.7)
