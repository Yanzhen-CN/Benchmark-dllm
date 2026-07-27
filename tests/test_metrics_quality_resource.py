import pytest

from dllm_bench.metrics.quality_resource import (
    energy_priority_score,
    resource_adjustment,
    resource_equivalent_quality,
    resource_ratio,
    score_per_compute,
    score_per_unit_energy,
    speed_ratio,
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


def test_speed_ratio_uses_model_tps_over_ar_tps():
    assert speed_ratio(model_tps=40.0, baseline_tps=10.0) == pytest.approx(4.0)


def test_resource_equivalent_quality_perfect_score_is_resource_independent():
    # q = 1 -> ideal quality is 1 regardless of resource ratio.
    assert resource_equivalent_quality(1.0, 0.1) == pytest.approx(1.0)
    assert resource_equivalent_quality(1.0, 100.0) == pytest.approx(1.0)


def test_resource_equivalent_quality_matches_hand_computation():
    # Delta is calibrated by q_AR, then added to the model's own q.
    q_model, q_ar, r = 0.6, 0.5, 2.0
    expected = q_model + ((1 - (1 - q_ar) ** r) - q_ar)
    assert resource_equivalent_quality(q_model, r, q_ar=q_ar) == pytest.approx(expected)


def test_resource_adjustment_can_be_negative_and_q_is_not_clipped():
    delta = resource_adjustment(q_ar=0.5, ratio=0.25)
    assert delta < 0
    assert resource_equivalent_quality(0.1, 0.25, q_ar=0.5) < 0


def test_beta_controls_how_much_of_adjustment_is_used():
    assert resource_equivalent_quality(0.6, 10, q_ar=0.5, beta=0) == pytest.approx(0.6)


def test_scenario_scores_weight_time_and_energy_oppositely():
    q_time, q_energy = 0.9, 0.4
    fast = time_priority_score(q_time, q_energy)
    green = energy_priority_score(q_time, q_energy)
    # time-priority should sit closer to q_time, energy-priority closer to q_energy.
    assert abs(fast - q_time) < abs(green - q_time)
    assert abs(green - q_energy) < abs(fast - q_energy)


def test_scenario_scores_are_linear_nine_to_one_combinations():
    assert time_priority_score(0.8, 0.2) == pytest.approx(0.9 * 0.8 + 0.1 * 0.2)
    assert energy_priority_score(0.8, 0.2) == pytest.approx(0.1 * 0.8 + 0.9 * 0.2)


def test_scenario_score_equal_inputs_collapses_to_that_value():
    assert time_priority_score(0.7, 0.7) == pytest.approx(0.7)
