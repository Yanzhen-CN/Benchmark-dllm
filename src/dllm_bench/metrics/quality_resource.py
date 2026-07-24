"""Section 3.2/3.3: unit-resource efficiency and AR-relative resource conversion.

    Score per Unit Energy = q / EnergyPerSample
    Score per Compute     = q / ComputePerSample
    r_time   = T_AR / T_model
    r_energy = E_AR / E_model
    Q_time   = 1 - (1 - q) ** r_time
    Q_energy = 1 - (1 - q) ** r_energy
    S_time-priority   = Q_time**0.9 * Q_energy**0.1     (gamma = 10)
    S_energy-priority = Q_time**0.1 * Q_energy**0.9     (gamma = 90)

Score-per-unit metrics are only meaningful within one dataset (3.2); the two
priority scenarios are deployment-preference views and are never combined
into a single cross-scenario ranking (3.3).
"""

from __future__ import annotations


def _validate_unit_score(q: float) -> None:
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")


def score_per_unit_energy(q: float, energy_per_sample: float) -> float:
    _validate_unit_score(q)
    if energy_per_sample <= 0:
        raise ValueError("energy_per_sample must be positive")
    return q / energy_per_sample


def score_per_compute(q: float, compute_per_sample: float) -> float:
    _validate_unit_score(q)
    if compute_per_sample <= 0:
        raise ValueError("compute_per_sample must be positive")
    return q / compute_per_sample


def resource_ratio(baseline_value: float, model_value: float) -> float:
    """r_time or r_energy: baseline (AR) resource divided by the model's."""
    if model_value <= 0:
        raise ValueError("model_value must be positive")
    if baseline_value <= 0:
        raise ValueError("baseline_value must be positive")
    return baseline_value / model_value


def resource_equivalent_quality(q: float, resource_ratio_value: float) -> float:
    """Q_time or Q_energy: ideal independent-retry resource-equivalent quality upper bound."""
    _validate_unit_score(q)
    if resource_ratio_value <= 0:
        raise ValueError("resource_ratio_value must be positive")
    return 1 - (1 - q) ** resource_ratio_value


def scenario_score(q_time: float, q_energy: float, *, time_weight: float) -> float:
    """S_time-priority (time_weight=0.9) or S_energy-priority (time_weight=0.1)."""
    _validate_unit_score(q_time)
    _validate_unit_score(q_energy)
    if not 0.0 <= time_weight <= 1.0:
        raise ValueError("time_weight must be in [0, 1]")
    return q_time**time_weight * q_energy ** (1 - time_weight)


def time_priority_score(q_time: float, q_energy: float) -> float:
    return scenario_score(q_time, q_energy, time_weight=0.9)


def energy_priority_score(q_time: float, q_energy: float) -> float:
    return scenario_score(q_time, q_energy, time_weight=0.1)
