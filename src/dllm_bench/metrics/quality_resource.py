"""Quality/resource metrics from design document sections 3.2 and 3.3."""

from __future__ import annotations


def _validate_unit_score(q: float, name: str = "q") -> None:
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {q}")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def score_per_unit_energy(q: float, energy_per_sample: float) -> float:
    """Section 3.2 ``q / EnergyPerSample`` (score per joule per sample)."""
    _validate_unit_score(q)
    _validate_positive(energy_per_sample, "energy_per_sample")
    return q / energy_per_sample


def score_per_compute(q: float, cps: float) -> float:
    """Section 3.2 ``q / Cps`` (score per compute-rate)."""
    _validate_unit_score(q)
    _validate_positive(cps, "cps")
    return q / cps


def resource_ratio(baseline_value: float, model_value: float) -> float:
    """A resource saving ratio (AR resource divided by model resource)."""
    _validate_positive(model_value, "model_value")
    _validate_positive(baseline_value, "baseline_value")
    return baseline_value / model_value


def speed_ratio(model_tps: float, baseline_tps: float) -> float:
    """Section 3.3 ``TPS_model / TPS_AR``."""
    _validate_positive(model_tps, "model_tps")
    _validate_positive(baseline_tps, "baseline_tps")
    return model_tps / baseline_tps


def resource_adjustment(q_model: float, ratio: float) -> float:
    """Pairwise retry adjustment computed from the evaluated model's quality."""
    _validate_unit_score(q_model, "q_model")
    _validate_positive(ratio, "ratio")
    return (1.0 - (1.0 - q_model) ** ratio) - q_model


def resource_equivalent_quality(
    q_model: float,
    ratio: float,
    *,
    beta: float = 100.0,
) -> float:
    """Return directional ``Q_A|B(r, beta)`` for pairwise sensitivity analysis."""
    _validate_unit_score(q_model, "q_model")
    if not 0.0 <= beta <= 100.0:
        raise ValueError("beta must be in [0, 100]")
    return q_model + (beta / 100.0) * resource_adjustment(q_model, ratio)


def scenario_score(q_speed: float, q_energy: float, *, gamma: float) -> float:
    """Linear scenario combination from section 3.3; values are not clipped."""
    if not 0.0 <= gamma <= 100.0:
        raise ValueError("gamma must be in [0, 100]")
    energy_weight = gamma / 100.0
    return (1.0 - energy_weight) * q_speed + energy_weight * q_energy


def time_priority_score(q_speed: float, q_energy: float) -> float:
    return scenario_score(q_speed, q_energy, gamma=10.0)


def energy_priority_score(q_speed: float, q_energy: float) -> float:
    return scenario_score(q_speed, q_energy, gamma=90.0)
