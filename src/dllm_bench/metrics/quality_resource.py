"""Quality/resource metrics from design document sections 3.2 and 3.3.

The resource adjustment is deliberately calibrated with the AR baseline's
quality.  ``q_model`` is only the model's measured quality to which that
common adjustment is added; using it inside the retry curve would make the
value of a resource advantage depend on the model being compared.
"""

from __future__ import annotations


def _validate_unit_score(q: float, name: str = "q") -> None:
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {q}")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def score_per_unit_energy(q: float, eps: float) -> float:
    """Section 3.2 ``q / EPS`` (score per unit energy-rate)."""
    _validate_unit_score(q)
    _validate_positive(eps, "eps")
    return q / eps


def score_per_compute(q: float, cps: float) -> float:
    """Section 3.2 ``q / CPS`` (score per compute-rate)."""
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


def resource_adjustment(q_ar: float, ratio: float) -> float:
    """Common AR-calibrated adjustment ``Delta(r)``; it may be negative."""
    _validate_unit_score(q_ar, "q_ar")
    _validate_positive(ratio, "ratio")
    return (1.0 - (1.0 - q_ar) ** ratio) - q_ar


def resource_equivalent_quality(
    q_model: float,
    ratio: float,
    *,
    q_ar: float | None = None,
    beta: float = 100.0,
) -> float:
    """Return ``Q(r,beta) = q_model + beta/100 * Delta_AR(r)``.

    ``q_ar`` is required by the design.  It defaults to ``q_model`` only for
    backward compatibility with callers of the pre-design API; leaderboard
    code always passes the actual AR quality explicitly.  The result is not
    clipped: Q is an inferred score on q's scale, not an accuracy.
    """
    _validate_unit_score(q_model, "q_model")
    if q_ar is None:
        q_ar = q_model
    if not 0.0 <= beta <= 100.0:
        raise ValueError("beta must be in [0, 100]")
    return q_model + (beta / 100.0) * resource_adjustment(q_ar, ratio)


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
