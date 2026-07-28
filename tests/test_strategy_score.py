import pytest

from dllm_bench.metrics.strategy_score import (
    auc_trapezoid,
    first_formation_distribution,
    is_eligible,
    normalized_progress_series,
    pairwise_structure_first_score,
    strategy_score,
    summarize_strategy_scores,
)


def test_normalized_progress_series_clips_to_one_and_scales_by_final():
    series = normalized_progress_series([0.0, 0.5, 1.0, 2.0])
    # final score is 2.0, so 0.5/2.0=0.25, 1.0/2.0=0.5, then 2.0/2.0=1.0 but clipped anyway
    assert series == pytest.approx([0.0, 0.25, 0.5, 1.0])


def test_auc_trapezoid_constant_series():
    assert auc_trapezoid([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_auc_trapezoid_linear_ramp_is_one_half():
    assert auc_trapezoid([0.0, 0.5, 1.0]) == pytest.approx(0.5)


def test_auc_trapezoid_single_point():
    assert auc_trapezoid([0.7]) == pytest.approx(0.7)


def test_is_eligible_threshold():
    assert is_eligible(0.5, 0.5) is True
    assert is_eligible(0.49, 0.9) is False


def test_first_formation_distribution_uses_first_attainment_only():
    distribution = first_formation_distribution([0.0, 0.5, 0.25, 1.0])
    assert distribution == pytest.approx([(1 / 3, 0.5), (1.0, 0.5)])


def test_strategy_score_structure_first_is_one():
    # structure/form finishes almost immediately (front-loaded), content ramps
    # up only at the end -> every structure/content pair is concordant.
    form_scores = [1.0, 1.0, 1.0, 1.0]
    content_scores = [0.0, 0.0, 0.0, 1.0]
    score = strategy_score(form_scores, content_scores)
    assert score is not None
    assert score == pytest.approx(1.0)


def test_strategy_score_content_first_is_zero():
    form_scores = [0.0, 0.0, 0.0, 1.0]
    content_scores = [1.0, 1.0, 1.0, 1.0]
    score = strategy_score(form_scores, content_scores)
    assert score is not None
    assert score == pytest.approx(0.0)


def test_strategy_score_synchronized_is_one_half():
    form_scores = [0.0, 0.5, 1.0]
    content_scores = [0.0, 0.5, 1.0]
    score = strategy_score(form_scores, content_scores)
    assert score == pytest.approx(0.5)


def test_pairwise_score_handles_mixed_order_as_pairwise_probability():
    # Half of structure forms first, half last; content forms in the middle.
    score = pairwise_structure_first_score(
        [0.5, 0.5, 1.0],
        [0.0, 1.0, 1.0],
    )
    assert score == pytest.approx(0.5)


def test_strategy_score_is_none_when_ineligible():
    form_scores = [0.0, 0.1, 0.2]
    content_scores = [0.0, 0.1, 0.2]
    assert strategy_score(form_scores, content_scores) is None


def test_strategy_score_stays_in_unit_range():
    # extreme case: form always saturated (AUC=1), content stays at 0 until the
    # very last point (AUC close to 0) -> raw could exceed 100 without clipping.
    form_scores = [1.0] * 10
    content_scores = [0.0] * 9 + [1.0]
    score = strategy_score(form_scores, content_scores)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_summarize_strategy_scores_computes_eligible_ratio():
    scores = [0.0, 0.5, None, 1.0, None]
    summary, ratio = summarize_strategy_scores(scores)
    assert ratio == pytest.approx(3 / 5)
    assert summary.n == 3
    assert summary.mean == pytest.approx(0.5)
