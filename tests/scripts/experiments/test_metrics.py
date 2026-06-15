import math

from scripts.experiments.metrics import (
    coverage,
    mae,
    near_term_mae,
    peak_time_error_minutes,
    peak_value_error,
    pinball_loss,
)


def test_mae_basic():
    assert mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == 2.0 / 3.0


def test_peak_value_error_uses_max_of_each():
    assert peak_value_error([10, 90, 20], [10, 100, 20]) == 10.0


def test_peak_time_error_minutes_uses_step():
    err = peak_time_error_minutes([1, 9, 2, 3], [1, 2, 3, 9], step_minutes=10)
    assert err == 20.0


def test_pinball_loss_p90_underprediction_penalized_more():
    # p90: an actual ABOVE the predicted quantile (under-prediction) is the
    # costly error and carries the alpha=0.9 weight.
    under = pinball_loss([100.0], [80.0], alpha=0.9)  # actual 100 > pred 80
    over = pinball_loss([100.0], [120.0], alpha=0.9)  # actual 100 < pred 120
    assert math.isclose(under, 0.9 * 20.0)  # 18.0
    assert math.isclose(over, 0.1 * 20.0)  # 2.0


def test_coverage_counts_inside_band():
    lo = [0.0, 0.0, 0.0, 0.0]
    hi = [10.0, 10.0, 10.0, 10.0]
    actual = [5.0, 15.0, 5.0, -1.0]
    assert coverage(lo, hi, actual) == 0.5


def test_near_term_mae_limits_horizon():
    pred = [1.0, 1.0, 100.0]
    actual = [2.0, 3.0, 0.0]
    assert near_term_mae(pred, actual, steps=2) == 1.5
