"""Forecast-quality metrics for the experiment harness.

All functions take plain lists of floats over a shared, ordered time grid.
Kept dependency-light (only the stdlib + numpy) so they are trivial to unit test.
"""

import numpy as np


def mae(pred: list[float], actual: list[float]) -> float:
    """Mean absolute error over the whole grid."""
    p, a = np.asarray(pred, float), np.asarray(actual, float)
    return float(np.mean(np.abs(p - a)))


def near_term_mae(pred: list[float], actual: list[float], steps: int) -> float:
    """MAE over only the first ``steps`` grid points (the near-term horizon)."""
    return mae(pred[:steps], actual[:steps])


def peak_value_error(pred: list[float], actual: list[float]) -> float:
    """Absolute error between the predicted and actual daily peak magnitude."""
    return abs(float(max(pred)) - float(max(actual)))


def peak_time_error_minutes(pred: list[float], actual: list[float], step_minutes: int) -> float:
    """Absolute error between predicted and actual peak time, in minutes."""
    pi = int(np.argmax(pred))
    ai = int(np.argmax(actual))
    return float(abs(pi - ai) * step_minutes)


def pinball_loss(actual: list[float], pred: list[float], alpha: float) -> float:
    """Mean pinball (quantile) loss for quantile ``alpha``.

    When pred > actual the (1-alpha) weight applies; when pred < actual the
    alpha weight applies, so high-alpha quantiles penalise over-prediction more.
    """
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    diff = p - a
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def coverage(lo: list[float], hi: list[float], actual: list[float]) -> float:
    """Fraction of actual points inside the inclusive ``[lo, hi]`` band."""
    lo_a, hi_a, a = np.asarray(lo, float), np.asarray(hi, float), np.asarray(actual, float)
    inside = (a >= lo_a) & (a <= hi_a)
    return float(np.mean(inside))
