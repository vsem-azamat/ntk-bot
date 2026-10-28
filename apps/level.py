"""What is known about a day at a given moment, relative to its norm.

``day_state`` condenses the target day's norm (see ``apps.reference``) and the
samples observed so far into a level ratio: how busy today runs compared with
its recent same-type days. ``shape_level`` is the simple forecaster built on it
(norm shape times today's level); the CatBoost model learns corrections on top
of the same quantity.
"""

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from apps.reference import MIN_NORM_FOR_RATIO, DayCurves, RefCurve, bucket_of, resample_day

# Shrinkage of the observed level ratio towards the prior, in buckets of
# observation: with fewer observed buckets than this the prior dominates, so a
# single noisy morning sample cannot rescale the whole day.
RATIO_PRIOR_BUCKETS = 6.0
# The last few buckets count more: the level right now is the best guide to the
# next hours, older buckets mostly confirm it.
RECENT_BUCKETS = 3
RECENT_WEIGHT = 3.0
# How many of the latest days feed ``recent_ratio`` (see
# ``DayCurves.recent_ratio``). It is a model feature only: applied directly it
# hurts, because single days around holidays and regime changes swing it widely.
RECENT_RATIO_DAYS = 2


@dataclass(frozen=True)
class DayState:
    """Snapshot of one day at a cut. ``today`` is the day's resampled curve up
    to the last observed sample (NaN after it, all NaN when nothing is known)."""

    day: date
    norm: RefCurve
    today: np.ndarray
    last_bucket: int | None
    last_count: float | None
    recent_ratio: float
    observed_ratio: float | None
    ratio: float
    observed_buckets: int


def _weighted_ratio(today: np.ndarray, norm: np.ndarray) -> tuple[float | None, int]:
    mask = ~np.isnan(today) & (norm >= MIN_NORM_FOR_RATIO)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return None, 0
    weights = np.ones(n)
    weights[-RECENT_BUCKETS:] = RECENT_WEIGHT
    ratio = float(np.sum(weights * today[mask]) / np.sum(weights * norm[mask]))
    return ratio, n


def day_state(
    curves: DayCurves,
    day: date,
    samples: list[tuple[datetime, int]],
    cut: datetime | None,
) -> DayState:
    """State of ``day`` as of ``cut`` from ``day``'s ``samples`` (only those at
    or before ``cut`` are used; ``cut=None`` means nothing is known yet)."""
    norm = curves.norm(day)
    seen = sorted(
        (dt, c)
        for dt, c in samples
        if dt.date() == day and c > 0 and (cut is not None and dt <= cut)
    )
    today = resample_day(seen)
    observed, n = _weighted_ratio(today, norm.p50)
    # Shrink towards 1.0 (= "a normal day of this type").
    ratio = 1.0
    if observed is not None:
        ratio = (RATIO_PRIOR_BUCKETS + n * observed) / (RATIO_PRIOR_BUCKETS + n)
    last_bucket = bucket_of(seen[-1][0]) if seen else None
    last_count = float(seen[-1][1]) if seen else None
    recent = curves.recent_ratio(day, RECENT_RATIO_DAYS)
    return DayState(day, norm, today, last_bucket, last_count, recent, observed, ratio, n)


def shape_level(state: DayState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Norm shape scaled by today's level ratio: (p10, p50, p90) per bucket."""
    r = state.ratio
    return state.norm.p10 * r, state.norm.p50 * r, state.norm.p90 * r
