"""Recent-days baseline: the median curve of the last few days of the same type.

For a target day it takes the last ``k`` days with the same ``day_type`` (Mon–Thu,
Fri, Sat, Sun/holiday) within ``max_age_days`` and computes per-time-of-day
percentiles, interpolated linearly between sample times so the curve stays smooth
whatever the collection interval. Doubles as the cold-start / fallback predictor
when no trained CatBoost model beats it.

Recent days rather than the whole history because the level drifts strongly
(semester vs break, and it grows year over year); an all-history median was
off by 200+ people in exam periods.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from apps.hours import day_type
from config import cnfg

DEFAULT_K = 6
DEFAULT_MAX_AGE_DAYS = 56
# A day counts as a usable reference curve when it has at least this share of
# the samples a full open window yields at the configured collection interval.
_MIN_DAY_COVERAGE = 0.5
_OPEN_WINDOW_MIN = 14 * 60  # shortest open window (weekend 10:00-24:00)


def _min_day_samples() -> int:
    return max(1, int(_OPEN_WINDOW_MIN / cnfg.DELTA_TIME_FOR_RECIEVE_NTK * _MIN_DAY_COVERAGE))


_BUCKET_MIN = 10


@dataclass
class Climatology:
    """Per-bucket percentiles (minute-of-day -> value) for one target day."""

    minutes: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray

    @property
    def empty(self) -> bool:
        return self.minutes.size == 0


def _reference_days(
    by_day: dict[date, list[tuple[datetime, int]]], target: date, k: int, max_age_days: int
) -> list[date]:
    kind = day_type(target)
    min_samples = _min_day_samples()
    oldest = target - timedelta(days=max_age_days)
    candidates = [
        d
        for d in sorted(by_day)
        if oldest <= d < target and day_type(d) == kind and len(by_day[d]) >= min_samples
    ]
    if not candidates:
        # Nothing recent (long gap / cold start): fall back to the latest days of
        # this type from any time, so the bot still draws something sensible.
        candidates = [
            d
            for d in sorted(by_day)
            if d < target and day_type(d) == kind and len(by_day[d]) >= min_samples
        ]
    return candidates[-k:]


def build_climatology(
    rows: list[tuple[datetime, int]],
    target: date,
    k: int = DEFAULT_K,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> Climatology:
    """Percentile curve for ``target`` from the last ``k`` same-type days before it.

    ``rows`` must already be restricted to open-hours, non-zero samples."""
    by_day: dict[date, list[tuple[datetime, int]]] = defaultdict(list)
    for dt, count in rows:
        if dt.date() < target:
            by_day[dt.date()].append((dt, count))

    grouped: dict[int, list[int]] = defaultdict(list)
    for d in _reference_days(by_day, target, k, max_age_days):
        for dt, count in by_day[d]:
            grouped[(dt.hour * 60 + dt.minute) // _BUCKET_MIN * _BUCKET_MIN].append(count)

    minutes = np.array(sorted(grouped), float)
    if minutes.size == 0:
        empty = np.array([], float)
        return Climatology(empty, empty, empty, empty)
    vals = [grouped[int(m)] for m in minutes]
    return Climatology(
        minutes=minutes,
        p10=np.array([np.percentile(v, 10) for v in vals]),
        p50=np.array([np.percentile(v, 50) for v in vals]),
        p90=np.array([np.percentile(v, 90) for v in vals]),
    )


def climatology_predict(
    clim: Climatology, timestamps: list[datetime]
) -> tuple[list[float], list[float], list[float]]:
    """Predict (p10, p50, p90) for ``timestamps`` by interpolating the bucket curve;
    an empty climatology predicts zeros."""
    if clim.empty:
        zeros = [0.0] * len(timestamps)
        return zeros, list(zeros), list(zeros)
    x = np.array([dt.hour * 60 + dt.minute for dt in timestamps], float)
    bands = [np.interp(x, clim.minutes, series) for series in (clim.p10, clim.p50, clim.p90)]
    p10: list[float] = []
    p50: list[float] = []
    p90: list[float] = []
    for lo, mid, hi in zip(*bands, strict=True):
        ordered = sorted((max(0.0, float(lo)), max(0.0, float(mid)), max(0.0, float(hi))))
        p10.append(ordered[0])
        p50.append(ordered[1])
        p90.append(ordered[2])
    return p10, p50, p90
