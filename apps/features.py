"""Single source of truth that turns timestamps (+ optional weather) into the
CatBoost-ready feature matrix used by BOTH training and prediction.

The matrix is a NumPy object array: categorical columns (weekday, month) are
strings (CatBoost requires int/str, never NaN), numeric columns are floats and
may be NaN (CatBoost handles NaN natively).
"""

import functools
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import holidays
import numpy as np

FEATURE_NAMES = [
    "minute_sin",
    "minute_cos",
    "is_weekend",
    "yday_sin",
    "yday_cos",
    "weekday",
    "month",
    "temp",
    "precip",
    "cloud",
    "wind",
]
CATEGORICAL_INDICES = [5, 6]

WeatherRow = tuple[float | None, float | None, float | None, float | None]


@dataclass
class Features:
    """Built feature matrix plus the metadata CatBoost needs."""

    X: np.ndarray
    names: list[str]
    categorical_indices: list[int]


def _row(dt: datetime, weather: dict[datetime, WeatherRow] | None) -> list:
    minute_of_day = dt.hour * 60 + dt.minute
    minute_angle = 2 * math.pi * minute_of_day / 1440
    yday_angle = 2 * math.pi * (dt.timetuple().tm_yday - 1) / 365.25

    hour_key = dt.replace(minute=0, second=0, microsecond=0)
    temp, precip, cloud, wind = (weather or {}).get(hour_key, (None, None, None, None))

    def num(v: float | None) -> float:
        return float("nan") if v is None else float(v)

    return [
        math.sin(minute_angle),
        math.cos(minute_angle),
        1.0 if dt.weekday() >= 5 else 0.0,
        math.sin(yday_angle),
        math.cos(yday_angle),
        str(dt.weekday()),
        str(dt.month),
        num(temp),
        num(precip),
        num(cloud),
        num(wind),
    ]


def build_features(
    timestamps: list[datetime], weather: dict[datetime, WeatherRow] | None = None
) -> Features:
    """Build the feature matrix for ``timestamps``, joining ``weather`` by hour."""
    if not timestamps:
        X = np.empty((0, len(FEATURE_NAMES)), dtype=object)
    else:
        X = np.array([_row(dt, weather) for dt in timestamps], dtype=object)
    return Features(X=X, names=list(FEATURE_NAMES), categorical_indices=list(CATEGORICAL_INDICES))


@dataclass
class ObsContext:
    """Occupancy history used to derive regime + as-of features without leakage.

    ``by_day`` maps a ``date`` to its sorted ``(datetime, count)`` samples. Every
    derived feature is computed strictly from samples at or before a cut time, so
    training and serving see identical inputs.
    """

    by_day: dict

    @classmethod
    def from_rows(cls, rows: list[tuple[datetime, int]]) -> "ObsContext":
        by_day: dict = defaultdict(list)
        for dt, count in rows:
            if count > 0:
                by_day[dt.date()].append((dt, count))
        for day in by_day:
            by_day[day].sort()
        return cls(by_day=dict(by_day))

    def days_before(self, day) -> list:
        return [d for d in self.by_day if d < day]

    def samples_on(self, day) -> list:
        return self.by_day.get(day, [])


@functools.lru_cache(maxsize=8)
def _cz_holidays(year: int) -> holidays.HolidayBase:
    return holidays.CZ(years=year)


def regime_features(now: datetime, ctx: ObsContext, trailing_days: int = 14) -> dict:
    """Empirical 'academic calendar': trailing occupancy level (exam vs break)
    and a Czech public-holiday flag. Trailing level ignores the current day."""
    today = now.date()
    past_days = sorted(ctx.days_before(today))[-trailing_days:]
    daily_peaks = [max(c for _, c in ctx.samples_on(d)) for d in past_days]
    trailing_level = float(np.median(daily_peaks)) if daily_peaks else 0.0
    is_holiday = 1.0 if now.date() in _cz_holidays(now.year) else 0.0
    return {"trailing_level": trailing_level, "is_holiday": is_holiday}


def asof_features(now: datetime, ctx: ObsContext) -> dict:
    """Features describing today's trajectory up to ``now`` (inclusive). Computed
    strictly from samples at or before ``now`` so they never leak the future."""
    seen = [(dt, c) for dt, c in ctx.samples_on(now.date()) if dt <= now]
    if not seen:
        return {
            "has_today": 0.0,
            "last_count": 0.0,
            "peak_so_far": 0.0,
            "slope": 0.0,
            "minutes_since_first": 0.0,
        }
    counts = [c for _, c in seen]
    last_count = float(counts[-1])
    prev = float(counts[-2]) if len(counts) >= 2 else last_count
    minutes_since_first = (seen[-1][0] - seen[0][0]).total_seconds() / 60.0
    return {
        "has_today": 1.0,
        "last_count": last_count,
        "peak_so_far": float(max(counts)),
        "slope": last_count - prev,
        "minutes_since_first": float(minutes_since_first),
    }
