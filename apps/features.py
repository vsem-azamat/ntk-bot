"""Single source of truth that turns timestamps (+ optional weather) into the
CatBoost-ready feature matrix used by BOTH training and prediction.

The matrix is a NumPy object array: categorical columns (weekday, month) are
strings (CatBoost requires int/str, never NaN), numeric columns are floats and
may be NaN (CatBoost handles NaN natively).
"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

FEATURE_NAMES = [
    "minute_sin", "minute_cos", "is_weekend",
    "yday_sin", "yday_cos", "weekday", "month",
    "temp", "precip", "cloud", "wind",
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
    X = np.array([_row(dt, weather) for dt in timestamps], dtype=object)
    return Features(X=X, names=list(FEATURE_NAMES), categorical_indices=list(CATEGORICAL_INDICES))
