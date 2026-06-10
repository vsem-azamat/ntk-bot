"""Empirical climatology baseline: per (weekday, 10-min bucket) percentiles.

Doubles as the cold-start / fallback predictor when no trained CatBoost model
beats it. Cheap enough to rebuild on demand from the full history.
"""

from collections import defaultdict
from datetime import datetime

import numpy as np

Climatology = dict[tuple[int, int], tuple[float, float, float]]


def _bucket(dt: datetime) -> tuple[int, int]:
    return (dt.weekday(), dt.hour * 60 + (dt.minute // 10) * 10)


def build_climatology(rows: list[tuple[datetime, int]]) -> Climatology:
    """Map ``(weekday, 10-min-of-day)`` -> ``(p10, p50, p90)`` of historical counts."""
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for dt, count in rows:
        grouped[_bucket(dt)].append(count)
    return {
        key: (
            float(np.percentile(vals, 10)),
            float(np.percentile(vals, 50)),
            float(np.percentile(vals, 90)),
        )
        for key, vals in grouped.items()
    }


def climatology_predict(
    clim: Climatology, timestamps: list[datetime]
) -> tuple[list[float], list[float], list[float]]:
    """Predict (p10, p50, p90) for ``timestamps``; unknown buckets fall back to 0."""
    p10: list[float] = []
    p50: list[float] = []
    p90: list[float] = []
    for dt in timestamps:
        lo, mid, hi = clim.get(_bucket(dt), (0.0, 0.0, 0.0))
        ordered = sorted((max(0.0, lo), max(0.0, mid), max(0.0, hi)))
        p10.append(ordered[0])
        p50.append(ordered[1])
        p90.append(ordered[2])
    return p10, p50, p90
