"""Calibrated uncertainty band: empirical quantiles of past forecast errors.

The band is not learned by separate quantile models. Instead, ``learn_models``
replays the served forecaster on recent weeks it was not trained on, collects
the relative errors ``log((y + C) / (p50 + C))`` and keeps their 10th and 90th
percentile per horizon (how far ahead of the last real sample a point is).
Applied to a new median this gives a p10-p90 band that held about 80 % of the
real samples on recent data, widening with the horizon and scaling with the
level (a band of +-60 people is right at noon in exam season, not at 08:00).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apps.features import TARGET_OFFSET, to_target

BAND_LO, BAND_HI = 0.1, 0.9
# Horizon buckets in minutes after the last real sample of the day. Bucket 0 is
# a forecast with no sample of the day at all (the morning forecast).
HORIZON_EDGES_MIN = (60.0, 180.0)
COLD = 0
# Fewer residuals than this in a bucket give an unstable quantile; such a bucket
# borrows the pooled quantiles of all buckets instead.
MIN_RESIDUALS = 200


def horizon_bucket(minutes_ahead: float | None) -> int:
    """Bucket of a point ``minutes_ahead`` of the last sample (None = cold)."""
    if minutes_ahead is None:
        return COLD
    return 1 + int(np.searchsorted(HORIZON_EDGES_MIN, minutes_ahead, side="left"))


def relative_error(actual: np.ndarray, p50: np.ndarray) -> np.ndarray:
    """The error the band is built from: the model target of ``actual`` vs ``p50``."""
    return to_target(actual, p50)


@dataclass(frozen=True)
class ResidualBand:
    """``(q_lo, q_hi)`` of the relative error per horizon bucket."""

    quantiles: dict[int, tuple[float, float]]

    @classmethod
    def fit(cls, errors: dict[int, list[float]]) -> ResidualBand:
        pooled = [e for v in errors.values() for e in v]
        if not pooled:
            return cls({})
        fallback = (float(np.quantile(pooled, BAND_LO)), float(np.quantile(pooled, BAND_HI)))
        quantiles = {
            k: (float(np.quantile(v, BAND_LO)), float(np.quantile(v, BAND_HI)))
            if len(v) >= MIN_RESIDUALS
            else fallback
            for k, v in errors.items()
        }
        quantiles.setdefault(-1, fallback)  # -1: pooled, for unseen buckets
        return cls(quantiles)

    def apply(self, p50: np.ndarray, buckets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(p10, p90) around ``p50``; without calibration data the band collapses
        onto the median rather than inventing a width."""
        lo = np.empty_like(p50)
        hi = np.empty_like(p50)
        none = (0.0, 0.0)
        for i, (mid, b) in enumerate(zip(p50, buckets, strict=True)):
            qlo, qhi = self.quantiles.get(int(b), self.quantiles.get(-1, none))
            base = mid + TARGET_OFFSET
            lo[i] = min(mid, max(0.0, base * math.exp(qlo) - TARGET_OFFSET))
            hi[i] = max(mid, base * math.exp(qhi) - TARGET_OFFSET)
        return lo, hi

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({str(k): v for k, v in self.quantiles.items()}))

    @classmethod
    def load(cls, path: Path) -> ResidualBand:
        if not path.exists():
            return cls({})
        raw = json.loads(path.read_text())
        return cls({int(k): (float(v[0]), float(v[1])) for k, v in raw.items()})
