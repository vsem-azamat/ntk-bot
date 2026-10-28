"""Reference curves: every open day resampled onto one 10-minute grid, and the
"norm" for a target day built from the recent days of the same type.

All forecasters work relative to this norm. The level of the library drifts a
lot (semester vs break, and it grows year over year), so absolute counts from old
days are a poor guide; the shape of a day of a given type is stable, and the
recent same-type days give its current level.

Resampling to a fixed grid also makes the 10-minute (before mid-2026) and
20-minute (after) collection cadences look the same to everything downstream.
"""

import bisect
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from apps.hours import GRID_STEP_MIN, day_type, opening_time

STEP_MIN = GRID_STEP_MIN
N_BUCKETS = 24 * 60 // STEP_MIN
# Gaps longer than this are a collector outage, not a smooth change: the grid
# stays empty there instead of drawing a straight line across the hole.
MAX_GAP_MIN = 60
# A day is a usable reference when at least this share of its open window has data.
MIN_DAY_COVERAGE = 0.5
# A day whose busiest moment stays below this was a closure or a stuck counter
# (the data has such days reading 0-12 all day); the quietest real days, summer
# weekends and Christmas Eve, still peak at about 70.
MIN_DAY_PEAK = 50
# Reference days: the last K days of the same type no older than MAX_AGE_DAYS.
REFERENCE_K = 3
REFERENCE_MAX_AGE_DAYS = 28


def bucket_of(dt: datetime) -> int:
    """Index of the 10-minute grid bucket that ``dt`` falls into."""
    return (dt.hour * 60 + dt.minute) // STEP_MIN


def bucket_minutes() -> np.ndarray:
    """Minute-of-day at the start of every bucket."""
    return np.arange(N_BUCKETS, dtype=float) * STEP_MIN


def resample_day(samples: list[tuple[datetime, int]]) -> np.ndarray:
    """Linear interpolation of one day's samples onto the bucket grid.

    Buckets before the first sample, after the last one, or inside a gap longer
    than ``MAX_GAP_MIN`` are NaN. ``samples`` must be sorted and from one day."""
    out = np.full(N_BUCKETS, np.nan)
    if not samples:
        return out
    x = np.array([dt.hour * 60 + dt.minute + dt.second / 60 for dt, _ in samples], float)
    y = np.array([c for _, c in samples], float)
    grid = bucket_minutes()
    inside = (grid >= x[0]) & (grid <= x[-1])
    out[inside] = np.interp(grid[inside], x, y)
    if x.size > 1:
        # Blank buckets that sit inside an over-long gap between two samples.
        right = np.searchsorted(x, grid, side="left").clip(1, x.size - 1)
        gap = x[right] - x[right - 1]
        out[inside & (gap > MAX_GAP_MIN) & (grid > x[right - 1]) & (grid < x[right])] = np.nan
    return out


def open_buckets(day: date) -> int:
    """First bucket of ``day``'s open window."""
    t = opening_time(day)
    return (t.hour * 60 + t.minute) // STEP_MIN


@dataclass(frozen=True)
class RefCurve:
    """Norm for one target day on the bucket grid. All arrays are length
    ``N_BUCKETS`` and NaN-free (edges are held flat); ``empty`` when no
    reference day exists."""

    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    n_days: int

    @property
    def empty(self) -> bool:
        return self.n_days == 0


def _fill_edges(curve: np.ndarray) -> np.ndarray:
    """Replace NaN buckets by interpolating / holding the nearest valid value."""
    valid = ~np.isnan(curve)
    if not valid.any():
        return np.zeros_like(curve)
    idx = np.arange(curve.size)
    return np.interp(idx, idx[valid], curve[valid])


class DayCurves:
    """Index of every usable day resampled onto the bucket grid.

    Built once from the full history; every query only reads days strictly before
    the target day, so the same index serves training, backtests and live
    prediction without leaking the future."""

    def __init__(self, rows: list[tuple[datetime, int]]) -> None:
        by_day: dict[date, list[tuple[datetime, int]]] = defaultdict(list)
        for dt, count in rows:
            if count > 0:
                by_day[dt.date()].append((dt, count))
        self.days: list[date] = []
        curves: list[np.ndarray] = []
        for day in sorted(by_day):
            curve = resample_day(sorted(by_day[day]))
            first = open_buckets(day)
            covered = np.count_nonzero(~np.isnan(curve[first:]))
            full = covered >= MIN_DAY_COVERAGE * (N_BUCKETS - first)
            if full and np.nanmax(curve) >= MIN_DAY_PEAK:
                self.days.append(day)
                curves.append(curve)
        self.curves = np.vstack(curves) if curves else np.empty((0, N_BUCKETS))
        self._types = [day_type(d) for d in self.days]
        self._pos = {d: i for i, d in enumerate(self.days)}
        self._norms: dict[date, RefCurve] = {}
        self._ratios: dict[date, float | None] = {}

    def reference_days(
        self, target: date, k: int = REFERENCE_K, max_age_days: int = REFERENCE_MAX_AGE_DAYS
    ) -> list[int]:
        """Row indices of the last ``k`` usable same-type days before ``target``,
        no older than ``max_age_days``; after a long gap (cold start, a summer
        without data) the latest same-type days from any time."""
        kind = day_type(target)
        end = bisect.bisect_left(self.days, target)
        oldest = target - timedelta(days=max_age_days)
        recent: list[int] = []
        older: list[int] = []
        for i in range(end - 1, -1, -1):
            if self._types[i] != kind:
                continue
            if self.days[i] >= oldest:
                recent.append(i)
            elif len(older) < k:
                older.append(i)
            if len(recent) >= k or (len(older) >= k and not recent):
                break
        chosen = recent or older
        return sorted(chosen[:k])

    def norm(self, target: date) -> RefCurve:
        """Per-bucket p10 / p50 / p90 of the reference days of ``target``."""
        cached = self._norms.get(target)
        if cached is None:
            cached = self._norms[target] = self._build_norm(target)
        return cached

    def _build_norm(self, target: date) -> RefCurve:
        idx = self.reference_days(target)
        if not idx:
            zeros = np.zeros(N_BUCKETS)
            return RefCurve(zeros, zeros, zeros, 0)
        block = self.curves[idx]
        with warnings.catch_warnings():
            # All-NaN buckets (outside every reference day) are filled below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p10, p50, p90 = np.nanpercentile(block, [10, 50, 90], axis=0)
        return RefCurve(_fill_edges(p10), _fill_edges(p50), _fill_edges(p90), len(idx))

    def ratio_to_norm(self, day: date) -> float | None:
        """How busy a usable ``day`` was compared with its own norm (None when it
        is not usable or had no norm)."""
        if day not in self._ratios:
            i = self._pos.get(day)
            ref = self.norm(day)
            self._ratios[day] = (
                None if i is None or ref.empty else level_ratio(self.curves[i], ref.p50)
            )
        return self._ratios[day]

    def recent_ratio(self, target: date, days: int) -> float:
        """Median ratio to norm of the last ``days`` usable days (any type)
        before ``target``: how far the latest days ran above or below what their
        reference days predicted. 1.0 when unknown."""
        end = bisect.bisect_left(self.days, target)
        found = (
            self.ratio_to_norm(self.days[i]) for i in range(end - 1, max(-1, end - 1 - days), -1)
        )
        ratios = [r for r in found if r is not None]
        return float(np.median(ratios)) if ratios else 1.0


# Buckets where the norm is this low (opening minutes, late evening) carry no
# information about the day's level, only noise, so the ratio ignores them.
MIN_NORM_FOR_RATIO = 30.0


def level_ratio(observed: np.ndarray, norm: np.ndarray) -> float | None:
    """Plain ratio ``sum(observed) / sum(norm)`` over buckets observed and with a
    meaningful norm; None when there are none."""
    mask = ~np.isnan(observed) & (norm >= MIN_NORM_FOR_RATIO)
    if not mask.any():
        return None
    return float(observed[mask].sum() / norm[mask].sum())
