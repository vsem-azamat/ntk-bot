"""Feature rows for the CatBoost model, all expressed relative to the day's norm.

The model predicts ``log((y + C) / (base + C))`` where ``base`` is the
shape-times-level forecast (``apps.level.shape_level``): it only has to learn how
today deviates from "the recent same-type days, rescaled to today's level".
Absolute levels never enter as a target, so year-over-year growth or a new
semester does not push the model outside the range it was trained on.

One function builds the rows for both training and serving, so the two can not
drift apart.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np

from apps.hours import day_type, is_holiday, opening_time
from apps.level import DayState, day_state, shape_level
from apps.reference import STEP_MIN, DayCurves, bucket_of

# Additive offset of the log-ratio target. Near opening the norm is close to 0,
# so a plain ratio explodes; 30 people is the size of the counter residue that
# the library still shows when it is practically empty.
TARGET_OFFSET = 30.0

_DAY_TYPES = ("mon_thu", "fri", "sat", "sun_holiday")

CALENDAR = [
    "minute",
    "weekday",
    "day_type",
    "is_holiday",
    "month",
    "yday_sin",
    "yday_cos",
]
NORM = ["norm", "norm_peak", "norm_spread", "n_ref_days", "recent_ratio"]
TODAY = [
    "has_today",
    "observed_ratio",
    "ratio",
    "observed_buckets",
    "horizon_min",
    "last_ratio",
    "minutes_open",
]

GROUPS: dict[str, list[str]] = {
    "calendar": CALENDAR,
    "norm": NORM,
    "today": TODAY,
}


def feature_names(groups: tuple[str, ...]) -> list[str]:
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    return [name for g in GROUPS if g in groups for name in GROUPS[g]]


def base_curve(state: DayState) -> np.ndarray:
    """The shape-times-level p50 every target is expressed against."""
    return shape_level(state)[1]


def to_target(y: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.log((y + TARGET_OFFSET) / (base + TARGET_OFFSET))


def from_target(z: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, (base + TARGET_OFFSET) * np.exp(z) - TARGET_OFFSET)


def state_rows(
    state: DayState,
    buckets: np.ndarray,
    groups: tuple[str, ...],
) -> np.ndarray:
    """Feature matrix (one row per target bucket) for ``state``'s day."""
    day = state.day
    norm = state.norm
    base = base_curve(state)
    n = buckets.size
    cols: dict[str, np.ndarray] = {}
    if "calendar" in groups:
        yday = 2 * math.pi * (day.timetuple().tm_yday - 1) / 365.25
        cols["minute"] = buckets * float(STEP_MIN)
        cols["weekday"] = np.full(n, float(day.weekday()))
        cols["day_type"] = np.full(n, float(_DAY_TYPES.index(day_type(day))))
        cols["is_holiday"] = np.full(n, 1.0 if is_holiday(day) else 0.0)
        cols["month"] = np.full(n, float(day.month))
        cols["yday_sin"] = np.full(n, math.sin(yday))
        cols["yday_cos"] = np.full(n, math.cos(yday))
    if "norm" in groups:
        cols["norm"] = norm.p50[buckets]
        cols["norm_peak"] = np.full(n, float(norm.p50.max()))
        cols["norm_spread"] = (norm.p90 - norm.p10)[buckets] / (norm.p50[buckets] + TARGET_OFFSET)
        cols["n_ref_days"] = np.full(n, float(norm.n_days))
        cols["recent_ratio"] = np.full(n, state.recent_ratio)
    if "today" in groups:
        known = state.last_bucket is not None
        nan = np.full(n, np.nan)
        cols["has_today"] = np.full(n, 1.0 if known else 0.0)
        cols["observed_ratio"] = (
            np.full(n, state.observed_ratio) if state.observed_ratio is not None else nan
        )
        cols["ratio"] = np.full(n, state.ratio)
        cols["observed_buckets"] = np.full(n, float(state.observed_buckets))
        if state.last_bucket is not None and state.last_count is not None:
            last = state.last_bucket
            cols["horizon_min"] = (buckets - last) * float(STEP_MIN)
            cols["last_ratio"] = np.full(
                n, (state.last_count + TARGET_OFFSET) / (base[last] + TARGET_OFFSET)
            )
        else:
            cols["horizon_min"] = nan
            cols["last_ratio"] = nan
        open_min = opening_time(day).hour * 60 + opening_time(day).minute
        cols["minutes_open"] = buckets * float(STEP_MIN) - open_min
    return np.column_stack([cols[name] for name in feature_names(groups)]).astype(float)


@dataclass(frozen=True)
class TrainingSet:
    """Training rows: features ``X``, log-ratio target ``z`` and the ``base``
    it is relative to."""

    X: np.ndarray
    z: np.ndarray
    base: np.ndarray


def training_rows(
    curves: DayCurves,
    by_day: dict[date, list[tuple[datetime, int]]],
    days: list[date],
    groups: tuple[str, ...],
    cuts: tuple[time | None, ...],
) -> TrainingSet:
    """Leak-free training set: for every day and cut, the state as of the cut
    and one row per real sample strictly after it."""
    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    for day in days:
        samples = by_day.get(day, [])
        if not samples:
            continue
        for ct in cuts:
            if ct is not None and ct < opening_time(day):
                continue  # nothing seen yet: a duplicate of the cold case
            cut = None if ct is None else datetime.combine(day, ct)
            after = [(dt, c) for dt, c in samples if cut is None or dt > cut]
            if not after:
                continue
            state = day_state(curves, day, samples, cut)
            if state.norm.empty:
                continue
            buckets = np.array([bucket_of(dt) for dt, _ in after])
            y = np.array([c for _, c in after], float)
            base = base_curve(state)[buckets]
            blocks.append(state_rows(state, buckets, groups))
            targets.append(to_target(y, base))
            bases.append(base)
    if not blocks:
        empty = np.empty(0)
        return TrainingSet(np.empty((0, len(feature_names(groups)))), empty, empty)
    return TrainingSet(np.vstack(blocks), np.concatenate(targets), np.concatenate(bases))
