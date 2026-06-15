"""Single source of truth that turns timestamps (+ optional weather) into the
CatBoost-ready feature matrix used by BOTH training and prediction.

The matrix is a NumPy object array: categorical columns (weekday, month) are
strings (CatBoost requires int/str, never NaN), numeric columns are floats and
may be NaN (CatBoost handles NaN natively).
"""

import bisect
import functools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Literal, overload

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

    by_day: dict[date, list[tuple[datetime, int]]]
    # Lazily built index over the day keys for fast trailing-level queries.
    _sorted_days: list[date] | None = field(default=None, compare=False, repr=False)
    _peaks: list[int] | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_rows(cls, rows: list[tuple[datetime, int]]) -> "ObsContext":
        by_day: dict = defaultdict(list)
        for dt, count in rows:
            if count > 0:
                by_day[dt.date()].append((dt, count))
        for day in by_day:
            by_day[day].sort()
        return cls(by_day=dict(by_day))

    def days_before(self, day: date) -> list[date]:
        return [d for d in self.by_day if d < day]

    def samples_on(self, day: date) -> list[tuple[datetime, int]]:
        return self.by_day.get(day, [])

    def _ensure_index(self) -> None:
        if self._sorted_days is None:
            self._sorted_days = sorted(self.by_day)
            self._peaks = [max(c for _, c in self.by_day[d]) for d in self._sorted_days]

    def trailing_level(self, day: date, trailing_days: int) -> float:
        """Median daily peak over the up-to-``trailing_days`` days strictly before
        ``day`` (the current day is always excluded). O(log n + trailing_days)
        after a one-time O(n log n) index build, vs the naive O(n) per call."""
        self._ensure_index()
        assert self._sorted_days is not None and self._peaks is not None
        i = bisect.bisect_left(self._sorted_days, day)  # days[:i] are strictly < day
        window = self._peaks[max(0, i - trailing_days) : i]
        return float(np.median(window)) if window else 0.0

    def truncated(self, day: date, cut: "datetime | None") -> "ObsContext":
        """Return a copy where ``day``'s samples are limited to those at or before
        ``cut`` (or the day removed entirely when ``cut`` is None). Other days are
        shared by reference. Cheap: only the one day's list is rebuilt."""
        new = dict(self.by_day)
        if cut is None:
            new.pop(day, None)
        else:
            kept = [(dt, c) for dt, c in self.by_day.get(day, []) if dt <= cut]
            if kept:
                new[day] = kept
            else:
                new.pop(day, None)
        return ObsContext(by_day=new)


@functools.lru_cache(maxsize=8)
def _cz_holidays(year: int) -> holidays.HolidayBase:
    return holidays.CZ(years=year)


def regime_features(now: datetime, ctx: ObsContext, trailing_days: int = 14) -> dict:
    """Empirical 'academic calendar': trailing occupancy level (exam vs break)
    and a Czech public-holiday flag. Trailing level ignores the current day."""
    trailing_level = ctx.trailing_level(now.date(), trailing_days)
    is_holiday = 1.0 if now.date() in _cz_holidays(now.year) else 0.0
    return {"trailing_level": trailing_level, "is_holiday": is_holiday}


_REGIME_NAMES = ["trailing_level", "is_holiday"]
_ASOF_NAMES = ["has_today", "last_count", "peak_so_far", "count_delta", "observed_span_min"]


def asof_features(now: datetime, ctx: ObsContext) -> dict:
    """Trajectory state for ``now`` derived from ``ctx``'s samples on ``now.date()``
    with sample-time <= ``now``.

    LEAKAGE CONTRACT: the caller controls the horizon via ``ctx``. At serve time
    ``ctx`` holds only real past data, so future grid points see the latest real
    sample. For TRAINING, the caller MUST pass a ``ctx`` whose same-day samples are
    truncated to strictly before the target timestamp, otherwise the target leaks
    into ``last_count``/``peak_so_far``. See the experiment harness / predictModels
    training-example construction.
    """
    seen = [(dt, c) for dt, c in ctx.samples_on(now.date()) if dt <= now]
    if not seen:
        return {
            "has_today": 0.0,
            "last_count": 0.0,
            "peak_so_far": 0.0,
            "count_delta": 0.0,
            "observed_span_min": 0.0,
        }
    counts = [c for _, c in seen]
    last_count = float(counts[-1])
    prev = float(counts[-2]) if len(counts) >= 2 else last_count
    observed_span_min = (seen[-1][0] - seen[0][0]).total_seconds() / 60.0
    return {
        "has_today": 1.0,
        "last_count": last_count,
        "peak_so_far": float(max(counts)),
        "count_delta": last_count - prev,
        "observed_span_min": float(observed_span_min),
    }


def build_matrix(
    timestamps: list[datetime],
    ctx: "ObsContext | None" = None,
    weather: dict[datetime, WeatherRow] | None = None,
    groups: tuple[str, ...] = ("base", "regime", "asof"),
) -> Features:
    """Build a feature matrix from selectable groups. ``base`` is the legacy
    calendar+weather block; ``regime`` and ``asof`` need ``ctx``. Column order is
    fixed (base, regime, asof) so names and categorical indices stay aligned."""
    valid = {"base", "regime", "asof"}
    unknown = set(groups) - valid
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    names: list[str] = []
    cat_idx: list[int] = []
    if "base" in groups:
        cat_idx = list(CATEGORICAL_INDICES)
        names += list(FEATURE_NAMES)
    if "regime" in groups:
        names += _REGIME_NAMES
    if "asof" in groups:
        names += _ASOF_NAMES

    rows: list[list] = []
    for dt in timestamps:
        row: list = []
        if "base" in groups:
            row += _row(dt, weather)
        if "regime" in groups:
            r = regime_features(dt, ctx) if ctx is not None else {n: 0.0 for n in _REGIME_NAMES}
            row += [r[n] for n in _REGIME_NAMES]
        if "asof" in groups:
            a = asof_features(dt, ctx) if ctx is not None else {n: 0.0 for n in _ASOF_NAMES}
            row += [a[n] for n in _ASOF_NAMES]
        rows.append(row)

    X = np.array(rows, dtype=object) if rows else np.empty((0, len(names)), dtype=object)
    return Features(X=X, names=names, categorical_indices=cat_idx)


@overload
def build_training_matrix(
    rows: list[tuple[datetime, int]],
    train_days: list[date],
    weather: dict[datetime, WeatherRow] | None = ...,
    groups: tuple[str, ...] = ...,
    cut_times: tuple[time | None, ...] = ...,
    return_timestamps: Literal[False] = ...,
) -> tuple[Features, np.ndarray]: ...


@overload
def build_training_matrix(
    rows: list[tuple[datetime, int]],
    train_days: list[date],
    weather: dict[datetime, WeatherRow] | None = ...,
    groups: tuple[str, ...] = ...,
    cut_times: tuple[time | None, ...] = ...,
    *,
    return_timestamps: Literal[True],
) -> tuple[Features, np.ndarray, list[datetime]]: ...


def build_training_matrix(
    rows: list[tuple[datetime, int]],
    train_days: list[date],
    weather: dict[datetime, WeatherRow] | None = None,
    groups: tuple[str, ...] = ("base", "regime", "asof"),
    cut_times: tuple[time | None, ...] = (None,),
    return_timestamps: bool = False,
) -> tuple[Features, np.ndarray] | tuple[Features, np.ndarray, list[datetime]]:
    """Build a leak-free training matrix with as-of augmentation.

    For each training day and each entry in ``cut_times`` (``None`` = cold/morning
    start, a ``time`` = observed up to that moment), build a context truncated to
    that cut and emit one example per actual sample that falls AFTER the cut
    (label = that sample's count). Because targets are strictly after the cut, the
    target never appears in its own as-of context, so ``last_count``/``peak_so_far``
    cannot leak the label.

    With ``return_timestamps=True`` also returns the target timestamp for each row
    (same order as ``X``/``y``), which lets a caller score a second model (e.g. the
    climatology baseline) on the exact same leak-free examples."""
    full = ObsContext.from_rows(rows)
    train_set = set(train_days)
    by_day: dict[date, list[tuple[datetime, int]]] = {}
    for dt, c in rows:
        if dt.date() in train_set and c > 0:
            by_day.setdefault(dt.date(), []).append((dt, c))

    blocks: list[np.ndarray] = []
    labels: list[float] = []
    timestamps: list[datetime] = []
    names: list[str] = []
    cat_idx: list[int] = []
    for day, samples in by_day.items():
        for ct in cut_times:
            cut = None if ct is None else datetime.combine(day, ct)
            tctx = full.truncated(day, cut)
            targets = [(dt, c) for dt, c in samples if cut is None or dt > cut]
            if not targets:
                continue
            feats = build_matrix(
                [dt for dt, _ in targets], ctx=tctx, weather=weather, groups=groups
            )
            blocks.append(feats.X)
            labels.extend(float(c) for _, c in targets)
            timestamps.extend(dt for dt, _ in targets)
            names, cat_idx = feats.names, feats.categorical_indices

    X = np.vstack(blocks) if blocks else np.empty((0, len(names)), dtype=object)
    feats_out = Features(X=X, names=names, categorical_indices=cat_idx)
    y = np.asarray(labels, float)
    if return_timestamps:
        return feats_out, y, timestamps
    return feats_out, y
