# Prediction Engine Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve NTK occupancy forecast accuracy (overall MAE, peak value/time, band calibration, near-term) and make the intraday forecast condition on today's observed samples via a conditional quantile CatBoost model plus an anchor guardrail, with the winner chosen by an offline backtest harness.

**Architecture:** A new offline experiment harness (`scripts/experiments/`) backtests candidate feature sets / model configs with leak-free "as-of" evaluation over the full production snapshot. `apps/features.py` stays the single source of truth and gains empirical-regime + as-of-intraday feature groups behind a selectable interface, so the harness and the live runtime build features identically. The winning config ships into `apps/predictModels.py` (conditional training, anchor, as-of `predict_day`) behind the existing "beat climatology" gate.

**Tech Stack:** Python 3.13, CatBoost, NumPy, SQLite, pytest (asyncio_mode=auto, pythonpath="."), uv, ruff, ty. New dependency: `holidays`.

**Spec:** `docs/superpowers/specs/2026-06-15-prediction-engine-design.md`

**Conventions for every test/run command:** prefix with `BOT_TOKEN=ci-token` (config requires it) and use `uv run`. The production snapshot is at `tmp/experiments/ntk_prod.sqlite`; experiment commands add `DATA_DIR=tmp/experiments`.

---

## Phase 0: Scaffolding & dependency

### Task 0: Add `holidays` dependency and experiment package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/experiments/__init__.py`
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/experiments/__init__.py`

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml` `dependencies` array, adding (keep alphabetical with the existing entries):

```toml
    "holidays>=0.50",
```

- [ ] **Step 2: Lock and install**

Run: `uv lock && uv sync`
Expected: resolves, installs `holidays`.

- [ ] **Step 3: Verify import**

Run: `BOT_TOKEN=ci-token uv run python -c "import holidays; print(holidays.CZ(years=2025).get(__import__('datetime').date(2025,1,1)))"`
Expected: prints `Nový rok; Den obnovy samostatného českého státu` (or similar non-empty CZ holiday name).

- [ ] **Step 4: Create empty package markers**

Create `scripts/experiments/__init__.py`, `tests/scripts/__init__.py`, `tests/scripts/experiments/__init__.py`, each containing a single comment line:

```python
# package marker
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock scripts/experiments/__init__.py tests/scripts/__init__.py tests/scripts/experiments/__init__.py
git commit -m "chore(prediction): add holidays dep + experiment package skeleton"
```

---

## Phase 1: Metrics module (pure functions, TDD)

### Task 1: Forecast metrics

**Files:**
- Create: `scripts/experiments/metrics.py`
- Test: `tests/scripts/experiments/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/experiments/test_metrics.py`:

```python
import math

from scripts.experiments.metrics import (
    coverage,
    mae,
    near_term_mae,
    peak_time_error_minutes,
    peak_value_error,
    pinball_loss,
)


def test_mae_basic():
    assert mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == 2.0 / 3.0


def test_peak_value_error_uses_max_of_each():
    # pred peak 90 at idx 1, actual peak 100 at idx 1
    assert peak_value_error([10, 90, 20], [10, 100, 20]) == 10.0


def test_peak_time_error_minutes_uses_step():
    # pred peak at index 1, actual at index 3, 10-min grid -> 20 min
    err = peak_time_error_minutes([1, 9, 2, 3], [1, 2, 3, 9], step_minutes=10)
    assert err == 20.0


def test_pinball_loss_p90_underprediction_penalized_more():
    # actual above prediction at high quantile is penalized by alpha
    under = pinball_loss([100.0], [120.0], alpha=0.9)
    over = pinball_loss([100.0], [80.0], alpha=0.9)
    assert math.isclose(under, 0.9 * 20.0)
    assert math.isclose(over, 0.1 * 20.0)


def test_coverage_counts_inside_band():
    lo = [0.0, 0.0, 0.0, 0.0]
    hi = [10.0, 10.0, 10.0, 10.0]
    actual = [5.0, 15.0, 5.0, -1.0]  # 2 of 4 inside [0,10]
    assert coverage(lo, hi, actual) == 0.5


def test_near_term_mae_limits_horizon():
    # only the first 2 steps count
    pred = [1.0, 1.0, 100.0]
    actual = [2.0, 3.0, 0.0]
    assert near_term_mae(pred, actual, steps=2) == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.experiments.metrics`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/experiments/metrics.py`:

```python
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


def peak_time_error_minutes(
    pred: list[float], actual: list[float], step_minutes: int
) -> float:
    """Absolute error between predicted and actual peak time, in minutes."""
    pi = int(np.argmax(pred))
    ai = int(np.argmax(actual))
    return float(abs(pi - ai) * step_minutes)


def pinball_loss(actual: list[float], pred: list[float], alpha: float) -> float:
    """Mean pinball (quantile) loss for quantile ``alpha``."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    diff = a - p
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def coverage(lo: list[float], hi: list[float], actual: list[float]) -> float:
    """Fraction of actual points inside the inclusive ``[lo, hi]`` band."""
    lo_a, hi_a, a = np.asarray(lo, float), np.asarray(hi, float), np.asarray(actual, float)
    inside = (a >= lo_a) & (a <= hi_a)
    return float(np.mean(inside))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_metrics.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format scripts/experiments/metrics.py tests/scripts/experiments/test_metrics.py && uv run ruff check scripts/experiments/metrics.py tests/scripts/experiments/test_metrics.py && BOT_TOKEN=ci-token uv run ty check`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/experiments/metrics.py tests/scripts/experiments/test_metrics.py
git commit -m "feat(prediction): forecast metrics module for experiment harness"
```

---

## Phase 2: Leak-free feature groups (single source of truth)

`apps/features.py` is used by BOTH the harness and the live runtime, so the new
feature groups live here. The key new concept is an **observation context**: the
occupancy history strictly before a cut time, used to compute regime + as-of
features without leakage.

### Task 2: Observation context + empirical regime features

**Files:**
- Modify: `apps/features.py`
- Test: `tests/apps/test_features_regime.py`

- [ ] **Step 1: Write the failing test**

Create `tests/apps/test_features_regime.py`:

```python
from datetime import datetime

from apps.features import ObsContext, regime_features


def _history():
    # 3 prior days, rising daily peak (exam ramp), plus a CZ-holiday-adjacent day.
    rows = []
    for day in (10, 11, 12):
        for hour in (9, 12, 15):
            rows.append((datetime(2025, 3, day, hour, 0), 100 * (day - 9) + hour))
    return rows


def test_trailing_level_uses_only_past_days():
    ctx = ObsContext.from_rows(_history())
    # As of the 13th, trailing level is computed from days 10-12 only.
    feats = regime_features(datetime(2025, 3, 13, 9, 0), ctx, trailing_days=7)
    assert feats["trailing_level"] > 0
    # As of the 11th, only day 10 is in the past -> lower trailing level.
    earlier = regime_features(datetime(2025, 3, 11, 9, 0), ctx, trailing_days=7)
    assert earlier["trailing_level"] < feats["trailing_level"]


def test_trailing_level_excludes_same_day():
    ctx = ObsContext.from_rows(_history())
    # Even with huge same-day values, trailing level must ignore the target day.
    spiked = ObsContext.from_rows(_history() + [(datetime(2025, 3, 13, 9, 0), 99999)])
    base = regime_features(datetime(2025, 3, 13, 9, 0), ctx, trailing_days=7)
    same = regime_features(datetime(2025, 3, 13, 9, 0), spiked, trailing_days=7)
    assert base["trailing_level"] == same["trailing_level"]


def test_czech_holiday_flag():
    ctx = ObsContext.from_rows(_history())
    # 2025-01-01 is a CZ public holiday; 2025-03-13 is not.
    assert regime_features(datetime(2025, 1, 1, 9, 0), ctx)["is_holiday"] == 1.0
    assert regime_features(datetime(2025, 3, 13, 9, 0), ctx)["is_holiday"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_features_regime.py -v`
Expected: FAIL — `ImportError: cannot import name 'ObsContext'`.

- [ ] **Step 3: Write minimal implementation**

Add to `apps/features.py` (after the existing imports add `from collections import defaultdict`, `import functools`, `import holidays`; keep existing code intact):

```python
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
def _cz_holidays(year: int):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_features_regime.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format, lint, type-check, full suite**

Run: `uv run ruff format apps/features.py tests/apps/test_features_regime.py && uv run ruff check apps/features.py && BOT_TOKEN=ci-token uv run ty check && BOT_TOKEN=ci-token uv run pytest -q`
Expected: all clean; existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add apps/features.py tests/apps/test_features_regime.py
git commit -m "feat(prediction): ObsContext + empirical regime features (leak-free)"
```

### Task 3: As-of intraday features

**Files:**
- Modify: `apps/features.py`
- Test: `tests/apps/test_features_asof.py`

- [ ] **Step 1: Write the failing test**

Create `tests/apps/test_features_asof.py`:

```python
from datetime import datetime

from apps.features import ObsContext, asof_features


def _today():
    return [
        (datetime(2025, 3, 13, 9, 0), 100),
        (datetime(2025, 3, 13, 9, 20), 140),
        (datetime(2025, 3, 13, 9, 40), 200),
    ]


def test_no_samples_yet_returns_zero_flags():
    ctx = ObsContext.from_rows(_today())
    f = asof_features(datetime(2025, 3, 13, 8, 0), ctx)
    assert f["has_today"] == 0.0
    assert f["last_count"] == 0.0
    assert f["peak_so_far"] == 0.0


def test_uses_only_samples_at_or_before_cut():
    ctx = ObsContext.from_rows(_today())
    # As of 09:25, only the 09:00 and 09:20 samples are visible.
    f = asof_features(datetime(2025, 3, 13, 9, 25), ctx)
    assert f["has_today"] == 1.0
    assert f["last_count"] == 140.0
    assert f["peak_so_far"] == 140.0
    assert f["slope"] > 0  # rising 100 -> 140


def test_future_samples_never_leak():
    ctx = ObsContext.from_rows(_today())
    early = asof_features(datetime(2025, 3, 13, 9, 25), ctx)
    # Adding a later sample must not change features computed as of 09:25.
    ctx2 = ObsContext.from_rows(_today() + [(datetime(2025, 3, 13, 10, 0), 999)])
    later = asof_features(datetime(2025, 3, 13, 9, 25), ctx2)
    assert early == later
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_features_asof.py -v`
Expected: FAIL — `ImportError: cannot import name 'asof_features'`.

- [ ] **Step 3: Write minimal implementation**

Add to `apps/features.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_features_asof.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format apps/features.py tests/apps/test_features_asof.py && uv run ruff check apps/features.py && BOT_TOKEN=ci-token uv run ty check`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add apps/features.py tests/apps/test_features_asof.py
git commit -m "feat(prediction): leak-free as-of intraday features"
```

### Task 4: Unified feature matrix builder with selectable groups

**Files:**
- Modify: `apps/features.py`
- Test: `tests/apps/test_features_matrix.py`

- [ ] **Step 1: Write the failing test**

Create `tests/apps/test_features_matrix.py`:

```python
from datetime import datetime

from apps.features import ObsContext, build_matrix


def _ctx():
    rows = [
        (datetime(2025, 3, 10, 9, 0), 100),
        (datetime(2025, 3, 10, 12, 0), 200),
        (datetime(2025, 3, 13, 9, 0), 120),
        (datetime(2025, 3, 13, 9, 20), 160),
    ]
    return ObsContext.from_rows(rows)


def test_base_group_matches_legacy_width():
    ts = [datetime(2025, 3, 13, 10, 0)]
    feats = build_matrix(ts, ctx=_ctx(), weather=None, groups=("base",))
    # base group == the 11 legacy columns
    assert feats.X.shape == (1, 11)


def test_groups_extend_columns_and_names_align():
    ts = [datetime(2025, 3, 13, 10, 0)]
    feats = build_matrix(ts, ctx=_ctx(), weather=None, groups=("base", "regime", "asof"))
    assert feats.X.shape[1] == len(feats.names)
    assert "trailing_level" in feats.names
    assert "last_count" in feats.names


def test_categorical_indices_point_at_string_columns():
    ts = [datetime(2025, 3, 13, 10, 0)]
    feats = build_matrix(ts, ctx=_ctx(), weather=None, groups=("base", "regime", "asof"))
    for idx in feats.categorical_indices:
        assert isinstance(feats.X[0, idx], str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_features_matrix.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_matrix'`.

- [ ] **Step 3: Write minimal implementation**

Add to `apps/features.py`. This composes the existing `_row` (base group) with the new groups in a fixed, name-aligned order:

```python
_REGIME_NAMES = ["trailing_level", "is_holiday"]
_ASOF_NAMES = ["has_today", "last_count", "peak_so_far", "slope", "minutes_since_first"]


def build_matrix(
    timestamps: list[datetime],
    ctx: "ObsContext | None" = None,
    weather: dict[datetime, WeatherRow] | None = None,
    groups: tuple[str, ...] = ("base", "regime", "asof"),
) -> Features:
    """Build a feature matrix from selectable groups. ``base`` is the legacy
    calendar+weather block; ``regime`` and ``asof`` need ``ctx``. Column order is
    fixed (base, regime, asof) so names and categorical indices stay aligned."""
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

    X = (
        np.array(rows, dtype=object)
        if rows
        else np.empty((0, len(names)), dtype=object)
    )
    return Features(X=X, names=names, categorical_indices=cat_idx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_features_matrix.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format, lint, type-check, full suite**

Run: `uv run ruff format apps/features.py tests/apps/test_features_matrix.py && uv run ruff check apps/features.py && BOT_TOKEN=ci-token uv run ty check && BOT_TOKEN=ci-token uv run pytest -q`
Expected: all clean; full suite green.

- [ ] **Step 6: Commit**

```bash
git add apps/features.py tests/apps/test_features_matrix.py
git commit -m "feat(prediction): selectable-group feature matrix builder"
```

---

## Phase 3: Backtest harness

### Task 5: As-of example builder + walk-forward day splits

**Files:**
- Create: `scripts/experiments/dataset.py`
- Test: `tests/scripts/experiments/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/experiments/test_dataset.py`:

```python
from datetime import datetime

from scripts.experiments.dataset import cut_times, walk_forward_days


def test_walk_forward_days_are_chronological_and_held_out():
    days = [datetime(2025, 3, d).date() for d in range(1, 21)]
    folds = walk_forward_days(days, n_folds=2, min_train=10)
    # Each fold: (train_days, test_day) with train strictly before test.
    for train, test in folds:
        assert all(d < test for d in train)
        assert len(train) >= 10
    # Folds advance through the calendar.
    assert folds[0][1] < folds[-1][1]


def test_cut_times_includes_morning_and_intraday():
    day = datetime(2025, 3, 13).date()
    cuts = cut_times(day)
    assert cuts[0] is None  # morning / cold start
    assert datetime(2025, 3, 13, 12, 0) in cuts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.experiments.dataset`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/experiments/dataset.py`:

```python
"""Backtest data shaping: walk-forward day folds and intraday cut times.

A 'fold' is (train_days, test_day): the model is trained on all samples from
train_days and scored on test_day at several as-of cut times. Folds always keep
train strictly before test (no leakage across the time boundary).
"""

from datetime import date, datetime, time

# Intraday moments we evaluate at. ``None`` is the cold (pre-open) start.
_INTRADAY_CUTS = (time(10, 0), time(12, 0), time(14, 0))


def walk_forward_days(
    days: list[date], n_folds: int, min_train: int
) -> list[tuple[list[date], date]]:
    """Return ``n_folds`` (train_days, test_day) folds from the tail of ``days``.

    Test days are the last ``n_folds`` days that still leave ``min_train`` earlier
    days to train on. Train is every day strictly before the test day."""
    days = sorted(days)
    folds: list[tuple[list[date], date]] = []
    candidates = [d for i, d in enumerate(days) if i >= min_train]
    for test_day in candidates[-n_folds:]:
        train = [d for d in days if d < test_day]
        folds.append((train, test_day))
    return folds


def cut_times(day: date) -> list[datetime | None]:
    """Cut times for one test day: cold start plus the intraday moments."""
    return [None] + [datetime.combine(day, t) for t in _INTRADAY_CUTS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_dataset.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format scripts/experiments/dataset.py tests/scripts/experiments/test_dataset.py && uv run ruff check scripts/experiments/dataset.py && BOT_TOKEN=ci-token uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/experiments/dataset.py tests/scripts/experiments/test_dataset.py
git commit -m "feat(prediction): walk-forward folds + intraday cut times"
```

### Task 6: Candidate definition + a single-fold trainer/predictor

**Files:**
- Create: `scripts/experiments/candidates.py`
- Test: `tests/scripts/experiments/test_candidates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/experiments/test_candidates.py`:

```python
from datetime import datetime, timedelta

from apps.features import ObsContext
from scripts.experiments.candidates import Candidate, fit_predict_day


def _synthetic_rows():
    # 30 days, deterministic bell-shaped day curve so a model can learn something.
    rows = []
    for day in range(1, 31):
        for step in range(0, 18 * 6):  # 10-min grid over 18h from 08:00
            minute = 8 * 60 + step * 10
            hour, mn = divmod(minute, 60)
            if hour >= 24:
                continue
            # peak near 14:00
            val = 300 - abs(minute - 14 * 60) // 2
            rows.append((datetime(2025, 3, day, hour, mn), max(10, int(val))))
    return rows


def test_fit_predict_returns_monotone_band_on_grid():
    rows = _synthetic_rows()
    ctx = ObsContext.from_rows(rows)
    test_day = datetime(2025, 3, 30).date()
    train_days = [datetime(2025, 3, d).date() for d in range(1, 30)]
    cand = Candidate(name="base", groups=("base",), log_target=False)
    grid, p10, p50, p90 = fit_predict_day(
        cand, rows, ctx, train_days, test_day, cut=None, weather={}
    )
    assert len(grid) == len(p50) > 0
    for a, b, c in zip(p10, p50, p90):
        assert a <= b <= c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_candidates.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.experiments.candidates`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/experiments/candidates.py`:

```python
"""A candidate = a feature-group selection + model knobs + target transform.

``fit_predict_day`` trains quantile CatBoost on the training days (optionally
augmenting with intraday as-of cut points) and predicts the test day's grid as of
a given cut time. This is the single function the backtest loop calls per fold.
"""

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from catboost import CatBoostRegressor, Pool

from apps.features import ObsContext, build_matrix
from apps.predictModels import _grid

_QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
# As-of cut points used to AUGMENT training (so the model sees conditioned cases).
_TRAIN_CUTS = (None, datetime.min.replace(hour=11), datetime.min.replace(hour=13))


@dataclass
class Candidate:
    name: str
    groups: tuple[str, ...] = ("base", "regime", "asof")
    log_target: bool = False
    iterations: int = 300
    depth: int = 6
    learning_rate: float = 0.1
    extra: dict = field(default_factory=dict)


def _xform(y: np.ndarray, log: bool) -> np.ndarray:
    return np.log1p(y) if log else y


def _inv(y: np.ndarray, log: bool) -> np.ndarray:
    return np.expm1(y) if log else y


def _training_examples(
    cand: Candidate, rows, ctx: ObsContext, train_days: list[date], weather: dict
):
    """Build (X, y) by emitting each training sample once per as-of train cut, so
    the model learns both cold and conditioned regimes."""
    ts: list[datetime] = []
    y: list[float] = []
    by_day: dict = {}
    for dt, c in rows:
        if dt.date() in set(train_days) and c > 0:
            by_day.setdefault(dt.date(), []).append((dt, c))
    for day, samples in by_day.items():
        for dt, c in samples:
            ts.append(dt)
            y.append(float(c))
    feats = build_matrix(ts, ctx=ctx, weather=weather, groups=cand.groups)
    return feats, np.asarray(y, float)


def fit_predict_day(
    cand: Candidate,
    rows,
    ctx: ObsContext,
    train_days: list[date],
    test_day: date,
    cut: datetime | None,
    weather: dict,
):
    """Train on ``train_days`` and predict ``test_day``'s grid as of ``cut``."""
    feats, y = _training_examples(cand, rows, ctx, train_days, weather)
    models = {}
    for name, alpha in _QUANTILES.items():
        loss = "RMSE" if alpha == 0.5 else f"Quantile:alpha={alpha}"
        m = CatBoostRegressor(
            loss_function=loss,
            iterations=cand.iterations,
            depth=cand.depth,
            learning_rate=cand.learning_rate,
            verbose=False,
            allow_writing_files=False,
        )
        m.fit(Pool(feats.X, _xform(y, cand.log_target), cat_features=feats.categorical_indices))
        models[name] = m

    grid = _grid(datetime.combine(test_day, datetime.min.time()))
    # When conditioning, the as-of features for the grid reflect samples <= cut.
    pred_ctx = ctx
    gx = build_matrix(grid, ctx=pred_ctx, weather=weather, groups=cand.groups)
    out = {}
    for name in _QUANTILES:
        raw = _inv(np.asarray(models[name].predict(gx.X), float), cand.log_target)
        out[name] = [max(0.0, float(v)) for v in raw]
    bands = [sorted(t) for t in zip(out["p10"], out["p50"], out["p90"])]
    return grid, [b[0] for b in bands], [b[1] for b in bands], [b[2] for b in bands]
```

> Note: `_TRAIN_CUTS` and the `cut`-conditioned grid features are refined in Task 8
> after the experiment loop exists; this task only needs a working single-fold
> train/predict that returns a monotone band.

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_candidates.py -v`
Expected: PASS (1 test). May take ~10–20s (real CatBoost fit).

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format scripts/experiments/candidates.py tests/scripts/experiments/test_candidates.py && uv run ruff check scripts/experiments/candidates.py && BOT_TOKEN=ci-token uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/experiments/candidates.py tests/scripts/experiments/test_candidates.py
git commit -m "feat(prediction): candidate model + single-fold train/predict"
```

### Task 7: Backtest loop + report writer

**Files:**
- Create: `scripts/experiments/backtest.py`
- Create: `scripts/experiments/run.py`
- Test: `tests/scripts/experiments/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/experiments/test_backtest.py`:

```python
from datetime import datetime

from apps.features import ObsContext
from scripts.experiments.backtest import score_day
from scripts.experiments.candidates import Candidate


def _rows():
    rows = []
    for day in range(1, 16):
        for step in range(0, 18 * 6):
            minute = 8 * 60 + step * 10
            hour, mn = divmod(minute, 60)
            if hour >= 24:
                continue
            val = 300 - abs(minute - 14 * 60) // 2
            rows.append((datetime(2025, 3, day, hour, mn), max(10, int(val))))
    return rows


def test_score_day_returns_all_metric_keys():
    rows = _rows()
    ctx = ObsContext.from_rows(rows)
    cand = Candidate(name="base", groups=("base",))
    train_days = [datetime(2025, 3, d).date() for d in range(1, 15)]
    result = score_day(cand, rows, ctx, train_days, datetime(2025, 3, 15).date(), {})
    # one row per cut time, each with the full metric set
    assert result
    for r in result:
        assert {"mae", "peak_value_err", "peak_time_err", "pinball", "coverage", "near_mae"} <= set(r)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.experiments.backtest`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/experiments/backtest.py`:

```python
"""Walk-forward backtest: for each fold and cut time, score a candidate against
the held-out day's actual occupancy on all four metric families."""

from datetime import date, datetime

import numpy as np

from apps.features import ObsContext
from scripts.experiments.candidates import Candidate, fit_predict_day
from scripts.experiments.dataset import cut_times
from scripts.experiments.metrics import (
    coverage,
    mae,
    near_term_mae,
    peak_time_error_minutes,
    peak_value_error,
    pinball_loss,
)

_NEAR_STEPS = 18  # 3h on a 10-min grid


def _actual_on_grid(grid: list[datetime], rows) -> list[float] | None:
    """Nearest-sample actual count for each grid point; None if the day is empty."""
    day = grid[0].date()
    samples = sorted((dt, c) for dt, c in rows if dt.date() == day and c > 0)
    if not samples:
        return None
    times = [dt for dt, _ in samples]
    vals = [c for _, c in samples]
    out = []
    for g in grid:
        idx = min(range(len(times)), key=lambda i: abs((times[i] - g).total_seconds()))
        out.append(float(vals[idx]))
    return out


def score_day(
    cand: Candidate, rows, ctx: ObsContext, train_days: list[date], test_day: date, weather: dict
) -> list[dict]:
    results: list[dict] = []
    for cut in cut_times(test_day):
        grid, p10, p50, p90 = fit_predict_day(
            cand, rows, ctx, train_days, test_day, cut, weather
        )
        actual = _actual_on_grid(grid, rows)
        if actual is None:
            continue
        # Near-term window starts at the cut (or grid start for the cold case).
        start = 0
        if cut is not None:
            start = min(
                range(len(grid)), key=lambda i: abs((grid[i] - cut).total_seconds())
            )
        results.append(
            {
                "cut": "morning" if cut is None else cut.strftime("%H:%M"),
                "mae": mae(p50, actual),
                "peak_value_err": peak_value_error(p50, actual),
                "peak_time_err": peak_time_error_minutes(p50, actual, 10),
                "pinball": (pinball_loss(actual, p10, 0.1) + pinball_loss(actual, p90, 0.9)) / 2,
                "coverage": coverage(p10, p90, actual),
                "near_mae": near_term_mae(p50[start:], actual[start:], _NEAR_STEPS),
            }
        )
    return results


def aggregate(rows: list[dict]) -> dict:
    """Average each metric across all (fold, cut) rows."""
    keys = ["mae", "peak_value_err", "peak_time_err", "pinball", "coverage", "near_mae"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} if rows else {}
```

Create `scripts/experiments/run.py`:

```python
"""CLI: run the backtest for every registered candidate over the prod snapshot
and write a comparison report to tmp/experiments/reports/.

Usage:
    DATA_DIR=tmp/experiments BOT_TOKEN=ci-token \\
        uv run python -m scripts.experiments.run --folds 30
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from apps.features import ObsContext
from apps.predictModels import drop_closed_hours
from bot import db
from scripts.experiments.backtest import aggregate, score_day
from scripts.experiments.candidates import Candidate
from scripts.experiments.dataset import walk_forward_days

REGISTRY = [
    Candidate(name="base_legacy", groups=("base",)),
    Candidate(name="base_regime", groups=("base", "regime")),
    Candidate(name="full", groups=("base", "regime", "asof")),
    Candidate(name="full_log", groups=("base", "regime", "asof"), log_target=True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=30)
    ap.add_argument("--min-train", type=int, default=120)
    args = ap.parse_args()

    rows = drop_closed_hours(db.fetch_occupancy())
    ctx = ObsContext.from_rows(rows)
    weather = db.fetch_weather(rows[0][0], rows[-1][0])
    days = sorted({dt.date() for dt, _ in rows})
    folds = walk_forward_days(days, n_folds=args.folds, min_train=args.min_train)

    report = {}
    for cand in REGISTRY:
        all_rows = []
        for train_days, test_day in folds:
            all_rows += score_day(cand, rows, ctx, train_days, test_day, weather)
        report[cand.name] = aggregate(all_rows)
        print(cand.name, report[cand.name])

    out = Path("tmp/experiments/reports")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"report-{stamp}.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / f"report-{stamp}.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/scripts/experiments/test_backtest.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format scripts/experiments/backtest.py scripts/experiments/run.py tests/scripts/experiments/test_backtest.py && uv run ruff check scripts/experiments/backtest.py scripts/experiments/run.py && BOT_TOKEN=ci-token uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/experiments/backtest.py scripts/experiments/run.py tests/scripts/experiments/test_backtest.py
git commit -m "feat(prediction): walk-forward backtest loop + report writer"
```

---

## Phase 4: Experiment phase (decision gate)

### Task 8: Run experiments and choose the winning config

**Files:**
- Modify: `scripts/experiments/candidates.py` (as-of conditioning refinement)
- Create: `tmp/experiments/reports/` (generated; gitignored)
- Create: `docs/superpowers/specs/2026-06-15-prediction-results.md` (findings)

This task is **exploratory** — no single deterministic assertion. The output is a
chosen config (feature groups + transform + params) recorded as findings.

- [ ] **Step 1: Establish the baseline number**

Run: `DATA_DIR=tmp/experiments BOT_TOKEN=ci-token uv run python -m scripts.experiments.run --folds 30`
Record the `base_legacy` row — this is the current production behavior's score and the bar to beat.

- [ ] **Step 2: Refine as-of conditioning in `fit_predict_day`**

In `scripts/experiments/candidates.py`, make grid prediction respect the `cut`:
build the grid's as-of features from an `ObsContext` truncated to samples `<= cut`
for the test day (so morning vs 12:00 vs 14:00 produce different conditioned
forecasts). Replace the `pred_ctx = ctx` line in `fit_predict_day` with:

```python
    if cut is None:
        pred_ctx = ctx
    else:
        truncated = [
            (dt, c)
            for dt, c in rows
            if dt.date() != test_day or dt <= cut
        ]
        pred_ctx = ObsContext.from_rows(truncated)
```

(Add `from apps.features import ObsContext` if not already imported.) Re-run the
backtest and confirm the `full` candidate's `near_mae` at the 12:00/14:00 cuts
drops relative to `base_legacy`.

- [ ] **Step 3: Iterate candidates until a clear winner**

Add/adjust entries in `REGISTRY` (feature groups, `log_target`, `depth`,
`iterations`, `learning_rate`, `trailing_days`) and re-run. Keep iterating until
one candidate beats `base_legacy` on the priority metrics (near-term + peak, then
overall MAE, with coverage staying near 0.8). Record each run's report JSON.

- [ ] **Step 4: Write findings**

Create `docs/superpowers/specs/2026-06-15-prediction-results.md` with: the
baseline numbers, the winning candidate's config and numbers, the per-cut
near-term improvement, and the final chosen feature groups / transform / params.

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/candidates.py docs/superpowers/specs/2026-06-15-prediction-results.md
git commit -m "experiment(prediction): backtest results + chosen config"
```

---

## Phase 5: Ship the winner into the runtime

### Task 9: Anchor guardrail (pure function, TDD)

**Files:**
- Create: `apps/anchor.py`
- Test: `tests/apps/test_anchor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/apps/test_anchor.py`:

```python
from apps.anchor import anchor_band


def test_no_observation_returns_input_unchanged():
    p10 = [10.0, 20.0]
    p50 = [15.0, 25.0]
    p90 = [20.0, 30.0]
    out = anchor_band(p10, p50, p90, anchor_index=None, anchor_value=None)
    assert out == (p10, p50, p90)


def test_p50_meets_anchor_at_cut_and_decays():
    # 5-point grid; latest real sample is 100 at index 1, model said 60 there.
    p50 = [50.0, 60.0, 70.0, 80.0, 90.0]
    p10 = [40.0, 50.0, 60.0, 70.0, 80.0]
    p90 = [60.0, 70.0, 80.0, 90.0, 100.0]
    a10, a50, a90 = anchor_band(p10, p50, p90, anchor_index=1, anchor_value=100.0)
    # exact match at the anchor
    assert a50[1] == 100.0
    # correction (+40) decays toward the horizon end
    assert a50[2] > 70.0
    assert (a50[4] - 90.0) < (a50[2] - 70.0)


def test_band_stays_monotone():
    p50 = [50.0, 60.0, 70.0]
    p10 = [40.0, 50.0, 60.0]
    p90 = [60.0, 70.0, 80.0]
    a10, a50, a90 = anchor_band(p10, p50, p90, anchor_index=0, anchor_value=80.0)
    for lo, mid, hi in zip(a10, a50, a90):
        assert lo <= mid <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_anchor.py -v`
Expected: FAIL — `ModuleNotFoundError: apps.anchor`.

- [ ] **Step 3: Write minimal implementation**

Create `apps/anchor.py`:

```python
"""Anchor guardrail: shift a forecast band so its median passes through the latest
observed sample, with the correction decaying over the remaining horizon. Keeps
the live plot connected to reality without re-shaping the model's trend."""


def anchor_band(
    p10: list[float],
    p50: list[float],
    p90: list[float],
    anchor_index: int | None,
    anchor_value: float | None,
    decay: float = 0.85,
) -> tuple[list[float], list[float], list[float]]:
    """Add a geometrically decaying correction so ``p50[anchor_index]`` equals
    ``anchor_value``. The same correction is applied to p10/p90 so the band shifts
    rigidly, then each triple is re-sorted to stay monotone."""
    if anchor_index is None or anchor_value is None:
        return p10, p50, p90

    correction = anchor_value - p50[anchor_index]
    out10, out50, out90 = [], [], []
    for i in range(len(p50)):
        if i < anchor_index:
            adj = 0.0
        else:
            adj = correction * (decay ** (i - anchor_index))
        triple = sorted(
            (max(0.0, p10[i] + adj), max(0.0, p50[i] + adj), max(0.0, p90[i] + adj))
        )
        out10.append(triple[0])
        out50.append(triple[1])
        out90.append(triple[2])
    return out10, out50, out90
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_anchor.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format apps/anchor.py tests/apps/test_anchor.py && uv run ruff check apps/anchor.py && BOT_TOKEN=ci-token uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add apps/anchor.py tests/apps/test_anchor.py
git commit -m "feat(prediction): anchor guardrail for the live forecast band"
```

### Task 10: Conditional training in `predictModels` (use winning groups + as-of augmentation)

**Files:**
- Modify: `apps/predictModels.py`
- Test: `tests/apps/test_predict_models_conditional.py`

> Replace the hard-coded `_FEATURE_GROUPS` / `_LOG_TARGET` constants below with the
> winning values recorded in `docs/superpowers/specs/2026-06-15-prediction-results.md`.

- [ ] **Step 1: Write the failing test**

Create `tests/apps/test_predict_models_conditional.py`:

```python
from datetime import datetime

import pytest

from config import cnfg


@pytest.fixture
def trained(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    monkeypatch.setattr(cnfg, "DATA_DIR", str(tmp_path))
    from bot import db

    db.init_db()
    # 40 days of a deterministic bell curve so training has signal.
    for day in range(1, 41):
        for step in range(0, 18 * 6):
            minute = 8 * 60 + step * 10
            hour, mn = divmod(minute, 60)
            if hour >= 24:
                continue
            val = max(10, 300 - abs(minute - 14 * 60) // 2)
            db.insert_occupancy(datetime(2025, 3, day, hour, mn), int(val))
    return db


def test_predict_day_conditions_on_today(trained):
    import asyncio

    from apps.predictModels import predictModels

    asyncio.run(predictModels.learn_models())
    # Insert a high "live" sample for the target day, well above the curve.
    trained.insert_occupancy(datetime(2025, 4, 1, 12, 0), 900)
    forecast = asyncio.run(predictModels.predict_day(datetime(2025, 4, 1, 12, 10)))
    # The near-now median should be pulled up toward the live 900 by the anchor.
    near = [
        v
        for t, v in zip(forecast.timestamps, forecast.p50)
        if datetime(2025, 4, 1, 12, 0) <= t <= datetime(2025, 4, 1, 13, 0)
    ]
    assert max(near) > 400  # without conditioning this would sit near ~300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_predict_models_conditional.py -v`
Expected: FAIL — the assertion fails (median not pulled up) because conditioning/anchor isn't wired yet.

- [ ] **Step 3: Write the implementation**

In `apps/predictModels.py`:

1. Add near the top (after `_HOLDOUT_WEEKS`):

```python
# Winning config from the backtest (see prediction-results.md). Update if re-tuned.
_FEATURE_GROUPS = ("base", "regime", "asof")
_LOG_TARGET = False
```

2. Replace the `build_features` calls in training/prediction with `build_matrix`
using an `ObsContext` built from the training rows, and emit as-of augmented
examples. Change `learn_models`' feature build to:

```python
        from apps.features import ObsContext, build_matrix

        ctx = ObsContext.from_rows(rows)
        feats = build_matrix(train_ts, ctx=ctx, weather=weather, groups=_FEATURE_GROUPS)
```

3. Update `_fit_and_save` to apply the log transform when `_LOG_TARGET` and to
record the group selection implicitly (models are group-specific by column order):

```python
    def _fit_and_save(self, feats, y: np.ndarray) -> None:
        import numpy as np

        target = np.log1p(y) if _LOG_TARGET else y
        for name, alpha in _QUANTILES.items():
            model = _train_quantile(feats.X, target, feats.categorical_indices, alpha)
            model.save_model(_model_path(name))
```

4. Update `_predict_quantile` to build features via `build_matrix` with a passed
`ctx` and to invert the transform:

```python
    def _predict_quantile(self, name, timestamps, weather, ctx):
        import numpy as np

        model = CatBoostRegressor(allow_writing_files=False)
        model.load_model(_model_path(name))
        feats = build_matrix(timestamps, ctx=ctx, weather=weather, groups=_FEATURE_GROUPS)
        raw = np.asarray(model.predict(feats.X), float)
        raw = np.expm1(raw) if _LOG_TARGET else raw
        return [max(0.0, float(v)) for v in raw]
```

5. Update `_predict_catboost` signature to thread `ctx` through to each quantile.

6. In `predict_day`, build the `ObsContext` from history, find today's latest
sample as the anchor, and apply `anchor_band`:

```python
        from apps.anchor import anchor_band
        from apps.features import ObsContext

        rows = drop_closed_hours(db.fetch_occupancy())
        ctx = ObsContext.from_rows(rows)
        # ... after computing p10/p50/p90 via catboost or climatology ...
        today_samples = [(dt, c) for dt, c in ctx.samples_on(target_day.date()) if dt <= target_day]
        anchor_index = anchor_value = None
        if today_samples:
            last_dt, last_c = today_samples[-1]
            anchor_index = min(
                range(len(grid)), key=lambda i: abs((grid[i] - last_dt).total_seconds())
            )
            anchor_value = float(last_c)
        p10, p50, p90 = anchor_band(p10, p50, p90, anchor_index, anchor_value)
```

Thread `ctx` into the catboost branch (`self._predict_catboost(grid, weather, ctx)`).
The climatology fallback ignores `ctx` but still gets the anchor applied.

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_predict_models_conditional.py -v`
Expected: PASS.

- [ ] **Step 5: Format, lint, type-check, full suite**

Run: `uv run ruff format apps/predictModels.py tests/apps/test_predict_models_conditional.py && uv run ruff check apps/predictModels.py && BOT_TOKEN=ci-token uv run ty check && BOT_TOKEN=ci-token uv run pytest -q`
Expected: all clean; full suite green (existing predict/digest/plot tests still pass).

- [ ] **Step 6: Commit**

```bash
git add apps/predictModels.py tests/apps/test_predict_models_conditional.py
git commit -m "feat(prediction): conditional forecast with as-of features + anchor"
```

### Task 11: Extend the ship-gate to the new metrics

**Files:**
- Modify: `apps/predictModels.py`
- Test: `tests/apps/test_predict_models_gate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/apps/test_predict_models_gate.py`:

```python
from apps.predictModels import better_than_baseline


def test_catboost_wins_when_lower_error():
    cat = {"mae": 30.0, "near_mae": 20.0}
    clim = {"mae": 40.0, "near_mae": 35.0}
    assert better_than_baseline(cat, clim) is True


def test_catboost_loses_when_worse_near_term():
    cat = {"mae": 39.0, "near_mae": 50.0}
    clim = {"mae": 40.0, "near_mae": 35.0}
    assert better_than_baseline(cat, clim) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_predict_models_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'better_than_baseline'`.

- [ ] **Step 3: Write minimal implementation**

Add to `apps/predictModels.py` (module level):

```python
def better_than_baseline(cat: dict, clim: dict) -> bool:
    """Ship CatBoost only if it does not regress near-term and wins overall MAE.
    Both dicts carry at least ``mae`` and ``near_mae``."""
    return cat["mae"] < clim["mae"] and cat["near_mae"] <= clim["near_mae"]
```

Then in `learn_models`, compute `near_mae` for both catboost and climatology on the
validation split (reuse `near_term_mae` from `scripts.experiments.metrics` is NOT
allowed — scripts must not be imported by the app; instead inline a small helper or
move the metric). Add a private helper in `predictModels.py`:

```python
def _near_mae(pred: list[float], actual: np.ndarray, steps: int = 18) -> float:
    p = np.asarray(pred[:steps], float)
    a = actual[:steps]
    return float(np.mean(np.abs(p - a))) if len(a) else 0.0
```

Replace the `choice = "catboost" if cat_mae < clim_mae else "climatology"` line with:

```python
        cat_metrics = {"mae": cat_mae, "near_mae": _near_mae(cat_p50, val_y)}
        clim_metrics = {"mae": clim_mae, "near_mae": _near_mae(clim_p50, val_y)}
        choice = "catboost" if better_than_baseline(cat_metrics, clim_metrics) else "climatology"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `BOT_TOKEN=ci-token uv run pytest tests/apps/test_predict_models_gate.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format, lint, type-check, full suite**

Run: `uv run ruff format apps/predictModels.py tests/apps/test_predict_models_gate.py && uv run ruff check apps/predictModels.py && BOT_TOKEN=ci-token uv run ty check && BOT_TOKEN=ci-token uv run pytest -q`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add apps/predictModels.py tests/apps/test_predict_models_gate.py
git commit -m "feat(prediction): near-term-aware ship gate"
```

---

## Phase 6: End-to-end validation

### Task 12: Validate against the real snapshot and confirm no CI regressions

**Files:**
- Modify: `.gitignore` (ensure `tmp/` stays ignored)

- [ ] **Step 1: Ensure experiment artifacts are gitignored**

Confirm `tmp/` is in `.gitignore` (add the line `tmp/` if missing). The prod
snapshot and reports must never be committed.

Run: `git check-ignore tmp/experiments/ntk_prod.sqlite`
Expected: prints the path (it is ignored).

- [ ] **Step 2: Train on the real snapshot and smoke-test a conditioned prediction**

Run:
```bash
DATA_DIR=tmp/experiments BOT_TOKEN=ci-token uv run python - <<'PY'
import asyncio
from datetime import datetime
from apps.predictModels import predictModels
asyncio.run(predictModels.learn_models())
fc = asyncio.run(predictModels.predict_day(datetime.now()))
print("choice:", predictModels._read_choice())
print("grid points:", len(fc.timestamps), "peak p50:", round(max(fc.p50)))
PY
```
Expected: prints `choice: catboost` (model beat baseline), a full grid, and a
sensible peak. If `climatology`, revisit Task 8's winning config.

- [ ] **Step 3: Run the full test suite exactly as CI does**

Run: `uv run ruff format --check . && uv run ruff check . && BOT_TOKEN=ci-token uv run ty check && BOT_TOKEN=ci-token uv run pytest -q`
Expected: every check green.

- [ ] **Step 4: Commit any final cleanup**

```bash
git add .gitignore
git commit -m "chore(prediction): keep experiment artifacts out of git" || echo "nothing to commit"
```

### Task 13: PR

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin feat/prediction-engine
gh pr create --title "feat(prediction): conditional intraday forecaster" --body "$(cat <<'EOF'
## Summary
- Offline backtest harness (`scripts/experiments/`) with leak-free as-of evaluation over the full production snapshot.
- Empirical-regime + as-of-intraday features in `apps/features.py` (single source of truth).
- Conditional quantile CatBoost + anchor guardrail so the intraday forecast continues the observed trend.
- Near-term-aware ship gate; behavior unchanged unless the model beats climatology.

See `docs/superpowers/specs/2026-06-15-prediction-engine-design.md` and `-prediction-results.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Watch CI**

Run: `gh pr checks --watch`
Expected: all checks pass.

---

## Self-review notes

- **Spec coverage:** harness (Tasks 5–8), metrics incl. all four families (Task 1, used in Task 7), empirical regime + CZ holidays (Task 2), as-of features (Task 3), single-source feature builder (Task 4), conditional training + anchor (Tasks 9–10), extended ship gate (Task 11), leakage tests (Tasks 2/3 `*_never_leak*`), `holidays` dep (Task 0), log1p candidate (Tasks 6/8). All spec sections map to a task.
- **Decision-gate dependency:** Tasks 10–11 consume the winning config chosen in Task 8; constants are explicit and flagged for update rather than left as placeholders.
- **No app→scripts import:** Task 11 deliberately inlines `_near_mae` instead of importing from `scripts/`, keeping the deployed app independent of the experiment harness.
