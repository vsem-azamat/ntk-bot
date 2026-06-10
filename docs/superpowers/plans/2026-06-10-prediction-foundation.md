# Prediction Foundation + CatBoost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the string-based, time-leaking, 740 MB-model occupancy predictor with a structured, time-validated, lightweight CatBoost forecaster (with quantile bands), benchmarked against a climatology baseline, fed by historical + forecast weather.

**Architecture:** SQLite gains structured occupancy access and an hourly `weather` table backfilled from open-meteo. A single `apps/features.py` builder turns timestamps (+ weather) into a CatBoost-ready matrix, shared by training and prediction so there is no train/serve skew. `apps/predictModels.py` trains a median + p10/p90 CatBoost trio on a time-based holdout, compares it to a climatology baseline, and exposes `predict_day()`. The plot layer consumes `predict_day()` only.

**Tech Stack:** Python 3.11, SQLite, `catboost`, NumPy, aiohttp (open-meteo), pytest / pytest-asyncio. Package manager: `uv`. Lint: `ruff`. Types: `ty`.

---

## File Structure

- `bot/db.py` (modify) — add `fetch_occupancy()`, `weather` table + `upsert_weather` / `fetch_weather` / `max_weather_ts`.
- `apps/weather_history.py` (create) — open-meteo archive backfill + forecast fetch, shared weather-variable definition.
- `apps/features.py` (create) — single feature builder (`build_features`, `Features`).
- `apps/baseline.py` (create) — climatology baseline (`build_climatology`, `climatology_predict`).
- `apps/predictModels.py` (rewrite) — CatBoost training, time-based validation, ship rule, `predict_day`, `DayForecast`.
- `apps/schedule_functions.py` (modify) — daily `weather_backfill_loop`.
- `bot/bot.py` (modify) — launch the backfill loop at startup.
- `apps/plot_functions.py` (modify) — consume `predict_day` instead of loading models / parsing strings.
- `pyproject.toml` (modify) — add `catboost` dependency.
- Tests under `tests/apps/` mirroring each module.

Notes for the implementer (codebase conventions):
- Run everything with `uv run` (e.g. `uv run pytest`, `uv run ruff check`).
- `pytest` config: `asyncio_mode = "auto"` — `async def test_*` needs no decorator.
- Timestamps in SQLite use the format `"%Y-%m-%d %H:%M"` (see `bot/db.py` `_TS_FORMAT`).
- `cnfg.DATA_DIR` is the directory for all on-disk artifacts; tests `monkeypatch` it to a tmp dir.
- Keep lines ≤ 100 chars where practical (`E501` is ignored, but match the style).

---

## Task 1: Add `catboost` dependency

**Files:**
- Modify: `pyproject.toml` (the `dependencies` array)

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"catboost>=1.2"` to the `dependencies` list (after `"joblib>=1.3"`).

- [ ] **Step 2: Sync the lockfile and environment**

Run: `uv sync`
Expected: resolves and installs `catboost` (and its deps) with no error.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import catboost; print(catboost.__version__)"`
Expected: prints a version like `1.2.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add catboost dependency"
```

---

## Task 2: Structured occupancy accessor in `bot/db.py`

**Files:**
- Modify: `bot/db.py`
- Test: `tests/bot/test_db_occupancy.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/bot/test_db_occupancy.py
from datetime import datetime

from config import cnfg


def _use_tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ntk.sqlite"
    monkeypatch.setattr(cnfg, "DB_PATH", str(db_path))


def test_fetch_occupancy_returns_structured_rows_oldest_first(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    from bot import db

    db.init_db()
    db.insert_occupancy(datetime(2024, 3, 1, 10, 0), 100)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 0), 50)

    rows = db.fetch_occupancy()

    assert rows == [
        (datetime(2024, 3, 1, 9, 0), 50),
        (datetime(2024, 3, 1, 10, 0), 100),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/bot/test_db_occupancy.py -v`
Expected: FAIL — `AttributeError: module 'bot.db' has no attribute 'fetch_occupancy'`.

- [ ] **Step 3: Implement `fetch_occupancy`**

Add to `bot/db.py` (after `iter_rows`):

```python
def fetch_occupancy() -> list[tuple[datetime, int]]:
    """Return all occupancy samples as ``(datetime, people)`` tuples, oldest first."""
    with closing(_connect()) as conn, conn:
        rows = conn.execute("SELECT ts, people FROM occupancy ORDER BY ts").fetchall()
    return [(datetime.strptime(ts, _TS_FORMAT), people) for ts, people in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/bot/test_db_occupancy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/db.py tests/bot/test_db_occupancy.py
git commit -m "feat(db): structured fetch_occupancy accessor"
```

---

## Task 3: Weather table + accessors in `bot/db.py`

**Files:**
- Modify: `bot/db.py`
- Test: `tests/bot/test_db_weather.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/bot/test_db_weather.py
from datetime import datetime

from config import cnfg


def _use_tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))


def test_weather_upsert_fetch_and_max_ts(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    from bot import db

    db.init_db()
    assert db.max_weather_ts() is None

    db.upsert_weather([(datetime(2024, 3, 1, 9, 0), 5.0, 0.0, 50.0, 3.0)])
    db.upsert_weather([(datetime(2024, 3, 1, 9, 0), 6.0, 0.1, 60.0, 4.0)])  # overwrite
    db.upsert_weather([(datetime(2024, 3, 1, 10, 0), 7.0, None, None, 5.0)])

    assert db.max_weather_ts() == datetime(2024, 3, 1, 10, 0)

    got = db.fetch_weather(datetime(2024, 3, 1, 9, 0), datetime(2024, 3, 1, 9, 0))
    assert got == {datetime(2024, 3, 1, 9, 0): (6.0, 0.1, 60.0, 4.0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/bot/test_db_weather.py -v`
Expected: FAIL — weather table / functions do not exist.

- [ ] **Step 3: Implement the weather schema and accessors**

In `bot/db.py`, extend `init_db()` to also create the table, and add the accessors:

```python
def init_db() -> None:
    """Create the data directory and tables if they do not exist."""
    Path(cnfg.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS occupancy ( ts TEXT PRIMARY KEY, people INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS weather ("
            " ts TEXT PRIMARY KEY, temp REAL, precip REAL, cloud REAL, wind REAL)"
        )


def upsert_weather(
    rows: list[tuple[datetime, float | None, float | None, float | None, float | None]],
) -> None:
    """Insert/overwrite hourly weather rows keyed by their floored-to-hour timestamp."""
    with closing(_connect()) as conn, conn:
        conn.executemany(
            "INSERT INTO weather (ts, temp, precip, cloud, wind) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(ts) DO UPDATE SET"
            " temp=excluded.temp, precip=excluded.precip,"
            " cloud=excluded.cloud, wind=excluded.wind",
            [
                (ts.replace(minute=0, second=0, microsecond=0).strftime(_TS_FORMAT), t, p, c, w)
                for ts, t, p, c, w in rows
            ],
        )


def fetch_weather(
    start: datetime, end: datetime
) -> dict[datetime, tuple[float | None, float | None, float | None, float | None]]:
    """Return ``{hour_datetime: (temp, precip, cloud, wind)}`` within ``[start, end]``."""
    with closing(_connect()) as conn, conn:
        rows = conn.execute(
            "SELECT ts, temp, precip, cloud, wind FROM weather WHERE ts BETWEEN ? AND ?",
            (start.strftime(_TS_FORMAT), end.strftime(_TS_FORMAT)),
        ).fetchall()
    return {datetime.strptime(ts, _TS_FORMAT): (t, p, c, w) for ts, t, p, c, w in rows}


def max_weather_ts() -> datetime | None:
    """Return the latest weather timestamp, or ``None`` if the table is empty."""
    with closing(_connect()) as conn, conn:
        row = conn.execute("SELECT MAX(ts) FROM weather").fetchone()
    return datetime.strptime(row[0], _TS_FORMAT) if row and row[0] else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/bot/test_db_weather.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/db.py tests/bot/test_db_weather.py
git commit -m "feat(db): hourly weather table and accessors"
```

---

## Task 4: Feature builder `apps/features.py`

**Files:**
- Create: `apps/features.py`
- Test: `tests/apps/test_features.py` (create)

The matrix is a NumPy **object** array so categorical columns (weekday, month)
can be strings as CatBoost requires, while numeric columns hold floats / `NaN`.
Column order is fixed:

```
0 minute_sin   1 minute_cos   2 is_weekend
3 yday_sin      4 yday_cos
5 weekday(cat)  6 month(cat)
7 temp          8 precip        9 cloud      10 wind
```

- [ ] **Step 1: Write the failing test**

```python
# tests/apps/test_features.py
import math
from datetime import datetime

import numpy as np

from apps.features import FEATURE_NAMES, CATEGORICAL_INDICES, build_features


def test_feature_names_and_categorical_indices():
    assert FEATURE_NAMES == [
        "minute_sin", "minute_cos", "is_weekend",
        "yday_sin", "yday_cos", "weekday", "month",
        "temp", "precip", "cloud", "wind",
    ]
    assert CATEGORICAL_INDICES == [5, 6]


def test_cyclical_minute_encoding_wraps_around():
    feats = build_features([datetime(2024, 3, 1, 0, 0), datetime(2024, 3, 1, 23, 59)])
    x = feats.X
    # 00:00 and 23:59 are one minute apart -> close in sin/cos space
    assert abs(float(x[0, 0]) - float(x[1, 0])) < 0.01  # minute_sin
    assert abs(float(x[0, 1]) - float(x[1, 1])) < 0.01  # minute_cos


def test_categoricals_are_strings_and_weekend_flag():
    # 2024-03-02 is a Saturday
    feats = build_features([datetime(2024, 3, 2, 12, 0)])
    x = feats.X
    assert x[0, 5] == "5"      # weekday: Saturday == 5
    assert x[0, 6] == "3"      # month: March
    assert float(x[0, 2]) == 1.0  # is_weekend


def test_weather_join_fills_matching_hour_and_nans_when_missing():
    ts = datetime(2024, 3, 1, 9, 30)
    weather = {datetime(2024, 3, 1, 9, 0): (5.0, 0.0, 50.0, 3.0)}
    x = build_features([ts], weather).X
    assert [float(v) for v in x[0, 7:11]] == [5.0, 0.0, 50.0, 3.0]

    x_missing = build_features([ts], {}).X
    assert all(math.isnan(float(v)) for v in x_missing[0, 7:11])
    assert isinstance(x, np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/apps/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'apps.features'`.

- [ ] **Step 3: Implement the feature builder**

```python
# apps/features.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/apps/test_features.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/features.py tests/apps/test_features.py
git commit -m "feat(model): shared feature builder with cyclical + weather features"
```

---

## Task 5: Climatology baseline `apps/baseline.py`

**Files:**
- Create: `apps/baseline.py`
- Test: `tests/apps/test_baseline.py` (create)

The baseline groups historical counts by `(weekday, 10-minute time bucket)` and
returns p10/p50/p90 per bucket. It is both the benchmark CatBoost must beat and
the cold-start / fallback predictor.

- [ ] **Step 1: Write the failing test**

```python
# tests/apps/test_baseline.py
from datetime import datetime

from apps.baseline import build_climatology, climatology_predict


def _rows():
    # Three Fridays, all 09:00, counts 100/200/300 -> p50 == 200
    return [
        (datetime(2024, 3, 1, 9, 0), 100),
        (datetime(2024, 3, 8, 9, 0), 200),
        (datetime(2024, 3, 15, 9, 0), 300),
    ]


def test_climatology_predicts_bucket_median():
    clim = build_climatology(_rows())
    p10, p50, p90 = climatology_predict(clim, [datetime(2024, 3, 22, 9, 0)])  # also a Friday
    assert p50[0] == 200
    assert p10[0] <= p50[0] <= p90[0]


def test_climatology_unknown_bucket_is_non_negative_not_crashing():
    clim = build_climatology(_rows())
    p10, p50, p90 = climatology_predict(clim, [datetime(2024, 3, 23, 14, 0)])  # unseen
    assert p10[0] >= 0 and p50[0] >= 0 and p90[0] >= 0
    assert p10[0] <= p50[0] <= p90[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/apps/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'apps.baseline'`.

- [ ] **Step 3: Implement the baseline**

```python
# apps/baseline.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/apps/test_baseline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/baseline.py tests/apps/test_baseline.py
git commit -m "feat(model): climatology baseline predictor"
```

---

## Task 6: Weather ingest `apps/weather_history.py`

**Files:**
- Create: `apps/weather_history.py`
- Test: `tests/apps/test_weather_history.py` (create)

This module owns the **single definition** of which weather variables the model
uses, so training (archive) and prediction (forecast) stay consistent. Network
calls are wrapped in tiny seams (`_get_json`) that tests monkeypatch.

- [ ] **Step 1: Write the failing test**

```python
# tests/apps/test_weather_history.py
from datetime import datetime

import apps.weather_history as wh


def test_parse_hourly_payload_to_rows():
    payload = {
        "hourly": {
            "time": ["2024-03-01T09:00", "2024-03-01T10:00"],
            "temperature_2m": [5.0, 6.0],
            "precipitation": [0.0, 0.2],
            "cloudcover": [50, 60],
            "windspeed_10m": [3.0, 4.0],
        }
    }
    rows = wh._payload_to_rows(payload)
    assert rows == [
        (datetime(2024, 3, 1, 9, 0), 5.0, 0.0, 50.0, 3.0),
        (datetime(2024, 3, 1, 10, 0), 6.0, 0.2, 60.0, 4.0),
    ]


async def test_backfill_upserts_from_last_known_ts(tmp_path, monkeypatch):
    from config import cnfg

    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    from bot import db

    db.init_db()
    db.insert_occupancy(datetime(2024, 3, 1, 9, 0), 100)  # gives backfill a start date

    async def fake_get_json(url, params):
        return {
            "hourly": {
                "time": ["2024-03-01T09:00"],
                "temperature_2m": [5.0],
                "precipitation": [0.0],
                "cloudcover": [50],
                "windspeed_10m": [3.0],
            }
        }

    monkeypatch.setattr(wh, "_get_json", fake_get_json)

    written = await wh.backfill()

    assert written == 1
    assert db.fetch_weather(datetime(2024, 3, 1, 9, 0), datetime(2024, 3, 1, 9, 0)) == {
        datetime(2024, 3, 1, 9, 0): (5.0, 0.0, 50.0, 3.0)
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/apps/test_weather_history.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the ingest module**

```python
# apps/weather_history.py
"""Open-meteo weather ingest. Owns the single definition of the model's weather
variables so archive (training) and forecast (prediction) stay consistent.
"""

import logging
from datetime import datetime, timedelta

import aiohttp

from apps.features import WeatherRow

logger = logging.getLogger(__name__)

# NTK coordinates (matches apps/weather_api.py).
_LAT = 50.1038
_LON = 14.3906
_TZ = "Europe/Berlin"

# The four variables the model consumes, in the order features.py expects them.
_HOURLY = "temperature_2m,precipitation,cloudcover,windspeed_10m"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


async def _get_json(url: str, params: dict) -> dict:
    """Network seam (monkeypatched in tests)."""
    async with aiohttp.ClientSession() as session, session.get(url, params=params) as resp:
        return await resp.json()


def _payload_to_rows(payload: dict) -> list[tuple[datetime, float, float, float, float]]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    return [
        (
            datetime.strptime(times[i], "%Y-%m-%dT%H:%M"),
            float(hourly["temperature_2m"][i]),
            float(hourly["precipitation"][i]),
            float(hourly["cloudcover"][i]),
            float(hourly["windspeed_10m"][i]),
        )
        for i in range(len(times))
    ]


def _start_date() -> datetime | None:
    """Earliest hour we still need: hour after the last stored weather, else the
    first occupancy date. Returns None when there is no occupancy at all."""
    from bot import db

    last = db.max_weather_ts()
    if last is not None:
        return last + timedelta(hours=1)
    occupancy = db.fetch_occupancy()
    return occupancy[0][0].replace(minute=0, second=0, microsecond=0) if occupancy else None


async def backfill() -> int:
    """Fetch archive weather from the last-known hour to today and upsert it.

    Idempotent and incremental. Returns the number of rows written."""
    from bot import db

    start = _start_date()
    if start is None:
        return 0
    end = datetime.now()
    if start.date() > end.date():
        return 0

    payload = await _get_json(
        _ARCHIVE_URL,
        {
            "latitude": _LAT,
            "longitude": _LON,
            "timezone": _TZ,
            "hourly": _HOURLY,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
        },
    )
    rows = _payload_to_rows(payload)
    if rows:
        db.upsert_weather(rows)
    return len(rows)


async def forecast_weather(target_day: datetime) -> dict[datetime, WeatherRow]:
    """Return ``{hour: (temp, precip, cloud, wind)}`` forecast for ``target_day``."""
    payload = await _get_json(
        _FORECAST_URL,
        {
            "latitude": _LAT,
            "longitude": _LON,
            "timezone": _TZ,
            "hourly": _HOURLY,
            "forecast_days": 2,
        },
    )
    return {dt: (t, p, c, w) for dt, t, p, c, w in _payload_to_rows(payload)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/apps/test_weather_history.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/weather_history.py tests/apps/test_weather_history.py
git commit -m "feat(weather): open-meteo archive backfill + forecast ingest"
```

---

## Task 7: Rewrite `apps/predictModels.py` — CatBoost, validation, ship rule, `predict_day`

**Files:**
- Rewrite: `apps/predictModels.py`
- Modify: `tests/apps/test_predict_models.py` (replace stale tests)
- Test: `tests/apps/test_predict_day.py` (create)

`predict_day` always returns a sane forecast: trained CatBoost models if they
exist and won the ship rule, otherwise the climatology baseline. The grid matches
the current plot (08:00 weekdays / 10:00 weekends, 18 h, 10-min steps).

- [ ] **Step 1: Replace the stale unit tests**

Overwrite `tests/apps/test_predict_models.py` with:

```python
# tests/apps/test_predict_models.py
from datetime import datetime, timedelta

from apps.predictModels import drop_closed_hours, time_split


def test_drop_closed_hours_removes_zero_counts():
    rows = [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 9, 20), 0),   # closed -> dropped
        (datetime(2024, 3, 1, 10, 10), 7),
    ]
    assert drop_closed_hours(rows) == [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 10, 10), 7),
    ]


def test_time_split_has_no_leakage():
    base = datetime(2024, 1, 1, 9, 0)
    rows = [(base + timedelta(days=i), i) for i in range(60)]
    train, val = time_split(rows, holdout_weeks=4)

    assert train and val
    assert max(dt for dt, _ in train) < min(dt for dt, _ in val)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/apps/test_predict_models.py -v`
Expected: FAIL — `ImportError` (`drop_closed_hours` / `time_split` not defined).

- [ ] **Step 3: Write the rewritten module**

Replace the entire contents of `apps/predictModels.py` with:

```python
# apps/predictModels.py
"""Occupancy forecaster: CatBoost median + p10/p90 quantiles, time-validated and
benchmarked against a climatology baseline. ``predict_day`` is the public API the
plot layer consumes; it serves CatBoost only when it beat the baseline, otherwise
the baseline itself.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

from apps.baseline import build_climatology, climatology_predict
from apps.collect_time import generate_datetime_list
from apps.features import build_features
from config import cnfg

logger = logging.getLogger(__name__)

_QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
_CHOICE_FILE = "model_choice"
_HOLDOUT_WEEKS = 4


@dataclass
class DayForecast:
    """Full-day forecast grid with a guaranteed-monotone band."""

    timestamps: list[datetime]
    p10: list[float]
    p50: list[float]
    p90: list[float]


def _model_path(name: str) -> str:
    return str(Path(cnfg.DATA_DIR) / f"model_{name}.cbm")


def _choice_path() -> str:
    return str(Path(cnfg.DATA_DIR) / _CHOICE_FILE)


def drop_closed_hours(rows: list[tuple[datetime, int]]) -> list[tuple[datetime, int]]:
    """Drop samples with a zero count (library closed / no reading)."""
    return [(dt, c) for dt, c in rows if c != 0]


def time_split(
    rows: list[tuple[datetime, int]], holdout_weeks: int = _HOLDOUT_WEEKS
) -> tuple[list[tuple[datetime, int]], list[tuple[datetime, int]]]:
    """Split chronologically: last ``holdout_weeks`` as validation, rest as train."""
    cutoff = rows[-1][0] - timedelta(weeks=holdout_weeks)
    train = [r for r in rows if r[0] < cutoff]
    val = [r for r in rows if r[0] >= cutoff]
    return train, val


def _grid(target_day: datetime) -> list[datetime]:
    """Plot-matching grid: 08:00 (10:00 on weekends) for 18 h at 10-min steps."""
    start = target_day.replace(
        hour=10 if target_day.isoweekday() >= 6 else 8,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = start + timedelta(hours=18)
    # generate_datetime_list is async; build the grid inline to keep this sync.
    out: list[datetime] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(minutes=10)
    return out


def _train_quantile(X: np.ndarray, y: np.ndarray, cat_idx: list[int], alpha: float):
    loss = "RMSE" if alpha == 0.5 else f"Quantile:alpha={alpha}"
    model = CatBoostRegressor(
        loss_function=loss,
        iterations=300,
        depth=6,
        learning_rate=0.1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(X, y, cat_features=cat_idx))
    return model


def _mae(pred: list[float], actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(pred) - actual)))


class PredictModels:
    async def learn_models(self) -> None:
        """Train CatBoost on a time-based holdout, benchmark vs climatology, and
        persist both the models and the ship decision."""
        from bot import db

        rows = drop_closed_hours(db.fetch_occupancy())
        if len(rows) <= 50:
            logger.info("Not enough data to train (%d rows); baseline only", len(rows))
            self._write_choice("climatology")
            return

        train, val = time_split(rows)
        if not train or not val:
            self._write_choice("climatology")
            return

        weather = db.fetch_weather(rows[0][0], rows[-1][0])

        train_ts = [dt for dt, _ in train]
        train_y = np.array([c for _, c in train], dtype=float)
        feats = build_features(train_ts, weather)

        await asyncio.to_thread(self._fit_and_save, feats, train_y)

        # Benchmark on the holdout.
        val_ts = [dt for dt, _ in val]
        val_y = np.array([c for _, c in val], dtype=float)

        cat_p50 = self._predict_quantile("p50", val_ts, weather)
        clim = build_climatology(train)
        _, clim_p50, _ = climatology_predict(clim, val_ts)

        cat_mae = _mae(cat_p50, val_y)
        clim_mae = _mae(clim_p50, val_y)
        choice = "catboost" if cat_mae < clim_mae else "climatology"
        logger.info("Holdout MAE — catboost=%.2f climatology=%.2f -> %s", cat_mae, clim_mae, choice)
        self._write_choice(choice)

    def _fit_and_save(self, feats, y: np.ndarray) -> None:
        for name, alpha in _QUANTILES.items():
            model = _train_quantile(feats.X, y, feats.categorical_indices, alpha)
            model.save_model(_model_path(name))

    def _predict_quantile(
        self, name: str, timestamps: list[datetime], weather: dict
    ) -> list[float]:
        model = CatBoostRegressor()
        model.load_model(_model_path(name))
        feats = build_features(timestamps, weather)
        return [max(0.0, float(v)) for v in model.predict(feats.X)]

    def _write_choice(self, choice: str) -> None:
        Path(_choice_path()).write_text(choice, encoding="utf-8")

    def _read_choice(self) -> str:
        path = Path(_choice_path())
        return path.read_text(encoding="utf-8").strip() if path.exists() else "climatology"

    def _catboost_available(self) -> bool:
        return all(Path(_model_path(n)).exists() for n in _QUANTILES)

    async def predict_day(self, target_day: datetime | None = None) -> DayForecast:
        """Public API: forecast the full day. CatBoost if it won, else baseline."""
        from bot import db

        target_day = target_day or datetime.now()
        grid = _grid(target_day)

        use_catboost = self._read_choice() == "catboost" and self._catboost_available()
        if use_catboost:
            from apps.weather_history import forecast_weather

            try:
                weather = await forecast_weather(target_day)
            except Exception:
                logger.exception("Forecast weather fetch failed; predicting without it")
                weather = {}
            p10 = self._predict_quantile("p10", grid, weather)
            p50 = self._predict_quantile("p50", grid, weather)
            p90 = self._predict_quantile("p90", grid, weather)
        else:
            clim = build_climatology(drop_closed_hours(db.fetch_occupancy()))
            p10, p50, p90 = climatology_predict(clim, grid)

        # Guarantee a monotone, non-negative band per point.
        bands = [sorted((max(0.0, a), max(0.0, b), max(0.0, c))) for a, b, c in zip(p10, p50, p90)]
        return DayForecast(
            timestamps=grid,
            p10=[b[0] for b in bands],
            p50=[b[1] for b in bands],
            p90=[b[2] for b in bands],
        )


predictModels = PredictModels()
```

Note: `generate_datetime_list` is intentionally **not** used inside `_grid`
(it is async); the grid is built inline so `_grid` stays synchronous. Remove the
now-unused import if `ruff` flags it.

- [ ] **Step 4: Run the unit tests**

Run: `uv run pytest tests/apps/test_predict_models.py -v`
Expected: PASS.

- [ ] **Step 5: Write the `predict_day` integration test**

```python
# tests/apps/test_predict_day.py
from datetime import datetime, timedelta

from config import cnfg


async def test_predict_day_falls_back_to_baseline_and_returns_monotone_band(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    monkeypatch.setattr(cnfg, "DATA_DIR", str(tmp_path))
    from bot import db

    db.init_db()
    # Seed a few weeks of Mondays-ish data so the baseline has buckets.
    base = datetime(2024, 3, 4, 9, 0)  # Monday
    for w in range(5):
        db.insert_occupancy(base + timedelta(weeks=w), 100 + w)

    from apps.predictModels import predictModels

    # No trained model on disk -> baseline path.
    forecast = await predictModels.predict_day(datetime(2024, 4, 8, 12, 0))  # a Monday

    assert len(forecast.timestamps) == len(forecast.p50) > 0
    for lo, mid, hi in zip(forecast.p10, forecast.p50, forecast.p90):
        assert lo <= mid <= hi
        assert lo >= 0
```

- [ ] **Step 6: Run the integration test**

Run: `uv run pytest tests/apps/test_predict_day.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/predictModels.py tests/apps/test_predict_models.py tests/apps/test_predict_day.py
git commit -m "feat(model): CatBoost forecaster with time validation and baseline ship rule"
```

---

## Task 8: Point the plot at `predict_day`

**Files:**
- Modify: `apps/plot_functions.py`
- Test: `tests/apps/test_plot_uses_predict_day.py` (create)

Replace the per-model spline drawing with a single median line from
`predict_day`. (The full restyle + visible confidence band is Spec 2 — here we
only swap the data source and keep the band data available.)

- [ ] **Step 1: Write the failing test**

```python
# tests/apps/test_plot_uses_predict_day.py
from datetime import datetime

from apps.predictModels import DayForecast


async def test_daily_graph_with_predictions_draws_predicted_median(monkeypatch):
    import apps.plot_functions as pf

    async def fake_get_ntk_data(self, start, end):
        return [start], [123]

    async def fake_predict_day(self, target_day=None):
        ts = [datetime(2024, 3, 1, 8, 0), datetime(2024, 3, 1, 9, 0)]
        return DayForecast(timestamps=ts, p10=[10, 20], p50=[15, 25], p90=[20, 30])

    monkeypatch.setattr(pf.PlotGraphs, "get_ntk_data", fake_get_ntk_data)
    monkeypatch.setattr(pf.predictModels, "predict_day", fake_predict_day.__get__(pf.predictModels))

    fig, ax = await pf.plotGraph.daily_graph_with_predictions(datetime(2024, 3, 1, 12, 0))

    labels = [line.get_label() for line in ax.get_lines()]
    assert "Prediction" in labels
    import matplotlib.pyplot as plt

    plt.close(fig)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/apps/test_plot_uses_predict_day.py -v`
Expected: FAIL — `daily_graph_with_predictions` still uses the old model loop / no `"Prediction"` label.

- [ ] **Step 3: Rewrite the prediction-drawing methods**

In `apps/plot_functions.py`:

1. Update imports — remove the model-loading / spline imports and import the predictor:

```python
from datetime import datetime, timedelta
from pathlib import Path  # remove if no longer used

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from apps.predictModels import parse_row_datetime  # remove if unused after edit
from apps.predictModels import predictModels
```

   After the edit below, run `uv run ruff check --fix apps/plot_functions.py` to
   drop any genuinely unused imports (`numpy`, `scipy.interpolate`, `joblib`,
   `model_filename`, `extract_features`, `MODEL_COLORS`).

2. Replace `get_ntk_data` to use the structured accessor instead of string parsing:

```python
    async def get_ntk_data(
        self, start_datetime: datetime, end_datetime: datetime
    ) -> tuple[list[datetime], list[int]]:
        """Get occupancy data from SQLite within the given datetime range."""
        from bot import db

        datetimes: list[datetime] = []
        quantities: list[int] = []
        for row_datetime, count in db.fetch_occupancy():
            if count != 0 and start_datetime <= row_datetime <= end_datetime:
                datetimes.append(row_datetime)
                quantities.append(count)
        return datetimes, quantities
```

3. Delete `add_daily_prediction` entirely and replace
   `daily_graph_with_predictions` with a version that draws the median from
   `predict_day`:

```python
    async def daily_graph_with_predictions(
        self, target_day: datetime | None = None
    ) -> tuple[Figure, Axes]:
        target_day = target_day or datetime.now()
        fig, ax, _, _ = await self.daily_graph(target_day)

        forecast = await predictModels.predict_day(target_day)
        ax.plot(
            forecast.timestamps,
            forecast.p50,
            linestyle="-",
            color="darkgreen",
            linewidth=1.5,
            label="Prediction",
            zorder=0,
            alpha=0.7,
        )
        ax.legend(loc="upper right")
        return fig, ax
```

   Leave `daily_graph` unchanged. The `forecast.p10` / `forecast.p90` band is
   intentionally carried but not drawn yet — Spec 2 renders it.

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/apps/test_plot_uses_predict_day.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/plot_functions.py tests/apps/test_plot_uses_predict_day.py
git commit -m "refactor(plot): draw prediction from predict_day, drop string parsing"
```

---

## Task 9: Schedule daily weather backfill

**Files:**
- Modify: `apps/schedule_functions.py`
- Modify: `bot/bot.py`
- Test: `tests/apps/test_weather_backfill_loop.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/apps/test_weather_backfill_loop.py
import asyncio

import apps.schedule_functions as sf


async def test_weather_backfill_loop_calls_backfill_then_sleeps(monkeypatch):
    calls = {"backfill": 0, "sleep": 0}

    async def fake_backfill():
        calls["backfill"] += 1
        return 0

    async def fake_sleep(_seconds):
        calls["sleep"] += 1
        raise asyncio.CancelledError  # break out after first cycle

    monkeypatch.setattr(sf, "backfill", fake_backfill)
    monkeypatch.setattr(sf.asyncio, "sleep", fake_sleep)

    try:
        await sf.weather_backfill_loop()
    except asyncio.CancelledError:
        pass

    assert calls["backfill"] == 1
    assert calls["sleep"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/apps/test_weather_backfill_loop.py -v`
Expected: FAIL — `weather_backfill_loop` / `backfill` not present in `schedule_functions`.

- [ ] **Step 3: Add the loop**

In `apps/schedule_functions.py`, add the import and the loop:

```python
from apps.weather_history import backfill
```

```python
async def weather_backfill_loop(interval_seconds: float = 24 * 60 * 60) -> None:
    """Keep the weather table current: backfill on startup, then once per day."""
    while True:
        try:
            written = await backfill()
            logger.info("Weather backfill wrote %d rows", written)
        except Exception:
            logger.exception("Weather backfill failed")
        await asyncio.sleep(interval_seconds)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/apps/test_weather_backfill_loop.py -v`
Expected: PASS.

- [ ] **Step 5: Launch it at startup**

In `bot/bot.py` `on_startup`, alongside the existing `create_task` calls, add:

```python
    asyncio.create_task(weather_backfill_loop())
```

and import it from `apps.schedule_functions` next to the existing imports
(`heartbeat_loop`, `receive_ntk_data`).

- [ ] **Step 6: Verify the bot module imports cleanly**

Run: `uv run python -c "import bot.bot"`
Expected: no error.

- [ ] **Step 7: Commit**

```bash
git add apps/schedule_functions.py bot/bot.py tests/apps/test_weather_backfill_loop.py
git commit -m "feat(weather): daily backfill loop wired into startup"
```

---

## Task 10: Full verification + cleanup

**Files:**
- Modify: `pyproject.toml` only if `ty` overrides need adjusting

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -v`
Expected: all tests PASS (old `extract_features` / `parse_row_datetime` /
`model_filename` tests were replaced in Task 7; confirm none remain referencing
removed symbols — grep below).

- [ ] **Step 2: Confirm no dangling references to removed symbols**

Run: `uv run grep -rn "extract_features\|model_filename\|MODEL_COLORS\|add_daily_prediction\|iter_rows" apps bot tests`
Expected: only legitimate hits — `iter_rows` remains used by `export_text` in
`bot/db.py`. No references to `extract_features`, `model_filename`,
`MODEL_COLORS`, or `add_daily_prediction` outside comments. Fix any stragglers.

- [ ] **Step 3: Lint and type-check**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: clean. If `ty` flags CatBoost (untyped), add an override block in
`pyproject.toml` mirroring the existing matplotlib one:

```toml
[[tool.ty.overrides]]
include = ["apps/predictModels.py"]
rules = { unresolved-import = "ignore" }
```

- [ ] **Step 4: Smoke-test prediction end to end (baseline path)**

Run:

```bash
uv run python -c "
import asyncio
from datetime import datetime
from apps.predictModels import predictModels
f = asyncio.run(predictModels.predict_day(datetime.now()))
print('points', len(f.timestamps), 'p50[0..3]', f.p50[:3])
assert all(a <= b <= c for a, b, c in zip(f.p10, f.p50, f.p90))
print('band monotone OK')
"
```

Expected: prints point count and a monotone-band confirmation (baseline path,
since no model is trained locally).

- [ ] **Step 5: Final commit (if Step 3 changed pyproject)**

```bash
git add pyproject.toml
git commit -m "chore: ty override for catboost untyped import"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Data layer `fetch_occupancy` + weather table/accessors → Tasks 2, 3. ✓
- Weather ingest (archive backfill, forecast, daily refresh) → Tasks 6, 9. ✓
- Single feature builder (cyclical time + weather, NaN handling) → Task 4. ✓
- CatBoost median + p10/p90, time-based holdout, no-leakage assertion → Task 7. ✓
- Climatology baseline as benchmark + fallback, ship rule → Tasks 5, 7. ✓
- `predict_day` API (monotone band, plot grid) → Task 7. ✓
- Plot consumes `predict_day`, no string parsing, old artifacts/paths dropped → Task 8. ✓
- `catboost` dependency + HistGradientBoosting fallback note → Task 1 (fallback documented in spec; swap is localized to Task 7 if needed). ✓
- Tests for determinism, leakage, baseline, band, weather join → Tasks 4–9. ✓

**Deferred deliberately (Spec 2):** confidence-band rendering, restyle, "now" marker.

**Type consistency:** `WeatherRow` defined in `apps/features.py` and reused by
`weather_history.py`; `DayForecast` defined in `predictModels.py` and consumed by
the plot test; `build_features` / `Features.X` / `categorical_indices` names
match across Tasks 4, 7. Weather tuple order `(temp, precip, cloud, wind)` is
identical in db, features, weather_history, and the feature columns 7–10.
