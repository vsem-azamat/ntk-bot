# Prediction foundation + CatBoost — design

Date: 2026-06-10
Status: approved (brainstorming)
Scope: Spec 1 of 2. Spec 2 (graph redesign) is a separate, later doc that
consumes the prediction API defined here.

## Goal

Replace the current prediction pipeline — string-formatted data round-tripping,
time-leaking validation, and a ~740 MB RandomForest artifact — with a
structured, honestly-validated, lightweight forecaster of the **typical
day-ahead occupancy curve**, including quantile bands, benchmarked against a
climatology baseline.

This spec covers the **foundation + model only**. The graph remains visually
untouched here; it is restyled in Spec 2, which plugs into the `predict_day`
API defined below.

## Non-goals

- No graph restyling, confidence-band rendering, or "now" marker (Spec 2).
- No near-term / lag-based forecast. The `/graph` use case is a full
  day-ahead baseline curve, for which future lag values do not exist;
  lag/rolling features are therefore **out of scope** and intentionally not
  built.
- No exam-session / holiday calendar feature (explicitly deferred by the user).

## Context: what is wrong today

- `apps/predictModels.py` and `apps/plot_functions.py` pass data as
  `"YYYY-MM-DD HH:MM - N"` strings, parsing them repeatedly.
- Features (`day_of_year, weekday, minutes_of_day, month`) are raw numerics with
  no cyclical encoding, so 23:59 and 00:00 look "far apart" to the trees.
- `train_test_split(random_state=42)` shuffles a time series → future leaks into
  the past, so reported MSE is meaningless.
- Default-parameter `RandomForestRegressor` memorises the data → ~740 MB pickle
  loaded into RAM on every restart.
- Two models are drawn as two lines with no comparison or ensembling.
- The existing `weather_api.py` is never used as a model input.

## Architecture

```
SQLite ──fetch_occupancy()──┐
                            ├──> features.build_features() ──> CatBoost (train)
SQLite weather ─fetch_weather┘                            └──> predict_day()
        ▲
        │ backfill()/daily update
   open-meteo archive + forecast
```

Single feature-builder is the one source of truth shared by training and
prediction, eliminating train/serve skew.

## Components

### 1. Data layer — `bot/db.py`

- Add `fetch_occupancy() -> list[tuple[datetime, int]]` returning structured
  rows (oldest first). ML and plotting use this; `iter_rows()` / `export_text()`
  stay only for the `/data` export command and are no longer used elsewhere.
- New table:
  `weather(ts TEXT PRIMARY KEY, temp REAL, precip REAL, cloud REAL, wind REAL)`
  — hourly samples (`ts` = `YYYY-MM-DD HH:00`).
- Accessors: `upsert_weather(rows)`, `fetch_weather(start, end) -> dict[ts,row]`,
  `max_weather_ts() -> datetime | None` (for incremental backfill).
- `init_db()` creates the weather table too.

### 2. Weather ingest — `apps/weather_history.py` (new)

- `backfill() -> int`: from `max_weather_ts()` (or first occupancy date if
  empty) to today, call open-meteo **archive** API
  (`https://archive-api.open-meteo.com/v1/archive`) for hourly
  `temperature_2m, precipitation, cloudcover, windspeed_10m` at NTK's
  lat/lon (Europe/Berlin tz), upsert into `weather`. Idempotent and
  incremental. Returns rows written.
- The archive endpoint lags ~5 days behind "now". The recent tail is filled
  from the regular forecast endpoint (already wrapped in `weather_api.py`); if
  still missing, the row is simply absent and features fall back to `NaN`.
- A daily task in `schedule_functions.py` calls `backfill()` to keep weather
  current. Backfill also runs once at startup if the table is empty.
- Prediction-time weather: `get_weather_forecast()` for the target day, mapped
  into the same `{ts: row}` shape `build_features` expects.

### 3. Feature builder — `apps/features.py` (new, single source of truth)

`build_features(timestamps: list[datetime], weather: dict[datetime, WeatherRow]
| None) -> Features` where `Features` carries the matrix `X`, ordered
`feature_names`, and `categorical_indices`.

- Time features:
  - minute-of-day → `sin`, `cos` (cyclical, period = 1440)
  - day-of-week → categorical (0–6)
  - `is_weekend` → 0/1
  - day-of-year → `sin`, `cos` (cyclical, period ≈ 365.25; seasonal)
  - month → categorical (1–12)
- Weather features: `temp, precip, cloud, wind`, joined on the floor-to-hour
  timestamp. Missing → `NaN` (CatBoost-native). When `weather is None` (e.g.
  weather disabled / fallback), the four columns are all `NaN`.
- Deterministic: same input → byte-identical matrix. This is unit-tested.

### 4. Model — `apps/predictModels.py` (rewrite)

- `CatBoostRegressor`:
  - median model — `loss_function="RMSE"` (or `Quantile:alpha=0.5`)
  - two quantile models — `loss_function="Quantile:alpha=0.1"` and `0.9`
  - persisted as compact `.cbm` files in `DATA_DIR` (single-digit MB).
- Training:
  - `fetch_occupancy()` → drop zero rows (closed hours) → `build_features`.
  - **Time-based holdout:** the last N weeks (default 4) form the validation
    set; everything earlier is train. No shuffle. Assert all validation
    timestamps are strictly after all training timestamps.
  - Report MAE and RMSE on the holdout.
- **Climatology baseline** (`apps/baseline.py` or inside predictModels):
  `groupby(weekday, time-bucket)` → median + p10/p90 percentiles. Evaluated on
  the *same* holdout. Both scores are logged.
  - The baseline is also the **fallback predictor** used by `predict_day` when
    no trained CatBoost model exists yet (cold start) or when CatBoost did not
    beat the baseline.
- Ship rule: keep CatBoost only if it beats the baseline MAE on holdout;
  otherwise `predict_day` serves the baseline. The decision is logged at train
  time and recorded (e.g. a small `model_choice` marker in `DATA_DIR`).
- Training runs off the event loop (`asyncio.to_thread`), as today.
- Remove the old `model_RandomForestRegressor.pkl` /
  `model_GradientBoostingRegressor.pkl` artifacts and their load paths.

### 5. Prediction API (consumed by Spec 2)

`predict_day(target_day: datetime | None = None) -> DayForecast` returning
ordered `timestamps`, and arrays `p50`, `p10`, `p90` (band guaranteed
`p10 ≤ p50 ≤ p90`, clamped ≥ 0). The grid matches the current plotting grid
(08:00→02:00, 10-minute steps, weekend start 10:00). The plot module calls only
this — no string parsing, no model loading in the plot layer.

### 6. Dependencies & footprint

- Add `catboost` to `pyproject.toml`.
- Net runtime memory drops massively (≈ few MB model vs 740 MB).
- **Fallback:** if the `catboost` wheel bloats the image unacceptably, swap to
  `HistGradientBoostingRegressor` (already available via sklearn, native
  categoricals + `loss="quantile"`). The feature-builder and `predict_day` API
  are identical either way, so this is a localized swap.

## Testing

- `build_features` determinism and correct cyclical encoding (00:00 ≈ 23:59 in
  sin/cos space).
- Time split has no leakage: every validation timestamp > every training
  timestamp.
- Climatology baseline returns a sane curve and ordered percentiles.
- `predict_day` returns equal-length arrays with a monotone band
  (`p10 ≤ p50 ≤ p90`) and non-negative values.
- Weather join: matching hour fills values, missing hour yields `NaN`.

## Risks

- **Archive lag (~5 days):** handled by forecast-tail fill + `NaN` fallback.
- **CatBoost image weight:** mitigated by the HistGradientBoosting fallback.
- **Cold start / sparse data:** `predict_day` falls back to climatology until a
  CatBoost model that beats baseline exists.
- **Weather backfill volume:** open-meteo archive is chunked by date range;
  backfill is incremental and idempotent, so a partial run resumes cleanly.

## Rollout

1. Data layer + weather table + backfill (no behaviour change yet).
2. Feature builder + climatology baseline + time-based validation.
3. CatBoost training + `predict_day`, wired behind the ship rule.
4. Point `plot_functions.py` at `predict_day` (minimal change; full restyle is
   Spec 2). Remove old model artifacts.
