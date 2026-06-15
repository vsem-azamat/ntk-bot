# Prediction Engine Improvement — Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation plan pending

## Problem

The NTK occupancy forecaster has two shortcomings:

1. **Accuracy.** `predict_day()` produces a static day curve from calendar +
   weather features only (time-of-day sin/cos, weekday, month, day-of-year, and
   four weather variables). CatBoost quantile models (p10/p50/p90) are retrained
   daily and shipped only if they beat a per-(weekday × 10-min) climatology on a
   4-week holdout MAE. Accuracy is "not good enough" per the user.

2. **Intraday disconnect.** The forecast never reads today's observed samples.
   The digest/plot draws the real data line *and* a static prediction band side
   by side, but the band is identical all day — it cannot anchor to the latest
   real count or "continue the current trend." This is by design today and is the
   primary user complaint.

## Goals & success criteria

Optimize **all four** of these, with near-term and peak being the most
user-visible (the digest caption headlines a live count and "expected peak ~N
around HH:MM"):

1. **Overall curve** — whole-day MAE (the existing benchmark).
2. **Peak value & time** — peak magnitude error and peak-time error (minutes).
3. **Band calibration** — pinball loss + empirical p10–p90 coverage (~80%).
4. **Live / near-term** — next 1–3h MAE given what's happened so far today.

A candidate ships only if it beats the climatology baseline on the primary
metrics — the same ship-gate philosophy as today, measured on more than one
number.

## Dataset

Full production snapshot pulled from the azamat VPS `ntk-bot_ntk_data` volume to
`tmp/experiments/ntk_prod.sqlite`:

- ~100k non-zero occupancy samples across **981 days** (2023-09-24 → 2026-06-15),
  cadence ~20 min (~54 samples/day).
- Occupancy 1–1352, median 283, p95 985 — heavy right tail (exam-season spikes).
- Hourly weather covers the full range (23,904 rows), no gaps.

## Decisions

- **Intraday conditioning:** conditional CatBoost with leak-free as-of features
  **plus** an anchor guardrail (chosen over anchor-only and level-scaling).
- **Academic calendar:** derived **empirically** — trailing regime level +
  Czech public holidays from a library — no manually maintained calendar.

## Architecture

```
prod DB snapshot (tmp/experiments/ntk_prod.sqlite)
        │
        ▼
[experiment harness]  walk-forward, leak-free as-of backtest
   candidates × metrics → comparison report
        │  pick winner (must beat climatology baseline)
        ▼
[features.py]  single source of truth — calendar + weather
               + empirical regime + as-of intraday features
        │
        ▼
[predictModels.py]  conditional quantile CatBoost
        │  daily retrain (existing maintenance loop)
        ▼
[predict_day]  past = real data, future = forecast conditioned
               on today-so-far, anchored to latest sample
        ▼
   digest / plot (unchanged consumers)
```

Nothing in the deployed bot changes behavior until a candidate wins the backtest
on the four metrics and beats the climatology baseline.

## Components

### 1. Experiment harness (`scripts/experiments/`)

- Loads the snapshot; runs a **walk-forward backtest**: for each held-out day,
  stand at cut times `T ∈ {morning, 10:00, 12:00, 14:00}`, build features from
  data strictly `≤ T`, forecast the remaining day, score.
- **Metrics module** computing all four targets per cut time: overall MAE,
  peak-value error, peak-time error (minutes), pinball loss + empirical p10–p90
  coverage, next-1–3h MAE.
- **Candidate registry**: each candidate = feature set + model params + target
  transform. Runs all, writes a comparison table (+ diagnostic plots) to
  `tmp/experiments/reports/`. Reproducible; **not part of the deployed bot**.

### 2. Feature upgrades (`apps/features.py` remains the single source of truth)

Three new leak-free groups, all computed strictly from data `≤` cut time:

- **Empirical regime** ("academic calendar without a calendar"): trailing
  occupancy level over the last N days, Czech public-holiday flag (`holidays`
  lib) — captures exam-spike vs break regimes.
- **As-of intraday**: last observed count, recent slope, peak-so-far, minutes
  since open, observed-vs-climatology ratio. Present only once the day has
  samples — this is what lets the curve continue the trend.
- A strict **cut-time contract** so the harness and the live runtime build
  features identically (no train/serve skew).

### 3. Conditional training + anchor (`apps/predictModels.py`)

- Training augments each day with **several as-of cut points** so the model
  learns both the cold morning case and the conditioned mid-day cases. Still
  p10/p50/p90 quantiles, time-based holdout, benchmark vs climatology extended to
  the four metrics.
- **Anchor guardrail** on prediction: p50 connects to the latest real sample, the
  correction decays over the horizon, band tightens near "now" and widens later.
- `predict_day` gains an as-of path: morning with no data → today's cold behavior
  (unchanged); during the day → conditional + anchored. `DayForecast` shape is
  unchanged, so digest/plot consume it as-is.

## Data flow

DB snapshot → harness backtest → pick config → encode in
`features.py`/`predictModels.py` → daily retrain in prod (existing
`daily_maintenance_loop`) → `predict_day` conditions on live data → digest/plot.

## Testing

- **Leakage test** (#1 risk): assert as-of features never touch future data.
- Unit tests for anchor-correction math and each metric function.
- Backtest reproducibility test on a tiny fixture.
- Existing tests stay green; cold-path forecast unchanged.
- The harness produces the empirical "is it better" evidence.

## Risks & mitigations

- **Leakage** with as-of features — strict cut-time contract + dedicated test.
- **Heavy exam-season tail** — `log1p` target transform is a candidate to test.
- **Overfitting augmented examples** — time-based holdout, limited cut points.
- **Weekend/holiday start-time** handling in the grid.

## Dependencies

- Add `holidays` (Czech public holidays).

## Out of scope

- Manual academic-calendar maintenance.
- Changes to digest scheduling, plot styling, or data collection cadence.
