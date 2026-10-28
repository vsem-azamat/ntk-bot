# How the occupancy forecast works

The bot forecasts the number of people in the library for the rest of the day
on a 10-minute grid from opening time to midnight, as a median (p50) and a
p10–p90 band. It can be asked at any moment: before opening (the morning
digest) or in the middle of the day, when part of the day is already known.

Code: `apps/reference.py` → `apps/level.py` → `apps/features.py` →
`apps/forecaster.py` (+ `apps/band.py`, `apps/anchor.py`); `apps/predictModels.py`
adds the database and the model files.

## Data

- Only samples from opening time (08:00 on weekdays, 10:00 at weekends and on
  Czech public holidays) until midnight are used. Before opening the counter
  still shows the previous evening's residue of a few hundred "people".
- Zero counts and days whose busiest moment stays under 50 people (closures, a
  stuck counter) are ignored.
- Every day is resampled onto 10-minute buckets, so the old 10-minute and the
  current 20-minute collection cadence look the same. Gaps over an hour are left
  empty rather than interpolated.

## The norm and today's level

The library's level drifts a lot: semester, exam period and break differ by a
factor of three, and the same month grows year over year. The *shape* of a day
of a given type (Mon–Thu, Fri, Sat, Sun/holiday) is stable.

- **Norm**: per bucket, the median of the last 3 days of the same type, no older
  than 28 days (older ones only after a long gap). Three days rather than six:
  fresher reference days follow the level drift much better.
- **Level ratio**: today's observed curve divided by the norm over the buckets
  seen so far (the last 30 minutes count three times), shrunk towards 1.0 with a
  prior of 6 buckets, so one noisy early sample cannot rescale the whole day.
- **Baseline (shape × level)**: norm × level ratio. Before any sample of the day
  it is the norm itself.

## The model

CatBoost predicts `log((y + 30) / (baseline + 30))`, the deviation of the real
count from the baseline, with a median loss (an honest p50). Because the target
is relative, the model never has to extrapolate absolute levels it has not
seen. Features, all known at the moment of the forecast:

- calendar: minute of day, weekday, day type, holiday, month, day of year;
- norm: the norm at that time, its peak and spread, number of reference days,
  how far the last two days ran above or below their own norm;
- today: whether anything is observed, the raw and shrunk level ratio, how many
  buckets are observed, minutes since the last sample, the last sample relative
  to the baseline, minutes since opening.

Training rows: for every past day and every cut (cold, then every hour
09:00–20:00), the state as of the cut and one row per real sample after it.
Rows are weighted by the baseline level, because the loss is relative while
errors matter in people. Three models with different seeds are averaged into
one: a single model's error moves by about 3 people with the seed alone.

## Serving

1. Point forecast: CatBoost if it won the gate, otherwise the baseline. CatBoost
   is blended with the baseline by how much of the day is observed: the
   baseline's weight is `n / (n + 24)` for `n` observed 10-minute buckets, so
   the cold forecast is pure CatBoost and the evening forecast mostly the
   rescaled norm.
2. Anchor: the median is shifted onto the latest real sample, with the shift
   decaying by 15 % per 10 minutes.
3. Band: around the median, the 10th and 90th percentile of recent relative
   errors `log((y + 30) / (p50 + 30))` for that horizon (no sample yet, ≤ 1 h,
   1–3 h, > 3 h after the last sample). The band therefore widens with the
   horizon and scales with the level.

## Daily training and the gate

Once a day (`learn_models`):

1. The last 12 weeks are split into three 4-week windows. For each window a
   (single-seed) model is fitted on the data before it, and both CatBoost and the baseline
   forecast every day of the window at every hourly cut, exactly as served.
2. CatBoost is served only if it beats the baseline on these replays (overall
   MAE, without losing on the 3 hours after the cut), and the challenger has to
   win by 5 % to replace the incumbent, so the forecast does not flip shape on
   noise. Both see the same information each day; only CatBoost's learned
   mapping is frozen at the window start, which makes the gate conservative.
3. The replay errors of the chosen forecaster calibrate the band (single-seed
   errors are slightly larger than the ensemble's, so the band errs on the
   wide side).
4. The final ensemble is fitted on every complete day.

Model files carry a version (`model_v2.cbm`, `band_v2.json`); after an upgrade
the bot serves the baseline and retrains on the first maintenance tick.

## Evaluation

`scripts/experiments` is a walk-forward backtest of exactly this pipeline:

- 17 fold weeks across 2024-01 … 2026-10: winter and summer exam periods,
  semester weeks, summer breaks, the Christmas run-up, a semester start and the
  latest weeks. Each fold trains (gate included) on the days before the week's
  Monday and forecasts every open day of the week.
- Cuts 06:45 (cold, before opening), 10:00, 12:00 and 14:00, plus 13:20, 16:40
  and 19:00 reported separately. A forecaster only sees the day's samples up to
  the cut; scores use the real samples after it.
- Metrics: MAE in people; near-term MAE (3 h after the cut); error of the peak
  value and of its time over the rest of the day; pinball loss of p10/p90;
  coverage of the p10–p90 band (target 0.80). Each (day, cut) weighs the same.
  Differences come with a paired bootstrap interval over days.

```sh
BOT_TOKEN=ci-token uv run python -m scripts.experiments.run served catboost baseline \
    [--sqlite snapshot.sqlite]   # default: the configured database
uv run python -m scripts.experiments.report served catboost baseline --halves
```

### Current numbers

Backtest of 2026-10-28 (17 fold weeks, 108 open days). MAE and peak error in
people, peak time in minutes; "extra" = the 13:20 / 16:40 / 19:00 cuts;
coverage of the p10–p90 band (target 0.80); Δ = paired bootstrap 95 % interval
of the MAE difference to the served forecaster, over days.

| forecaster | 06:45 | 10:00 | 12:00 | 14:00 | extra | near 3 h @14 | peak | peak time | pinball | coverage | Δ MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| served (gate) | 55 | 58 | 50 | 43 | 39 | 35 | 58 | 31 | 16.4 | 0.84 | ref |
| CatBoost only | 53 | 56 | 50 | 43 | 39 | 35 | 56 | 31 | 16.4 | 0.84 | −1.1 [−2.2, −0.2] |
| baseline (shape × level) | 77 | 75 | 55 | 45 | 39 | 40 | 81 | 32 | 23.0 | 0.81 | +11.2 [+4.5, +19.3] |

Band coverage per cut stays between 0.76 (19:00) and 0.85. The gate served
the baseline in one of the 17 folds (March 2024, with only six
months of history); every later fold served CatBoost. The two halves of the
folds (odd and even weeks) rank the forecasters the same way. The cold
forecast benefits most; once a few hours are observed the baseline is close,
and in the evening the two are equal, which is why they are blended. The
harness starts every run from CatBoost as the incumbent, as production does.

### What did not help

Measured on the same folds (single runs; differences of 1–2 are within the
seed noise of one model):

| change | effect on MAE |
|---|---|
| 6 reference days instead of 3 | baseline +20 at 06:45, +10 at 12:00 (stale level) |
| Weather features (even with the observed weather of the day) | ±0 |
| Ratio to norm of the same week in earlier years ("empirical academic calendar") | ±0 |
| Scaling the cold forecast by the last two days' ratio to norm | +85 at 06:45 |
| Training cuts every two hours instead of hourly | +5 overall |
| RMSE instead of the median loss | ±0 (within seed noise) |
| Recency weighting of training days (half-life one year) | +2.4 |
| No anchor on the CatBoost forecast | +1, +4 in the evening |
| Single model instead of a 3-seed ensemble | ±3 depending on the seed |
| No blend with the baseline | +0.6 overall, +2.4 at 13:20 and later |
| Unweighted rows instead of weighting by level | +1.3 overall, +3.7 at 06:45 |

An explicit ČVUT / VŠCHT semester calendar was not added: the dates are not
available in a stable machine-readable form, and the empirical calendar above
did not help.
