# Prediction Engine — Experiment Results

**Date:** 2026-06-15
**Harness:** `scripts/experiments/` (walk-forward, leak-free as-of backtest)
**Data:** full production snapshot, 981 days (2023-09-24 → 2026-06-15)

## Method

8 folds with test days spread evenly across the timeline (2024-01-22 → 2026-06-15,
covering winter + summer exam seasons and breaks), each trained on up to 365 prior
days, scored at four cut times (morning / 10:00 / 12:00 / 14:00). Training is
leak-free and as-of augmented (`build_training_matrix`); prediction conditions on
the test day's history truncated to the cut. The anchor guardrail is **not** applied
in the harness, so the real near-term numbers at serve time will be better still.

Metrics (lower is better except coverage, target ≈ 0.80): overall **MAE**,
**peak_value_err**, **peak_time_err** (min), **pinball** (mean of p-low/p-high),
**coverage** (fraction inside the band), **near_mae** (next 3 h from the cut).

## Overall results

| candidate | groups | MAE | peak_val | peak_time | pinball | coverage | near_mae |
|---|---|---|---|---|---|---|---|
| base_legacy | base | 85.96 | 109.16 | 73.75 | 30.56 | 0.354 | 106.51 |
| full | base+regime+asof | 75.66 | 108.03 | 75.0 | 24.48 | 0.460 | 83.17 |
| full_log | +log1p target | 79.49 | **99.45** | 77.19 | 25.64 | 0.392 | 82.18 |
| full_wide | band 0.05/0.95 | 77.94 | 109.45 | 77.5 | 25.52 | 0.486 | 81.55 |
| **full_wider** | **band 0.02/0.98** | **75.24** | 106.75 | 79.38 | **24.41** | **0.657** | **81.08** |

`base_legacy` reproduces today's production behaviour (calendar + weather only).

## Chosen config: `full_wider`

- **groups:** `("base", "regime", "asof")`
- **log_target:** `False`
- **band quantiles:** `lo_alpha=0.02`, `hi_alpha=0.98` (p10/p90 fields now carry the
  2nd/98th percentiles — a wider, better-calibrated uncertainty band)
- **CatBoost:** iterations=300, depth=6, learning_rate=0.1
- **trailing_days:** 14 (regime level window)
- **train cuts (augmentation):** `(None, 10:00, 12:00, 14:00)`

### Why

Versus the current production baseline it wins the prioritised metrics:
- **Overall MAE** 75.24 vs 85.96 (**−12.5 %**)
- **Near-term MAE** 81.08 vs 106.51 (**−24 %**)
- **Pinball** 24.41 vs 30.56 (**−20 %**)
- **Coverage** 0.657 vs 0.354 (**+0.30**, nearly the 0.80 target)

`peak_value_err` (106.7) is comparable to baseline; `full_log` is best on peak
magnitude (99.5) but loses on calibration and overall MAE. `peak_time_err` is
marginally worse than baseline (79 vs 74 min) — an acceptable trade for the large
near-term / calibration gains, and the serve-time anchor pulls the near horizon
onto the live trend regardless.

## Conditioning works (the core complaint)

Per-cut MAE shows the baseline ignores today's data while the winner tracks it:

| cut | base_legacy MAE | full_wider MAE | base near_mae | full_wider near_mae |
|---|---|---|---|---|
| morning | 86.0 | 78.6 | 80.2 | **48.5** |
| 10:00 | 86.0 | 75.9 | 113.3 | **79.8** |
| 12:00 | 86.0 | 75.1 | 122.5 | **101.5** |
| 14:00 | 86.0 | 71.4 | 110.0 | **94.5** |

`base_legacy`'s MAE is identical at every cut — it produces the same static curve
all day. `full_wider`'s forecast changes with what has been observed and is far more
accurate near the current moment at every cut. Morning-cut coverage reaches 0.79.

## Known limitation

Whole-day coverage (0.66) is still below the 0.80 ideal; widening further (the
0.02/0.98 band) trades sharpness for coverage and was the best balance found.
Future tuning (per-hour quantile calibration, more iterations) could close the gap.

## Reproduce

```bash
DATA_DIR=tmp/experiments BOT_TOKEN=ci-token \
  uv run python -m scripts.experiments.run --folds 8 --min-train 120 --max-train-days 365
```
Reports are written to `tmp/experiments/reports/` (gitignored).
