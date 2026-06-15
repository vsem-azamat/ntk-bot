"""Walk-forward backtest: for each fold and cut time, score a candidate against
the held-out day's actual occupancy on all four metric families."""

from datetime import date, datetime

import numpy as np

from apps.features import ObsContext
from scripts.experiments.candidates import Candidate, fit_models, predict_grid
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


def _actual_on_grid(grid: list[datetime], rows: list[tuple[datetime, int]]) -> list[float] | None:
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
    cand: Candidate,
    rows: list[tuple[datetime, int]],
    ctx: ObsContext,
    train_days: list[date],
    test_day: date,
    weather: dict,
) -> list[dict]:
    results: list[dict] = []
    models = fit_models(cand, rows, train_days, weather)  # train once, reuse per cut
    for cut in cut_times(test_day):
        grid, p10, p50, p90 = predict_grid(cand, models, ctx, test_day, cut, weather)
        actual = _actual_on_grid(grid, rows)
        if actual is None:
            continue
        start = 0
        if cut is not None:
            start = min(range(len(grid)), key=lambda i: abs((grid[i] - cut).total_seconds()))
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
