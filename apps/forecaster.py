"""The occupancy forecaster, free of I/O: training, the model gate and serving.

Two point forecasters share one pipeline:

* ``baseline`` -- shape x level (``apps.level.shape_level``): the norm curve of
  the recent same-type days, rescaled to how busy today has been so far.
* ``catboost`` -- a CatBoost model that predicts the log-ratio of the real count
  to that baseline (see ``apps.features``), i.e. learns when and how today
  deviates from it.

Serving is the same for both: point forecast -> anchor onto the latest real
sample -> calibrated band (``apps.band``). ``train`` replays both forecasters on
the latest weeks they were not trained on (walk-forward, several windows), uses
those errors to pick the served one and to calibrate its band, then fits the
final model on everything.

``apps.predictModels`` wraps this with the database and files; the backtest in
``scripts/experiments`` drives exactly these functions.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Literal

import numpy as np
from catboost import CatBoost, CatBoostRegressor, Pool, sum_models

from apps.anchor import anchor_band
from apps.band import ResidualBand, horizon_bucket, relative_error
from apps.features import TARGET_OFFSET, TrainingSet, from_target, state_rows, training_rows
from apps.hours import forecast_grid, opening_time
from apps.level import day_state, shape_level
from apps.reference import DayCurves, bucket_of

logger = logging.getLogger(__name__)

Rows = list[tuple[datetime, int]]

Choice = Literal["baseline", "catboost"]
BASELINE: Choice = "baseline"
CATBOOST: Choice = "catboost"

# Feature groups of the served model (weather and a last-year ratio were tried
# and left out: neither lowered the backtest error, see docs/prediction.md).
FEATURE_GROUPS: tuple[str, ...] = ("calendar", "norm", "today")
# Cut times the training set is built at and the gate and the band replay the
# forecasters at: the cold morning case plus every hour of the day, so the
# model learns every stage of the day it is asked about (hourly cuts beat
# two-hourly ones in the backtest).
TRAIN_CUTS: tuple[time | None, ...] = (None,) + tuple(time(h, 0) for h in range(9, 21))
REPLAY_CUTS = TRAIN_CUTS
CATBOOST_PARAMS: dict[str, object] = {
    # Median of the log-ratio = median of the count: an honest p50.
    "loss_function": "Quantile:alpha=0.5",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.08,
    # The bot shares its host with other services; the daily retrain must not
    # take every core.
    "thread_count": 2,
}
# Walk-forward gate: this many consecutive windows of this length right before
# the training end; each window is forecast by a model fitted before it.
GATE_WINDOWS = 3
GATE_WINDOW = timedelta(weeks=4)
# The "near-term" error of the gate: points up to this far after the cut.
NEAR_HORIZON = timedelta(hours=3)
# The served forecaster only changes when the challenger beats the incumbent by
# this relative MAE margin, so the forecast does not flip shape on noise.
SWITCH_MARGIN = 0.05
# Fewer training rows than this cannot fit a sensible model.
MIN_TRAIN_ROWS = 500


@dataclass
class DayForecast:
    """Full-day forecast grid with a guaranteed-monotone band."""

    timestamps: list[datetime]
    p10: list[float]
    p50: list[float]
    p90: list[float]


class History:
    """Open-hours samples indexed per day plus their reference curves."""

    def __init__(self, rows: Rows) -> None:
        by_day: dict[date, Rows] = defaultdict(list)
        for dt, count in rows:
            if count > 0:
                by_day[dt.date()].append((dt, count))
        self.by_day: dict[date, Rows] = {d: sorted(v) for d, v in by_day.items()}
        self.curves = DayCurves(rows)

    def seen(self, day: date, now: datetime | None) -> Rows:
        """``day``'s samples at or before ``now`` (None: nothing yet)."""
        if now is None:
            return []
        return [(dt, c) for dt, c in self.by_day.get(day, []) if dt <= now]


# Serving and training constants measured in the backtest (docs/prediction.md).
# Anchor: the shift onto the latest sample decays by this factor per 10 min.
ANCHOR_DECAY = 0.85
# CatBoost is blended with the baseline by how much of the day is observed:
# the baseline's weight is n / (n + BLEND_HALF_BUCKETS) for n observed buckets.
# Late in the day the rescaled norm is the better guide (MAE at 14:00 and later
# dropped by 2-3 people), while the cold forecast stays pure CatBoost.
BLEND_HALF_BUCKETS = 24.0
# Models with different seeds averaged into one: a single model's backtest MAE
# moved by ~3 people (6 at the cold start) with the seed alone. The gate's
# window models are single models to keep the daily training time down; their
# slightly larger errors make the calibrated band a little conservative.
ENSEMBLE_SIZE = 3
GATE_ENSEMBLE_SIZE = 1
# Row weight ~ (base + C) ** LEVEL_WEIGHT_POWER. The loss is relative (log
# ratio) while what matters is people, so busy hours count more: the cold MAE
# dropped by ~4 people, for under 1 person more in the evening.
LEVEL_WEIGHT_POWER = 1.0


def row_weights(ts: TrainingSet) -> np.ndarray:
    w = np.power(ts.base + TARGET_OFFSET, LEVEL_WEIGHT_POWER)
    return w / w.mean()


def fit_catboost(history: History, train_end: date, ensemble: int | None = None) -> CatBoost | None:
    """Fit the model (an average of ``ensemble`` seeds, ``ENSEMBLE_SIZE`` by
    default) on every usable day strictly before ``train_end``."""
    days = [d for d in history.curves.days if d < train_end]
    ts = training_rows(history.curves, history.by_day, days, FEATURE_GROUPS, TRAIN_CUTS)
    if len(ts.z) < MIN_TRAIN_ROWS or np.ptp(ts.z) == 0:
        return None  # too little data, or nothing to learn beyond the baseline
    pool = Pool(ts.X, ts.z, weight=row_weights(ts))
    models: list[CatBoostRegressor] = []
    for seed in range(ENSEMBLE_SIZE if ensemble is None else ensemble):
        params = {**CATBOOST_PARAMS, "random_seed": seed}
        model = CatBoostRegressor(**params, verbose=False, allow_writing_files=False)
        model.fit(pool)
        models.append(model)
    if len(models) == 1:
        return models[0]
    return sum_models(models, weights=[1.0 / len(models)] * len(models))


def point_forecast(
    history: History,
    day: date,
    seen: Rows,
    now: datetime | None,
    model: CatBoost | None,
) -> tuple[list[datetime], np.ndarray]:
    """Grid and anchored p50 of ``day`` knowing ``seen``; CatBoost when a
    ``model`` is given, else the baseline."""
    grid = forecast_grid(day)
    buckets = np.array([bucket_of(t) for t in grid])
    state = day_state(history.curves, day, seen, now if seen else None)
    base = shape_level(state)[1][buckets]
    if model is None:
        p50 = base
    else:
        X = state_rows(state, buckets, FEATURE_GROUPS)
        p50 = from_target(np.asarray(model.predict(X), float), base)
        if BLEND_HALF_BUCKETS > 0:  # 0 disables the blend (ablations)
            n = state.observed_buckets
            w = n / (n + BLEND_HALF_BUCKETS)
            p50 = (1 - w) * p50 + w * base
    if seen:
        last_dt, last_c = seen[-1]
        idx = min(range(len(grid)), key=lambda i: abs((grid[i] - last_dt).total_seconds()))
        anchored = anchor_band(list(p50), list(p50), list(p50), idx, float(last_c), ANCHOR_DECAY)
        p50 = np.array(anchored[1])
    return grid, np.maximum(p50, 0.0)


def horizons(grid: list[datetime], seen: Rows) -> np.ndarray:
    """Band horizon bucket of every grid point."""
    if not seen:
        return np.array([horizon_bucket(None)] * len(grid))
    last = seen[-1][0]
    return np.array([horizon_bucket((t - last).total_seconds() / 60) for t in grid])


def forecast(
    history: History,
    day: date,
    now: datetime | None,
    model: CatBoost | None,
    band: ResidualBand,
) -> DayForecast:
    """The served forecast of ``day`` as of ``now``."""
    seen = history.seen(day, now)
    grid, p50 = point_forecast(history, day, seen, now, model)
    p10, p90 = band.apply(p50, horizons(grid, seen))
    return DayForecast(
        timestamps=grid,
        p10=[float(v) for v in p10],
        p50=[float(v) for v in p50],
        p90=[float(v) for v in p90],
    )


# --- gate -----------------------------------------------------------------


@dataclass
class Replay:
    """Errors of one forecaster replayed on held-out days."""

    abs_errors: list[float] = field(default_factory=list)
    near_errors: list[float] = field(default_factory=list)
    rel_errors: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))

    def metrics(self) -> dict[str, float]:
        return {
            "mae": float(np.mean(self.abs_errors)) if self.abs_errors else float("inf"),
            "near_mae": float(np.mean(self.near_errors)) if self.near_errors else float("inf"),
        }


def replay(
    history: History,
    days: list[date],
    model: CatBoost | None,
    into: Replay,
    cuts: tuple[time | None, ...] = REPLAY_CUTS,
) -> None:
    """Forecast every ``day`` at every cut exactly as served and record the
    errors on the real samples after the cut."""
    for day in days:
        samples = history.by_day.get(day, [])
        for ct in cuts:
            if ct is not None and ct < opening_time(day):
                continue  # nothing seen yet: a duplicate of the cold forecast
            now = None if ct is None else datetime.combine(day, ct)
            seen = history.seen(day, now)
            targets = [(dt, c) for dt, c in samples if now is None or dt > now]
            if not targets:
                continue
            grid, p50 = point_forecast(history, day, seen, now, model)
            gx = np.array([t.hour * 60 + t.minute for t in grid], float)
            tx = np.array([t.hour * 60 + t.minute for t, _ in targets], float)
            y = np.array([c for _, c in targets], float)
            pred = np.interp(tx, gx, p50)
            err = np.abs(pred - y)
            into.abs_errors.extend(err.tolist())
            start = targets[0][0] if now is None else now
            near = np.array([dt <= start + NEAR_HORIZON for dt, _ in targets])
            if seen:
                into.near_errors.extend(err[near].tolist())
            rel = relative_error(y, pred)
            last = seen[-1][0] if seen else None
            for (dt, _), e in zip(targets, rel, strict=True):
                ahead = None if last is None else (dt - last).total_seconds() / 60
                into.rel_errors[horizon_bucket(ahead)].append(float(e))


def better_than_baseline(cat: dict[str, float], base: dict[str, float]) -> bool:
    """CatBoost beats the baseline if it wins overall MAE AND does not regress
    the near-term error."""
    return cat["mae"] < base["mae"] and cat["near_mae"] <= base["near_mae"]


def choose_model(
    cat: dict[str, float], base: dict[str, float], incumbent: Choice, margin: float = SWITCH_MARGIN
) -> Choice:
    """Pick the served forecaster with hysteresis: the challenger must beat the
    incumbent by ``margin`` (relative) to replace it. An incumbent CatBoost is
    also dropped when its near-term error is clearly worse than the baseline's."""
    if incumbent == CATBOOST:
        if base["mae"] < cat["mae"] * (1 - margin):
            return BASELINE
        if base["near_mae"] < cat["near_mae"] * (1 - margin):
            return BASELINE
        return CATBOOST
    if better_than_baseline(cat, base) and cat["mae"] < base["mae"] * (1 - margin):
        return CATBOOST
    return BASELINE


@dataclass
class Trained:
    choice: Choice
    model: CatBoost | None
    band: ResidualBand
    scores: dict[str, dict[str, float]]


def gate_replays(history: History, train_end: date) -> dict[str, Replay]:
    """Replay both forecasters over the ``GATE_WINDOWS`` windows right before
    ``train_end``; in each window CatBoost is a model fitted before it.

    Both forecasters see the same information on every replayed day (the norm
    and today's samples as of the cut); only CatBoost's learned mapping is
    frozen at the window start, which makes the gate slightly conservative
    towards it -- in production it is refitted daily."""
    runs = {CATBOOST: Replay(), BASELINE: Replay()}
    for w in range(GATE_WINDOWS, 0, -1):
        start = train_end - GATE_WINDOW * w
        days = [d for d in history.curves.days if start <= d < start + GATE_WINDOW]
        if not days:
            continue
        window_model = fit_catboost(history, start, GATE_ENSEMBLE_SIZE)
        if window_model is None:
            continue  # score both forecasters on the same windows only
        replay(history, days, window_model, runs[CATBOOST])
        replay(history, days, None, runs[BASELINE])
    return runs


def train(history: History, train_end: date, incumbent: Choice) -> Trained:
    """Gate, band calibration and the final fit on every day before ``train_end``."""
    runs = gate_replays(history, train_end)
    model = fit_catboost(history, train_end)
    return decide(runs, model, incumbent)


def decide(runs: dict[str, Replay], model: CatBoost | None, incumbent: Choice) -> Trained:
    """Pick the served forecaster from the replays and calibrate its band."""
    scores = {k: r.metrics() for k, r in runs.items()}
    choice = BASELINE
    if model is not None and runs[CATBOOST].abs_errors:
        choice = choose_model(scores[CATBOOST], scores[BASELINE], incumbent)
    logger.info(
        "Gate: catboost(mae=%.1f near=%.1f) baseline(mae=%.1f near=%.1f) incumbent=%s -> %s",
        scores[CATBOOST]["mae"],
        scores[CATBOOST]["near_mae"],
        scores[BASELINE]["mae"],
        scores[BASELINE]["near_mae"],
        incumbent,
        choice,
    )
    band = ResidualBand.fit(dict(runs[choice].rel_errors))
    return Trained(choice, model if choice == CATBOOST else None, band, scores)
