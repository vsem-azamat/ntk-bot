"""Occupancy forecaster: CatBoost median + a wide quantile band, time-validated and
benchmarked against a climatology baseline. ``predict_day`` is the public API the
plot layer consumes; it serves CatBoost only when it beat the baseline, otherwise
the baseline itself.

The model is CONDITIONAL: it reads today's observations so far (leak-free as-of
features) so the intraday forecast continues the current trend, and an anchor
guardrail snaps the band onto the latest real sample. The feature set, target
transform and band quantiles below are the winner of the offline backtest — see
``docs/superpowers/specs/2026-06-15-prediction-results.md``.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

from apps.baseline import build_climatology, climatology_predict
from apps.features import ObsContext, WeatherRow, build_matrix, build_training_matrix
from apps.hours import day_end, is_open_sample, opens_at
from config import cnfg

logger = logging.getLogger(__name__)

# Winning config from the backtest. Update if re-tuned (and re-run the harness).
_FEATURE_GROUPS = ("base", "regime", "asof")
_LOG_TARGET = False
_LO_ALPHA, _HI_ALPHA = 0.02, 0.98  # wide band => better p10/p90 coverage
# As-of cut points that AUGMENT training so the model learns both the cold morning
# case and conditioned mid-day cases. None == cold/morning start.
_TRAIN_CUTS: tuple[time | None, ...] = (None, time(10, 0), time(12, 0), time(14, 0))

_QUANTILES = {"p10": _LO_ALPHA, "p50": 0.5, "p90": _HI_ALPHA}
_CHOICE_FILE = "model_choice"
_HOLDOUT_WEEKS = 4
# The holdout is a single noisy window, so the served model only changes when the
# challenger wins by this relative MAE margin; otherwise the incumbent stays.
_SWITCH_MARGIN = 0.05


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


def usable_samples(rows: list[tuple[datetime, int]]) -> list[tuple[datetime, int]]:
    """Keep samples the model can trust: non-zero counts taken after the day's
    opening time. Zeros mean closed / no reading; pre-opening samples are the
    previous night's counter residue (see ``apps.hours``)."""
    return [(dt, c) for dt, c in rows if c != 0 and is_open_sample(dt)]


def time_split(
    rows: list[tuple[datetime, int]], holdout_weeks: int = _HOLDOUT_WEEKS
) -> tuple[list[tuple[datetime, int]], list[tuple[datetime, int]]]:
    """Split chronologically at a day boundary: the last ``holdout_weeks`` as
    validation, the rest as train (a whole day is always on one side)."""
    cutoff = datetime.combine((rows[-1][0] - timedelta(weeks=holdout_weeks)).date(), time())
    train = [r for r in rows if r[0] < cutoff]
    val = [r for r in rows if r[0] >= cutoff]
    return train, val


def _grid(target_day: datetime) -> list[datetime]:
    """Plot-matching grid: opening time until midnight at 10-min steps."""
    start = opens_at(target_day.date())
    end = day_end(target_day.date())
    out: list[datetime] = []
    cur = start
    while cur < end:
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


def _conditioned_mae(pred: list[float], actual: np.ndarray, feats) -> float:
    """MAE over only the examples that have observed today's data (``has_today``).

    This is the near-term proxy for the ship gate: it measures accuracy precisely
    when the model is conditioning on a partial day. Falls back to overall MAE when
    the feature set has no as-of column or no conditioned rows exist."""
    if "has_today" not in feats.names:
        return _mae(pred, actual)
    idx = feats.names.index("has_today")
    mask = feats.X[:, idx].astype(float) == 1.0
    if not mask.any():
        return _mae(pred, actual)
    return _mae(list(np.array(pred)[mask]), actual[mask])


def better_than_baseline(cat: dict, clim: dict) -> bool:
    """CatBoost beats the baseline if it wins overall MAE AND does not regress the
    near-term (conditioned) error. Both dicts carry at least ``mae`` and ``near_mae``."""
    return cat["mae"] < clim["mae"] and cat["near_mae"] <= clim["near_mae"]


def choose_model(cat: dict, clim: dict, incumbent: str, margin: float = _SWITCH_MARGIN) -> str:
    """Pick the served model with hysteresis: the challenger must beat the
    incumbent by ``margin`` (relative) to replace it, so a coin-flip holdout does
    not swap the forecast shape every day. An incumbent CatBoost is also dropped
    when its near-term error is clearly worse than the baseline's."""
    if incumbent == "catboost":
        if clim["mae"] < cat["mae"] * (1 - margin):
            return "climatology"
        if clim["near_mae"] < cat["near_mae"] * (1 - margin):
            return "climatology"
        return "catboost"
    if better_than_baseline(cat, clim) and cat["mae"] < clim["mae"] * (1 - margin):
        return "catboost"
    return "climatology"


def _climatology_on(rows: list[tuple[datetime, int]], timestamps: list[datetime]) -> list[float]:
    """Baseline p50 for ``timestamps``, built per day from the days before it."""
    out: list[float] = [0.0] * len(timestamps)
    by_day: dict = {}
    for i, ts in enumerate(timestamps):
        by_day.setdefault(ts.date(), []).append(i)
    for day, idxs in by_day.items():
        _, p50, _ = climatology_predict(build_climatology(rows, day), [timestamps[i] for i in idxs])
        for i, v in zip(idxs, p50, strict=True):
            out[i] = v
    return out


class PredictModels:
    async def learn_models(self) -> None:
        """Train CatBoost on a time-based holdout, benchmark vs climatology, and
        persist both the models and the ship decision."""
        from bot import db

        rows = usable_samples(db.fetch_occupancy())
        if len(rows) <= 50:
            logger.info("Not enough data to train (%d rows); baseline only", len(rows))
            self._write_choice("climatology")
            return

        train, val = time_split(rows)
        if not train or not val:
            self._write_choice("climatology")
            return

        weather = db.fetch_weather(
            rows[0][0].replace(minute=0, second=0, microsecond=0), rows[-1][0]
        )

        train_days = sorted({dt.date() for dt, _ in train})
        feats, train_y = build_training_matrix(
            train, train_days, weather=weather, groups=_FEATURE_GROUPS, cut_times=_TRAIN_CUTS
        )
        await asyncio.to_thread(self._fit_and_save, feats, train_y)

        # Leak-free validation: build val examples from the FULL history (so regime
        # / as-of context reaches back before the holdout) but only the holdout days
        # emit targets. CatBoost and climatology are scored on the identical rows.
        val_days = sorted({dt.date() for dt, _ in val})
        val_feats, val_y, val_ts = build_training_matrix(
            rows,
            val_days,
            weather=weather,
            groups=_FEATURE_GROUPS,
            cut_times=_TRAIN_CUTS,
            return_timestamps=True,
        )
        if val_feats.X.shape[0] == 0:
            logger.info("No validation examples; defaulting to climatology")
            self._write_choice("climatology")
            return

        cat_p50 = await asyncio.to_thread(self._predict_X, "p50", val_feats.X)
        clim_p50 = _climatology_on(rows, val_ts)

        cat_metrics = {
            "mae": _mae(cat_p50, val_y),
            "near_mae": _conditioned_mae(cat_p50, val_y, val_feats),
        }
        clim_metrics = {
            "mae": _mae(clim_p50, val_y),
            "near_mae": _conditioned_mae(clim_p50, val_y, val_feats),
        }
        incumbent = self._read_choice()
        choice = choose_model(cat_metrics, clim_metrics, incumbent)
        logger.info(
            "Holdout — catboost(mae=%.2f near=%.2f) climatology(mae=%.2f near=%.2f)"
            " incumbent=%s -> %s",
            cat_metrics["mae"],
            cat_metrics["near_mae"],
            clim_metrics["mae"],
            clim_metrics["near_mae"],
            incumbent,
            choice,
        )
        self._write_choice(choice)

    def _fit_and_save(self, feats, y: np.ndarray) -> None:
        target = np.log1p(y) if _LOG_TARGET else y
        for name, alpha in _QUANTILES.items():
            model = _train_quantile(feats.X, target, feats.categorical_indices, alpha)
            model.save_model(_model_path(name))

    def _predict_X(self, name: str, X: np.ndarray) -> list[float]:
        """Predict a quantile from a prebuilt feature matrix, inverting the target
        transform and clamping at zero."""
        model = CatBoostRegressor(allow_writing_files=False)
        model.load_model(_model_path(name))
        raw = np.asarray(model.predict(X), float)
        raw = np.expm1(raw) if _LOG_TARGET else raw
        return [max(0.0, float(v)) for v in raw]

    def _predict_quantile(
        self,
        name: str,
        timestamps: list[datetime],
        weather: dict[datetime, WeatherRow] | None,
        ctx: ObsContext | None,
    ) -> list[float]:
        feats = build_matrix(timestamps, ctx=ctx, weather=weather, groups=_FEATURE_GROUPS)
        return self._predict_X(name, feats.X)

    def _predict_catboost(
        self,
        grid: list[datetime],
        weather: dict[datetime, WeatherRow] | None,
        ctx: ObsContext | None,
    ) -> tuple[list[float], list[float], list[float]]:
        # Build the feature matrix once and reuse it for all three quantiles.
        feats = build_matrix(grid, ctx=ctx, weather=weather, groups=_FEATURE_GROUPS)
        return (
            self._predict_X("p10", feats.X),
            self._predict_X("p50", feats.X),
            self._predict_X("p90", feats.X),
        )

    def _write_choice(self, choice: str) -> None:
        Path(_choice_path()).write_text(choice, encoding="utf-8")

    def _read_choice(self) -> str:
        path = Path(_choice_path())
        return path.read_text(encoding="utf-8").strip() if path.exists() else "climatology"

    def _catboost_available(self) -> bool:
        return all(Path(_model_path(n)).exists() for n in _QUANTILES)

    def has_trained(self) -> bool:
        """True once ``learn_models`` has run at least once (a ship decision was
        recorded), so a cold start can be told apart from a plain restart."""
        return Path(_choice_path()).exists()

    async def predict_day(self, target_day: datetime | None = None) -> DayForecast:
        """Public API: forecast the full day. CatBoost (conditioned on today's data
        so far, anchored to the latest real sample) if it won, else the baseline."""
        from apps.anchor import anchor_band
        from bot import db

        target_day = target_day or datetime.now()
        grid = _grid(target_day)
        rows = usable_samples(db.fetch_occupancy())
        ctx = ObsContext.from_rows(rows)

        use_catboost = self._read_choice() == "catboost" and self._catboost_available()
        if use_catboost:
            from apps.weather_history import forecast_weather

            try:
                weather = await forecast_weather(target_day)
            except Exception:
                logger.exception("Forecast weather fetch failed; predicting without it")
                weather = {}
            p10, p50, p90 = await asyncio.to_thread(self._predict_catboost, grid, weather, ctx)
        else:
            clim = build_climatology(rows, target_day.date())
            p10, p50, p90 = climatology_predict(clim, grid)

        # Anchor the band onto the latest real sample observed so far today (if any),
        # so the live curve connects to reality and continues the current trend.
        today_samples = [(dt, c) for dt, c in ctx.samples_on(target_day.date()) if dt <= target_day]
        anchor_index: int | None = None
        anchor_value: float | None = None
        if today_samples:
            last_dt, last_c = today_samples[-1]
            anchor_index = min(
                range(len(grid)), key=lambda i: abs((grid[i] - last_dt).total_seconds())
            )
            anchor_value = float(last_c)
        p10, p50, p90 = anchor_band(p10, p50, p90, anchor_index, anchor_value)

        bands = [
            sorted((max(0.0, a), max(0.0, b), max(0.0, c)))
            for a, b, c in zip(p10, p50, p90, strict=True)
        ]
        return DayForecast(
            timestamps=grid,
            p10=[b[0] for b in bands],
            p50=[b[1] for b in bands],
            p90=[b[2] for b in bands],
        )


predictModels = PredictModels()
