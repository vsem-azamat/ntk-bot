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

        bands = [sorted((max(0.0, a), max(0.0, b), max(0.0, c))) for a, b, c in zip(p10, p50, p90, strict=True)]
        return DayForecast(
            timestamps=grid,
            p10=[b[0] for b in bands],
            p50=[b[1] for b in bands],
            p90=[b[2] for b in bands],
        )


predictModels = PredictModels()
