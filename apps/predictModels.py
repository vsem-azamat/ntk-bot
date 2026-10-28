"""Database and file wrapper around ``apps.forecaster``.

``learn_models`` (daily, and via /learn) runs the walk-forward gate, fits the
final model and stores the model, the calibrated band and the choice of
forecaster in ``DATA_DIR``. ``predict_day`` is the public API the plot and the
digest consume. How the forecaster works: ``docs/prediction.md``.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from catboost import CatBoost

from apps.band import ResidualBand
from apps.forecaster import (
    BASELINE,
    CATBOOST,
    Choice,
    DayForecast,
    History,
    Trained,
    forecast,
    train,
)
from apps.hours import usable_samples
from config import cnfg

__all__ = ["DayForecast", "PredictModels", "predictModels"]

logger = logging.getLogger(__name__)

# Model and band files are versioned so artefacts of an older, incompatible
# model are never loaded: after an upgrade the bot serves the baseline until it
# retrains. The choice file is not: the served kind (CatBoost or baseline)
# carries over as the gate's incumbent, so the new CatBoost keeps serving unless
# the baseline clearly beats it.
_MODEL_FILE = "model_v2.cbm"
_BAND_FILE = "band_v2.json"
_CHOICE_FILE = "model_choice"
# Files of the previous model generation, removed after the first v2 training.
_STALE_FILES = ("model_p10.cbm", "model_p50.cbm", "model_p90.cbm")


def _path(name: str) -> Path:
    return Path(cnfg.DATA_DIR) / name


def _load_history() -> History:
    """Blocking (database + indexing): call it in a worker thread."""
    from bot import db

    return History(usable_samples(db.fetch_occupancy()))


def _read_choice() -> Choice:
    path = _path(_CHOICE_FILE)
    choice = path.read_text(encoding="utf-8").strip() if path.exists() else BASELINE
    return CATBOOST if choice == CATBOOST else BASELINE


def _load_model() -> CatBoost | None:
    path = _path(_MODEL_FILE)
    if _read_choice() != CATBOOST or not path.exists():
        return None
    model = CatBoost()
    model.load_model(str(path))
    return model


def _replace(name: str, write: Callable[[Path], object]) -> None:
    """Write ``name`` via a temporary file and an atomic rename, so a forecast
    running at the same moment never reads a half-written file."""
    target = _path(name)
    tmp = target.with_name(target.name + ".tmp")
    write(tmp)
    os.replace(tmp, target)


def _save(trained: Trained) -> None:
    model = trained.model
    if model is not None:
        _replace(_MODEL_FILE, lambda p: model.save_model(str(p)))
    _replace(_BAND_FILE, trained.band.save)
    _replace(_CHOICE_FILE, lambda p: p.write_text(trained.choice, encoding="utf-8"))
    for name in _STALE_FILES:
        _path(name).unlink(missing_ok=True)


def _predict(now: datetime) -> DayForecast:
    """Blocking (database, files, model): call it in a worker thread."""
    history = _load_history()
    return forecast(history, now.date(), now, _load_model(), ResidualBand.load(_path(_BAND_FILE)))


class PredictModels:
    async def learn_models(self) -> None:
        """Gate, calibrate and fit on all stored data, then persist the result."""
        history = await asyncio.to_thread(_load_history)
        if not history.by_day:
            logger.info("No occupancy data yet; serving the baseline")
            return
        # Train on complete days only: today's partial day is never a target.
        train_end = min(date.today(), max(history.by_day) + timedelta(days=1))
        trained = await asyncio.to_thread(train, history, train_end, _read_choice())
        await asyncio.to_thread(_save, trained)

    def has_trained(self) -> bool:
        """True once ``learn_models`` has stored the current model generation,
        so a cold start (or an upgrade) can be told apart from a plain restart."""
        return _path(_CHOICE_FILE).exists() and _path(_BAND_FILE).exists()

    async def predict_day(self, target_day: datetime | None = None) -> DayForecast:
        """Forecast the full day of ``target_day`` as of that moment (now by
        default): CatBoost if it won the gate, else the baseline."""
        now = target_day or datetime.now()
        return await asyncio.to_thread(_predict, now)


predictModels = PredictModels()
