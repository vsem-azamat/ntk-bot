"""Forecasters the harness scores: the production pipeline and its two parts."""

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from catboost import CatBoost

from apps.band import ResidualBand
from apps.forecaster import (
    CATBOOST,
    Choice,
    Replay,
    decide,
    fit_catboost,
    forecast,
    gate_replays,
)
from scripts.experiments.harness import Band, Data

# Gate replays and final models per (dataset, fold), shared by the variants so a
# run of all three trains each fold once.
_FITS: dict[tuple[int, date], tuple[dict[str, Replay], CatBoost | None]] = {}


@dataclass
class Production:
    """``apps.forecaster`` end to end: walk-forward gate, calibrated band,
    anchored point forecast. ``force`` serves one forecaster whatever the gate
    says, with its band calibrated on the same replay."""

    name: str
    force: Choice | None = None
    # Hysteresis state carried from fold to fold, starting where production is:
    # serving CatBoost.
    incumbent: Choice = CATBOOST
    choices: list[tuple[str, str]] = field(default_factory=list)

    def fit(self, data: Data, train_end: date) -> None:
        key = (id(data), train_end)
        if key not in _FITS:
            _FITS[key] = (
                gate_replays(data.history, train_end),
                fit_catboost(data.history, train_end),
            )
        runs, model = _FITS[key]
        trained = decide(runs, model, incumbent=self.incumbent)
        self.incumbent = trained.choice
        self.choices.append((train_end.isoformat(), trained.choice))
        if self.force is not None:
            trained.choice = self.force
            trained.model = model if self.force == CATBOOST else None
            trained.band = ResidualBand.fit(dict(runs[self.force].rel_errors))
        self._model = trained.model
        self._band = trained.band

    def predict(self, data: Data, day: date, cut: datetime) -> Band:
        f = forecast(data.history, day, cut, self._model, self._band)
        return Band(f.timestamps, np.array(f.p10), np.array(f.p50), np.array(f.p90))


@dataclass
class CatBoostOnly:
    """``fit_catboost`` + the served point forecast, no gate and no band: the
    fast loop for point-forecast ablations (knobs set via ``EXPERIMENT_KNOBS``)."""

    name: str

    def fit(self, data: Data, train_end: date) -> None:
        self._model = fit_catboost(data.history, train_end)

    def predict(self, data: Data, day: date, cut: datetime) -> Band:
        f = forecast(data.history, day, cut, self._model, ResidualBand({}))
        return Band(f.timestamps, np.array(f.p10), np.array(f.p50), np.array(f.p90))
