import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest

import apps.forecaster as forecaster
from apps.band import ResidualBand
from apps.forecaster import History, fit_catboost, forecast

START = date(2025, 1, 6)
DAYS = 60


def _rows(scale_after: datetime | None = None) -> list[tuple[datetime, int]]:
    """Bell-shaped days with a day-to-day wobble; everything after
    ``scale_after`` tripled."""
    out = []
    for d in range(DAYS):
        day = datetime.combine(START + timedelta(days=d), datetime.min.time())
        first = 10 * 60 if day.weekday() >= 5 else 8 * 60
        for minute in range(first, 24 * 60, 20):
            dt = day + timedelta(minutes=minute)
            c = round(max(10, 300 - abs(minute - 840) // 2) * (1 + 0.1 * math.sin(d)))
            out.append((dt, c * 3 if scale_after and dt > scale_after else c))
    return out


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    monkeypatch.setattr(
        forecaster, "CATBOOST_PARAMS", {**forecaster.CATBOOST_PARAMS, "iterations": 30}
    )
    monkeypatch.setattr(forecaster, "ENSEMBLE_SIZE", 1)


def test_forecast_does_not_see_anything_after_the_cut():
    day = START + timedelta(days=50)
    cut = datetime.combine(day, datetime.min.time()).replace(hour=12)
    train_end = START + timedelta(days=45)
    band = ResidualBand({0: (-0.1, 0.1), 1: (-0.1, 0.1), 2: (-0.2, 0.2), 3: (-0.3, 0.3)})

    def run(rows):
        history = History(rows)
        model = fit_catboost(history, train_end)
        assert model is not None
        return forecast(history, day, cut, model, band)

    clean, perturbed = run(_rows()), run(_rows(scale_after=cut))
    assert np.allclose(clean.p50, perturbed.p50)
    assert np.allclose(clean.p90, perturbed.p90)


def test_training_does_not_see_the_training_end_or_later():
    train_end = START + timedelta(days=45)
    boundary = datetime.combine(train_end, datetime.min.time())
    clean = fit_catboost(History(_rows()), train_end)
    perturbed = fit_catboost(History(_rows(scale_after=boundary)), train_end)
    assert clean is not None and perturbed is not None
    X = np.random.default_rng(0).random((20, len(clean.feature_names_)))
    assert np.allclose(clean.predict(X), perturbed.predict(X))


def test_band_is_monotone_and_non_negative():
    history = History(_rows())
    model = fit_catboost(history, START + timedelta(days=45))
    band = ResidualBand({0: (-0.5, 0.5), -1: (-0.5, 0.5)})
    day = START + timedelta(days=50)
    for hour in (7, 9, 13, 20, 23):
        now = datetime.combine(day, datetime.min.time()).replace(hour=hour)
        f = forecast(history, day, now, model, band)
        assert all(0 <= lo <= mid <= hi for lo, mid, hi in zip(f.p10, f.p50, f.p90, strict=True))


def test_too_little_history_gives_no_model():
    assert fit_catboost(History(_rows()[:30]), START + timedelta(days=5)) is None
