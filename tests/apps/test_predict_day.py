"""End-to-end: stored samples -> learn_models -> predict_day."""

import asyncio
import math
from datetime import datetime, timedelta

import pytest

START = datetime(2025, 1, 6)  # a Monday
DAYS = 70


def _bell(minute: int, day: int = 0) -> int:
    """Peak 300 at 14:00; the level wobbles +-10 % from day to day."""
    return round(max(10, 300 - abs(minute - 14 * 60) // 2) * (1 + 0.1 * math.sin(day)))


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    """Keep CatBoost tiny: these tests check the plumbing, not the accuracy."""
    import apps.forecaster as forecaster

    params = {**forecaster.CATBOOST_PARAMS, "iterations": 50}
    monkeypatch.setattr(forecaster, "CATBOOST_PARAMS", params)
    monkeypatch.setattr(forecaster, "ENSEMBLE_SIZE", 1)


@pytest.fixture
def db():
    """The empty test database (``tests/conftest.py``)."""
    from bot import db

    return db


@pytest.fixture
def seeded(db):
    """``DAYS`` consecutive days of a deterministic bell curve (peak ~300 at 14:00)
    every 20 minutes from 08:00 (10:00 at weekends) to midnight."""
    rows = []
    for d in range(DAYS):
        day = START + timedelta(days=d)
        first = 10 * 60 if day.weekday() >= 5 else 8 * 60
        for minute in range(first, 24 * 60, 20):
            rows.append((day.replace(hour=minute // 60, minute=minute % 60), _bell(minute, d)))
    db._write_occupancy(rows)  # one batch: row-by-row inserts take ~10 s
    return db


def _assert_valid(forecast):
    assert len(forecast.timestamps) == len(forecast.p50) > 0
    for lo, mid, hi in zip(forecast.p10, forecast.p50, forecast.p90, strict=True):
        assert 0 <= lo <= mid <= hi


def test_untrained_bot_serves_the_baseline(seeded):
    from apps.predictModels import predictModels

    assert not predictModels.has_trained()
    target = START + timedelta(days=DAYS)
    forecast = asyncio.run(predictModels.predict_day(target.replace(hour=7)))
    _assert_valid(forecast)
    noon = forecast.timestamps.index(target.replace(hour=14))
    assert forecast.p50[noon] == pytest.approx(300, rel=0.15)


def test_empty_database_predicts_zeros(db):
    from apps.predictModels import predictModels

    asyncio.run(predictModels.learn_models())
    forecast = asyncio.run(predictModels.predict_day(datetime(2025, 3, 3, 12, 0)))
    _assert_valid(forecast)
    assert max(forecast.p50) == 0


def test_learn_then_predict_with_catboost(seeded, monkeypatch):
    """Also covers the seed ensemble: one summed model is saved and loaded."""
    import apps.forecaster as forecaster
    from apps.forecaster import CATBOOST
    from apps.predictModels import _CHOICE_FILE, _MODEL_FILE, _path, predictModels

    monkeypatch.setattr(forecaster, "ENSEMBLE_SIZE", 2)
    asyncio.run(predictModels.learn_models())
    assert predictModels.has_trained()
    assert _path(_MODEL_FILE).exists()
    _path(_CHOICE_FILE).write_text(CATBOOST)  # serve CatBoost whatever the gate said
    target = START + timedelta(days=DAYS)
    forecast = asyncio.run(predictModels.predict_day(target.replace(hour=7)))
    _assert_valid(forecast)
    noon = forecast.timestamps.index(target.replace(hour=14))
    assert 200 < forecast.p50[noon] < 400


def test_forecast_conditions_on_today(seeded):
    from apps.predictModels import predictModels

    asyncio.run(predictModels.learn_models())
    day = START + timedelta(days=DAYS)
    seeded.insert_occupancy(day.replace(hour=12), 900)
    forecast = asyncio.run(predictModels.predict_day(day.replace(hour=12, minute=10)))
    near = [
        v
        for t, v in zip(forecast.timestamps, forecast.p50, strict=True)
        if day.replace(hour=12) <= t <= day.replace(hour=13)
    ]
    # The plain curve sits at ~250 here; the live 900 must pull it up.
    assert max(near) > 400


def test_night_residue_does_not_reach_the_morning_forecast(seeded):
    from apps.predictModels import predictModels

    asyncio.run(predictModels.learn_models())
    day = START + timedelta(days=DAYS)
    morning = day.replace(hour=6, minute=45)
    clean = asyncio.run(predictModels.predict_day(morning))
    seeded.insert_occupancy(day.replace(hour=1, minute=40), 205)
    with_residue = asyncio.run(predictModels.predict_day(morning))
    assert with_residue.p50 == clean.p50
    assert with_residue.timestamps[0] == day.replace(hour=8)
    assert with_residue.timestamps[-1] == day.replace(hour=23, minute=50)


def test_old_model_generation_triggers_a_retrain(seeded):
    """Artefacts of the previous model (choice file only) do not count as
    trained, so the first maintenance tick after an upgrade retrains."""
    from apps.predictModels import _CHOICE_FILE, _path, predictModels

    _path(_CHOICE_FILE).write_text("catboost")
    assert not predictModels.has_trained()
    forecast = asyncio.run(predictModels.predict_day(START + timedelta(days=DAYS, hours=7)))
    _assert_valid(forecast)  # no v2 model file: falls back to the baseline


def test_training_removes_the_previous_model_generation(seeded):
    from apps.predictModels import _STALE_FILES, _path, predictModels

    for name in _STALE_FILES:
        _path(name).write_bytes(b"old")
    asyncio.run(predictModels.learn_models())
    assert not any(_path(name).exists() for name in _STALE_FILES)
