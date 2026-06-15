"""The forecast must condition on today's data: a high live sample pulls the
near-now median up (via as-of features + the anchor guardrail), instead of the
old behaviour where the curve was the same regardless of what happened today."""

import asyncio
from datetime import datetime, timedelta

import pytest

from config import cnfg


@pytest.fixture
def trained(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    monkeypatch.setattr(cnfg, "DATA_DIR", str(tmp_path))

    async def _no_weather(_target):
        return {}

    monkeypatch.setattr("apps.weather_history.forecast_weather", _no_weather)

    from bot import db

    db.init_db()
    # 60 consecutive days of a deterministic bell curve (peak ~300 near 14:00).
    base = datetime(2025, 1, 1)
    for d in range(60):
        day = base + timedelta(days=d)
        for step in range(0, 18 * 6):
            minute = 8 * 60 + step * 10
            hour, mn = divmod(minute, 60)
            if hour >= 24:
                continue
            val = max(10, 300 - abs(minute - 14 * 60) // 2)
            db.insert_occupancy(day.replace(hour=hour, minute=mn), int(val))
    return db


def test_predict_day_conditions_on_today(trained):
    from apps.predictModels import predictModels

    asyncio.run(predictModels.learn_models())

    # A day after the training window, with a single very high live reading.
    target = datetime(2025, 3, 5, 12, 10)
    trained.insert_occupancy(datetime(2025, 3, 5, 12, 0), 900)

    forecast = asyncio.run(predictModels.predict_day(target))
    near = [
        v
        for t, v in zip(forecast.timestamps, forecast.p50, strict=True)
        if datetime(2025, 3, 5, 12, 0) <= t <= datetime(2025, 3, 5, 13, 0)
    ]
    assert near, "expected grid points in the 12:00-13:00 window"
    # Without conditioning the median would sit near the ~250-300 bell curve;
    # the live 900 sample must pull the near-now band up.
    assert max(near) > 400


def test_predict_day_morning_has_no_anchor_jump(trained):
    """Before any sample today, the forecast is the plain (cold) curve."""
    from apps.predictModels import predictModels

    asyncio.run(predictModels.learn_models())
    forecast = asyncio.run(predictModels.predict_day(datetime(2025, 3, 6, 7, 0)))
    # Cold morning: nothing observed yet, so the peak stays in a sane range.
    assert max(forecast.p50) < 600
