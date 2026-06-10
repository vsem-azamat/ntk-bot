import asyncio
import contextlib

import apps.schedule_functions as sf


async def test_weather_backfill_loop_calls_backfill_then_sleeps(monkeypatch):
    calls = {"backfill": 0, "sleep": 0}

    async def fake_backfill():
        calls["backfill"] += 1
        return 0

    async def fake_sleep(_seconds):
        calls["sleep"] += 1
        raise asyncio.CancelledError  # break out after first cycle

    monkeypatch.setattr(sf, "backfill", fake_backfill)
    monkeypatch.setattr(sf.asyncio, "sleep", fake_sleep)

    with contextlib.suppress(asyncio.CancelledError):
        await sf.weather_backfill_loop()

    assert calls["backfill"] == 1
    assert calls["sleep"] == 1
