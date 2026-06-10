import asyncio
import contextlib

import apps.schedule_functions as sf


def _patch(monkeypatch, *, has_trained):
    calls = {"backfill": 0, "learn": 0, "sleep": 0}

    async def fake_backfill():
        calls["backfill"] += 1
        return 0

    async def fake_learn():
        calls["learn"] += 1

    async def fake_sleep(_seconds):
        calls["sleep"] += 1
        raise asyncio.CancelledError  # break out after the first cycle

    monkeypatch.setattr(sf, "backfill", fake_backfill)
    monkeypatch.setattr(sf.asyncio, "sleep", fake_sleep)
    from apps.predictModels import predictModels

    monkeypatch.setattr(predictModels, "learn_models", fake_learn)
    monkeypatch.setattr(predictModels, "has_trained", lambda: has_trained)
    return calls


async def test_cold_start_backfills_and_trains(monkeypatch):
    calls = _patch(monkeypatch, has_trained=False)
    with contextlib.suppress(asyncio.CancelledError):
        await sf.daily_maintenance_loop()
    assert calls == {"backfill": 1, "learn": 1, "sleep": 1}


async def test_warm_restart_backfills_but_skips_training(monkeypatch):
    calls = _patch(monkeypatch, has_trained=True)
    with contextlib.suppress(asyncio.CancelledError):
        await sf.daily_maintenance_loop()
    assert calls["backfill"] == 1
    assert calls["learn"] == 0
    assert calls["sleep"] == 1
