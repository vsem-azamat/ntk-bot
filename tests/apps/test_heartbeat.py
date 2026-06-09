import asyncio

import pytest

from apps import schedule_functions


def test_touch_heartbeat_creates_file(tmp_path, monkeypatch):
    hb = tmp_path / "heartbeat"
    monkeypatch.setattr(schedule_functions.cnfg, "HEARTBEAT_PATH", str(hb))
    schedule_functions._touch_heartbeat()
    assert hb.exists()


def test_touch_heartbeat_swallows_oserror(monkeypatch):
    # A bad path must not raise — liveness updates can never crash the loop.
    monkeypatch.setattr(schedule_functions.cnfg, "HEARTBEAT_PATH", "/nonexistent-dir/hb")
    schedule_functions._touch_heartbeat()  # should not raise


async def test_heartbeat_loop_refreshes_then_can_be_cancelled(tmp_path, monkeypatch):
    hb = tmp_path / "heartbeat"
    monkeypatch.setattr(schedule_functions.cnfg, "HEARTBEAT_PATH", str(hb))
    task = asyncio.create_task(schedule_functions.heartbeat_loop(interval_seconds=0.01))
    await asyncio.sleep(0.03)
    assert hb.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
