from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from apps.alerts import CollectionMonitor
from apps.schedule_functions import collect_once

T0 = datetime(2026, 10, 21, 17, 40)


def test_alerts_once_at_threshold_then_recovers():
    m = CollectionMonitor(threshold=3)
    texts = [m.record_failure(T0 + timedelta(minutes=20 * i), TimeoutError("t")) for i in range(5)]
    assert texts[:2] == [None, None]
    assert (
        texts[2] is not None and "3 samples failed" in texts[2] and "2026-10-21 17:40" in texts[2]
    )
    assert texts[3:] == [None, None]
    recovered = m.record_success(T0 + timedelta(hours=3))
    assert recovered is not None and "after 5 failed samples" in recovered
    assert m.record_success(T0 + timedelta(hours=4)) is None


def test_short_blip_is_silent():
    m = CollectionMonitor(threshold=3)
    assert m.record_failure(T0, TimeoutError()) is None
    assert m.record_success(T0 + timedelta(minutes=20)) is None
    assert m.failures == 0


@pytest.mark.asyncio
async def test_collect_once_sends_alert_and_keeps_going():
    bot = AsyncMock()
    m = CollectionMonitor(threshold=1)
    with (
        patch("apps.schedule_functions.get_ntk_quantity", AsyncMock(side_effect=TimeoutError())),
        patch("apps.alerts.cnfg.ALERT_CHAT_ID", -100123),
        patch("apps.alerts.cnfg.ALERT_THREAD_ID", 8),
    ):
        await collect_once(bot, m, T0)
    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["message_thread_id"] == 8


@pytest.mark.asyncio
async def test_collect_once_stores_sample():
    bot = AsyncMock()
    with (
        patch("apps.schedule_functions.get_ntk_quantity", AsyncMock(return_value=812)),
        patch("bot.db.insert_occupancy") as insert,
    ):
        await collect_once(bot, CollectionMonitor(), T0)
    insert.assert_called_once_with(T0, 812)
    bot.send_message.assert_not_awaited()
