import asyncio
import logging
from datetime import datetime
from pathlib import Path

from apps.collect_time import generate_time_list
from apps.parse_functions import get_ntk_quantity
from apps.weather_history import backfill
from config import cnfg

logger = logging.getLogger(__name__)


def _touch_heartbeat() -> None:
    """Update the liveness file the container healthcheck inspects."""
    try:
        Path(cnfg.HEARTBEAT_PATH).touch()
    except OSError:
        logger.exception("Failed to update heartbeat file")


async def heartbeat_loop(interval_seconds: float = 30) -> None:
    """Refresh the liveness file on a fixed cadence, independent of collection.

    The collection loop sleeps for ~19 minutes after each sample, far longer
    than the container healthcheck window, so liveness cannot be tied to it.
    This dedicated loop keeps the heartbeat fresh whenever the event loop runs.
    """
    while True:
        _touch_heartbeat()
        await asyncio.sleep(interval_seconds)


async def receive_ntk_data(delta_minutes: int = 20) -> None:
    """Collect occupancy data from the NTK website every ``delta_minutes``."""
    from bot import db  # local import avoids a circular import at module load

    time_list = await generate_time_list(delta_minutes=delta_minutes)

    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time in time_list:
            try:
                quantity_ntk = await get_ntk_quantity()
                db.insert_occupancy(datetime.now(), quantity_ntk)
            except Exception:
                logger.exception("Failed to collect NTK occupancy data")
            await asyncio.sleep(delta_minutes * 60 - 60)
        else:
            await asyncio.sleep(1)


async def weather_backfill_loop(interval_seconds: float = 24 * 60 * 60) -> None:
    """Keep the weather table current: backfill on startup, then once per day."""
    while True:
        try:
            written = await backfill()
            logger.info("Weather backfill wrote %d rows", written)
        except Exception:
            logger.exception("Weather backfill failed")
        await asyncio.sleep(interval_seconds)
