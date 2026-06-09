import asyncio
import logging
from datetime import datetime
from pathlib import Path

from apps.collect_time import generate_time_list
from apps.parse_functions import get_ntk_quantity
from bot import db
from config import cnfg

logger = logging.getLogger(__name__)


def _touch_heartbeat() -> None:
    """Update the liveness file the container healthcheck inspects."""
    try:
        Path(cnfg.HEARTBEAT_PATH).touch()
    except OSError:
        logger.exception("Failed to update heartbeat file")


async def receive_ntk_data(delta_minutes: int = 20) -> None:
    """Collect occupancy data from the NTK website every ``delta_minutes``."""
    time_list = await generate_time_list(delta_minutes=delta_minutes)

    while True:
        _touch_heartbeat()
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
