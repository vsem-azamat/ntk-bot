import asyncio
import logging
from datetime import datetime

from apps.collect_time import generate_time_list
from apps.parse_functions import get_ntk_quantity
from config import cnfg

logger = logging.getLogger(__name__)


async def receive_ntk_data(delta_minutes: int = 20) -> None:
    """Collect occupancy data from the NTK website every ``delta_minutes``."""
    time_list = await generate_time_list(delta_minutes=delta_minutes)

    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time in time_list:
            try:
                quantity_ntk = await get_ntk_quantity()
                date = datetime.now().strftime("%Y-%m-%d")
                with open(cnfg.NTK_DATA_PATH, "a", encoding="utf-8") as file:
                    file.write(f"{date} {current_time} - {quantity_ntk}\n")
            except Exception:
                logger.exception("Failed to collect NTK occupancy data")
            await asyncio.sleep(delta_minutes * 60 - 60)
        else:
            await asyncio.sleep(1)
