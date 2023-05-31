import asyncio
from typing import List
from datetime import datetime, time

from aiogram import Bot

from apps.collect_time import generaet_time_list
from apps.parse_functions import get_ntk_quantity


async def scheduler(bot: Bot, func: object, target_times: List[time]) -> None:
        while True:
            current_time = datetime.now().time().replace(second=0, microsecond=0)
            if current_time in target_times:
                await func(bot)
                await asyncio.sleep(60)
            await asyncio.sleep(1)


async def recieve_ntk_data(delta_minutes: int = 20) -> None:
    time_list = await generaet_time_list(delta_minutes=delta_minutes)
    
    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time in time_list:
            with open('ntk_data.txt', 'a') as file:
                date = datetime.now().strftime("%Y-%m-%d")
                quantity_ntk = await get_ntk_quantity()
                file.write(f"{date} {current_time} - {quantity_ntk}\n")
            await asyncio.sleep(delta_minutes*60-60)
        else:
            await asyncio.sleep(1)