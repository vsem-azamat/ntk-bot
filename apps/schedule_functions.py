import io
import asyncio
from typing import List
from datetime import datetime, time

from aiogram import Bot, types

from config import config
from apps.plot_functions import plotGraph
from apps.collect_time import generaet_time_list
from apps.parse_functions import get_ntk_quantity
from apps.weather_api import weatherAPI
from apps.predictModels import predictModels


async def scheduler(bot: Bot, func: object, target_times: List[time]) -> None:
    """Run func every target_times"""
    while True:
        current_time = datetime.now().time().replace(second=0, microsecond=0)
        if current_time in target_times:
            await func(bot) # type: ignore
            await asyncio.sleep(60)
        await asyncio.sleep(1)


async def recieve_ntk_data(delta_minutes: int = 20) -> None:
    """Collect data from NTK every delta_minutes"""
    time_list = await generaet_time_list(delta_minutes=delta_minutes)
    
    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time in time_list:
            try:
                with open('ntk_data.txt', 'a') as file:
                    date = datetime.now().strftime("%Y-%m-%d")
                    quantity_ntk = await get_ntk_quantity()
                    file.write(f"{date} {current_time} - {quantity_ntk}\n")
            except  Exception as e:
                print(e)
            await asyncio.sleep(delta_minutes*60-60)
        else:
            await asyncio.sleep(1)
            
            
async def daily_graph(bot: Bot) -> None:
    """Send daily graphs every target_times to NTK big chat"""
    await predictModels.learn_models()
    fig_visits, _ = await plotGraph.daily_graph_with_predictions()
    fig_weather, _, _ = await weatherAPI.plot_daily_weather_forecast()
    
    buffer_visits = io.BytesIO()
    fig_visits.savefig(buffer_visits, format='png')
    buffer_visits.seek(0)
    
    buffer_weather = io.BytesIO()
    fig_weather.savefig(buffer_weather, format='png')
    buffer_weather.seek(0)
    
    media_group = [
        types.InputMediaPhoto(buffer_visits), 
        types.InputMediaPhoto(buffer_weather)
        ]
    
    await bot.send_media_group(
        chat_id=config.ID_NTK_BIG_CHAT,
        media=types.MediaGroup(media_group),
        disable_notification=True
        )
    await bot.send_message(
        chat_id=config.ID_NTK_BIG_CHAT,
        text=f"📊График посещаемости и прогноз погоды:\n {datetime.now().strftime('%A')} {datetime.now().strftime('%d-%m-%Y')}"
        )
