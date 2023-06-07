# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, types

# OTHER
import asyncio
from datetime import time

from config import config
from apps.schedule_functions import scheduler, recieve_ntk_data, daily_graph


# Initialize bot and dispatcher
bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)


async def on_startup(dp):
    from config import setup
    from apps.predictModels import predictModels

    await predictModels.learn_models()

    setup(dp)
    asyncio.create_task(recieve_ntk_data(config.DELTA_TIME_FOR_RECIEVE_NTK))

    target_times = [time(8, 10), time(12, 00), time(16, 00), time(22, 00)]
    asyncio.create_task(scheduler(bot=bot, func=daily_graph, target_times=target_times))


if __name__ == '__main__':
    from aiogram import executor
    from hanlders import dp
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
