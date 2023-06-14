# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, types

# OTHER
import logging
import asyncio
from datetime import time

from config import config
from apps.schedule_functions import scheduler, recieve_ntk_data, daily_graph


# Initialize bot and dispatcher
bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Logging to file
file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logging
logging.getLogger().addHandler(file_handler)

async def on_startup(dp):
    from config import setup
    from apps.predictModels import predictModels

    logging.info("Starting...")

    await predictModels.learn_models()

    setup(dp)
    asyncio.create_task(recieve_ntk_data(config.DELTA_TIME_FOR_RECIEVE_NTK))

    target_times = [time(8, 10), time(12, 00), time(16, 00), time(22, 00)]
    asyncio.create_task(scheduler(bot=bot, func=daily_graph, target_times=target_times))


if __name__ == '__main__':
    from aiogram import executor
    from hanlders import dp
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
