# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, types

# OTHER
import asyncio
from parse_functions import recieve_ntk_data
from configs import config


# Initialize bot and dispatcher
bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)


async def on_startup(dp):
    from configs import setup

    setup(dp)
    asyncio.create_task(recieve_ntk_data(config.DELTA_TIME_FOR_RECIEVE_NTK))


if __name__ == '__main__':
    from aiogram import executor
    from hanlders import dp
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
