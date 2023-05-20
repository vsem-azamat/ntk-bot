# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, types

# OTHER
import asyncio
from parse_functions import recieve_ntk_data
from configs import Config, NtkGroup


# Initialize bot and dispatcher
bot = Bot(token=Config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)


async def on_startup(dp):
    from configs import NtkGroup

    dp.filters_factory.bind(NtkGroup)
    asyncio.create_task(recieve_ntk_data(Config.DELTA_TIME_FOR_RECIEVE_NTK))


if __name__ == '__main__':
    from aiogram import executor
    from hanlders import dp
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
