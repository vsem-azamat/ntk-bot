# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, types

# OTHER
import asyncio
from datetime import datetime, time, timedelta
from parse_functions import recieve_ntk_data, make_day_graph
from configs import config


# Initialize bot and dispatcher
bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)


async def perform_action_at_time(action, target_time):
    while True:
        current_time = datetime.now().time()
        if target_time+timedelta(seconds=10) >= current_time >= target_time:
            await action()
            break
        asyncio.sleep(1)


async def send_daily_graph():
    image = await make_day_graph()
    await bot.send_photo(
        chat_id=config.SUPER_ADMINS[1],
        photo=types.InputFile(image),
        caption=str(datetime.now().strftime('%d-%m-%Y'))
    )


async def on_startup(dp):
    from configs import setup

    setup(dp)
    asyncio.create_task(recieve_ntk_data(config.DELTA_TIME_FOR_RECIEVE_NTK))

    await perform_action_at_time(send_daily_graph, time(10,00))
    await perform_action_at_time(send_daily_graph, time(14,00))
    await perform_action_at_time(send_daily_graph, time(18,00))
    await perform_action_at_time(send_daily_graph, time(22,00))


if __name__ == '__main__':
    from aiogram import executor
    from hanlders import dp
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
