# TELEGRAM BOT API
from aiogram import Bot, Dispatcher, types

# OTHER
import asyncio
from datetime import datetime, time
from parse_functions import recieve_ntk_data, make_day_graph
from configs import config


# Initialize bot and dispatcher
bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)


async def daily_graph(bot: Bot):
    image = await make_day_graph()
    print(1)
    await bot.send_photo(
        chat_id=config.SUPER_ADMINS[1],
        photo=types.InputFile(image),
        caption=str(datetime.now().strftime('%d-%m-%Y'))
    )


async def on_startup(dp):
    from configs import setup

    setup(dp)
    asyncio.create_task(recieve_ntk_data(config.DELTA_TIME_FOR_RECIEVE_NTK))

    async def scheduler():
        target_times = [time(10, 00), time(14, 00), time(18, 00), time(22, 00)]
        while True:
            current_time = datetime.now().time()
            if current_time in target_times:
                await daily_graph(bot)
            await asyncio.sleep(60)
    
    asyncio.create_task(scheduler())

    # # shit code
    # while True:
    #     current_time = datetime.now().time()
    #     target_time1 = time(10,00)
    #     target_time2 = time(14,00)
    #     target_time3 = time(18,00)
    #     target_time4 = time(22,00)
        
    #     if current_time == target_time1 or current_time == target_time2 or \
    #        current_time == target_time3 or current_time == target_time4:
    #         await daily_graph(bot)
    #         await asyncio.sleep(1)


if __name__ == '__main__':
    from aiogram import executor
    from hanlders import dp
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
