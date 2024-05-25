import asyncio
from aiogram import Bot, Dispatcher

from config import cnfg
from bot.hanlders import router
from apps.schedule_functions import recieve_ntk_data


async def on_startup(bot: Bot) -> None:
    await bot.delete_webhook()
    from apps.predictModels import predictModels

    # Run learning models
    await predictModels.learn_models()
    asyncio.create_task(recieve_ntk_data(cnfg.DELTA_TIME_FOR_RECIEVE_NTK))


async def on_shutdown(bot: Bot) -> None:
    await bot.delete_webhook()
    await bot.session.close()


async def main() -> None:
    bot = Bot(token=cnfg.BOT_TOKEN)
    dp = Dispatcher()

    try:
        dp.include_router(router)
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)
        await dp.start_polling(bot)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    asyncio.run(main())
