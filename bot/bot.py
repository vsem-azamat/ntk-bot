import asyncio
import logging

from aiogram import Bot, Dispatcher

from apps.schedule_functions import receive_ntk_data
from bot.handlers import router
from config import cnfg, ensure_data_file

logger = logging.getLogger(__name__)


async def on_startup(bot: Bot) -> None:
    await bot.delete_webhook()
    from apps.predictModels import predictModels

    ensure_data_file()
    # Train the regression models and start collecting occupancy data.
    await predictModels.learn_models()
    asyncio.create_task(receive_ntk_data(cnfg.DELTA_TIME_FOR_RECIEVE_NTK))


async def on_shutdown(bot: Bot) -> None:
    await bot.delete_webhook()
    await bot.session.close()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=cnfg.BOT_TOKEN)
    dp = Dispatcher()

    dp.include_router(router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    try:
        await dp.start_polling(bot)
    except Exception:
        logger.exception("Polling stopped with an error")


if __name__ == "__main__":
    asyncio.run(main())
