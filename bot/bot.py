import asyncio
import logging

from aiogram import Bot, Dispatcher

from apps.schedule_functions import (
    daily_maintenance_loop,
    heartbeat_loop,
    morning_digest_loop,
    receive_ntk_data,
)
from bot import db
from bot.handlers import router
from config import INSTRUCTIONS_PATH, cnfg

logger = logging.getLogger(__name__)


async def on_startup(bot: Bot) -> None:
    await bot.delete_webhook()

    db.init_db()

    if not cnfg.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set; GPT replies are disabled")
    if not cnfg.INSTRUCTIONS:
        logger.warning(
            "Instructions file '%s' is missing; GPT will use an empty system prompt",
            INSTRUCTIONS_PATH,
        )

    # Background workers; polling starts immediately. Daily maintenance handles
    # weather backfill + model retraining (training only on a cold start or once
    # a day, never on every restart).
    asyncio.create_task(heartbeat_loop())
    asyncio.create_task(receive_ntk_data(cnfg.DELTA_TIME_FOR_RECIEVE_NTK))
    asyncio.create_task(daily_maintenance_loop())
    asyncio.create_task(morning_digest_loop(bot))


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
