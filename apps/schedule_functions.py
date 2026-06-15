import asyncio
import logging
from datetime import datetime
from pathlib import Path

from apps.collect_time import generate_time_list
from apps.parse_functions import get_ntk_quantity
from apps.weather_history import backfill
from config import cnfg

logger = logging.getLogger(__name__)


def _touch_heartbeat() -> None:
    """Update the liveness file the container healthcheck inspects."""
    try:
        Path(cnfg.HEARTBEAT_PATH).touch()
    except OSError:
        logger.exception("Failed to update heartbeat file")


async def heartbeat_loop(interval_seconds: float = 30) -> None:
    """Refresh the liveness file on a fixed cadence, independent of collection.

    The collection loop sleeps for ~19 minutes after each sample, far longer
    than the container healthcheck window, so liveness cannot be tied to it.
    This dedicated loop keeps the heartbeat fresh whenever the event loop runs.
    """
    while True:
        _touch_heartbeat()
        await asyncio.sleep(interval_seconds)


async def receive_ntk_data(delta_minutes: int = 20) -> None:
    """Collect occupancy data from the NTK website every ``delta_minutes``."""
    from bot import db  # local import avoids a circular import at module load

    time_list = await generate_time_list(delta_minutes=delta_minutes)

    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time in time_list:
            try:
                quantity_ntk = await get_ntk_quantity()
                db.insert_occupancy(datetime.now(), quantity_ntk)
            except Exception:
                logger.exception("Failed to collect NTK occupancy data")
            await asyncio.sleep(delta_minutes * 60 - 60)
        else:
            await asyncio.sleep(1)


async def daily_maintenance_loop(interval_seconds: float = 24 * 60 * 60) -> None:
    """Once a day: refresh weather, then retrain models.

    Retraining runs on the first tick only for a cold start (no model yet); a
    plain restart does not trigger an expensive retrain. Subsequent daily ticks
    always retrain so the model tracks fresh data.
    """
    from apps.predictModels import predictModels

    first = True
    while True:
        try:
            written = await backfill()
            logger.info("Weather backfill wrote %d rows", written)
        except Exception:
            logger.exception("Weather backfill failed")

        if not first or not predictModels.has_trained():
            try:
                await predictModels.learn_models()
                logger.info("Models retrained")
            except Exception:
                logger.exception("Model retrain failed")
        first = False

        await asyncio.sleep(interval_seconds)


async def _digest_tick(bot, now: datetime) -> None:
    """Run one digest iteration: re-derive the action from the clock + stored
    state and post / refresh / delete accordingly.

    Restart-safe: state lives in SQLite, so the action is a pure function of the
    current time and the persisted row.
    """
    from datetime import date

    from aiogram.exceptions import TelegramBadRequest
    from aiogram.types import BufferedInputFile, InputMediaPhoto

    from apps.digest import DigestAction, DigestState, decide_digest_action, render
    from bot import db

    row = db.get_digest_state()
    state = DigestState(date.fromisoformat(row[0]), row[1], row[2], row[3]) if row else None
    action = decide_digest_action(now, state)

    if action is DigestAction.POST:
        image, caption = await render(now)
        message = await bot.send_photo(
            cnfg.DIGEST_CHAT_ID,
            BufferedInputFile(image, "ntk.png"),
            caption=caption,
            parse_mode="HTML",
        )
        db.save_digest_state(
            now.date().isoformat(), cnfg.DIGEST_CHAT_ID, message.message_id, now.hour
        )
    elif action is DigestAction.UPDATE and state is not None:
        image, caption = await render(now)
        try:
            await bot.edit_message_media(
                chat_id=state.chat_id,
                message_id=state.message_id,
                media=InputMediaPhoto(
                    media=BufferedInputFile(image, "ntk.png"),
                    caption=caption,
                    parse_mode="HTML",
                ),
            )
        except TelegramBadRequest as exc:
            # "message is not modified" is benign (two identical renders); any
            # other bad request means the message is gone or unreachable. In
            # NEITHER case do we repost — re-posting would duplicate the digest,
            # which is exactly the bug we are avoiding. Keep the state and mark
            # this hour done so we don't retry the same edit every tick.
            if "not modified" not in str(exc).lower():
                logger.warning("Digest edit failed; keeping existing message: %s", exc)
        # Transient (e.g. network) errors propagate to the caller's handler so
        # the hour is NOT marked done and the edit is retried on the next tick.
        db.save_digest_state(
            state.post_date.isoformat(), state.chat_id, state.message_id, now.hour
        )
    elif action is DigestAction.DELETE and state is not None:
        try:
            await bot.delete_message(state.chat_id, state.message_id)
        except Exception:
            logger.exception("Digest delete failed; forgetting the message anyway")
        finally:
            db.clear_digest_state()


async def morning_digest_loop(bot, interval_seconds: float = 60) -> None:
    """Post / refresh / delete the daily occupancy digest.

    Time-driven and restart-safe. No-op unless DIGEST_ENABLED.
    """
    if not cnfg.DIGEST_ENABLED:
        logger.info("Morning digest disabled (DIGEST_ENABLED is off)")
        return

    while True:
        try:
            await _digest_tick(bot, datetime.now())
        except Exception:
            logger.exception("Morning digest tick failed")
        await asyncio.sleep(interval_seconds)
