"""Morning digest: one daily occupancy-forecast message that is posted at 06:45,
refreshed every hour, and deleted at 16:00.

The scheduling decision is a pure function (`decide_digest_action`) so it can be
unit-tested without Telegram, and so the loop stays restart-safe: state lives in
SQLite and every tick simply re-derives the right action from the clock.
"""

import io
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum

POST_AT = time(6, 45)
DELETE_AT = time(16, 0)


class DigestAction(Enum):
    NONE = "none"
    POST = "post"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class DigestState:
    post_date: date
    chat_id: int
    message_id: int
    last_update_hour: int


def decide_digest_action(now: datetime, state: DigestState | None) -> DigestAction:
    """Decide what to do this tick from the current time and the stored state."""
    today = now.date()

    # A message left over from a previous day must be removed first.
    if state is not None and state.post_date != today:
        return DigestAction.DELETE

    if state is None:
        # Nothing posted today yet — post once we are inside the active window.
        if POST_AT <= now.time() < DELETE_AT:
            return DigestAction.POST
        return DigestAction.NONE

    # Today's message exists.
    if now.time() >= DELETE_AT:
        return DigestAction.DELETE
    if now.hour > state.last_update_hour:
        return DigestAction.UPDATE
    return DigestAction.NONE


def build_caption(forecast, current_count: int | None) -> str:
    """Short caption: current count (if the library is open) + expected peak."""
    peak = max(forecast.p50)
    peak_time = forecast.timestamps[forecast.p50.index(peak)].strftime("%H:%M")

    lines = ["📚 НТК сегодня"]
    if current_count:
        lines.append(f"Сейчас: {current_count} чел.")
    lines.append(f"Ожидаемый пик: ~{round(peak)} около {peak_time}")
    return "\n".join(lines)


def current_count(now: datetime) -> int | None:
    """Latest occupancy sample for today, or ``None`` if none yet."""
    from bot import db

    rows = db.fetch_occupancy()
    if rows and rows[-1][0].date() == now.date() and rows[-1][1] > 0:
        return rows[-1][1]
    return None


async def render(now: datetime) -> tuple[bytes, str]:
    """Render the digest image (PNG bytes) and its caption for ``now``."""
    import matplotlib.pyplot as plt

    from apps.plot_functions import plotGraph
    from apps.predictModels import predictModels

    fig, _ = await plotGraph.daily_graph_with_predictions(now)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    forecast = await predictModels.predict_day(now)
    return buffer.read(), build_caption(forecast, current_count(now))
