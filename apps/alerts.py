"""Operator alerts for the occupancy collector.

``CollectionMonitor`` is a pure state machine: it counts consecutive scrape
failures and returns the text to send when the outage crosses the threshold and
again when collection recovers. ``send_alert`` delivers it to the configured
alert chat (optionally a forum topic); without ``ALERT_CHAT_ID`` alerts are only
logged.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from config import cnfg

logger = logging.getLogger(__name__)

# Three missed samples (an hour at the default 20-min cadence) before alerting,
# so a single slow response does not page anyone.
FAILURE_THRESHOLD = 3


@dataclass
class CollectionMonitor:
    subject: str = "NTK collector"
    what: str = "samples failed"
    threshold: int = FAILURE_THRESHOLD
    failures: int = 0
    first_failure: datetime | None = None
    alerted: bool = False
    last_error: str = field(default="")

    def record_failure(self, now: datetime, error: BaseException) -> str | None:
        """Register a failed scrape; return an alert text when the outage starts."""
        self.failures += 1
        self.last_error = f"{type(error).__name__}: {error}"[:300]
        if self.first_failure is None:
            self.first_failure = now
        if self.alerted or self.failures < self.threshold:
            return None
        self.alerted = True
        return (
            f"{self.subject}: {self.failures} {self.what} in a row since "
            f"{self.first_failure:%Y-%m-%d %H:%M}.\nLast error: {self.last_error}"
        )

    def record_success(self, now: datetime) -> str | None:
        """Register a successful scrape; return a recovery text if we had alerted."""
        text = None
        if self.alerted and self.first_failure is not None:
            text = (
                f"{self.subject} recovered at {now:%Y-%m-%d %H:%M} after "
                f"{self.failures} {self.what} (since {self.first_failure:%Y-%m-%d %H:%M})."
            )
        self.failures = 0
        self.first_failure = None
        self.alerted = False
        self.last_error = ""
        return text


async def send_alert(bot, text: str) -> None:
    """Post ``text`` to the alert chat; never raises (alerts must not kill the loop)."""
    logger.warning("Alert: %s", text)
    if not cnfg.ALERT_CHAT_ID:
        return
    try:
        await bot.send_message(
            cnfg.ALERT_CHAT_ID,
            text,
            message_thread_id=cnfg.ALERT_THREAD_ID or None,
        )
    except Exception:
        logger.exception("Failed to deliver alert")
