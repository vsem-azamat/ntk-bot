"""Loop-level digest tests: a failed hourly edit must never produce a second
message. Regression guard for the "digest posts twice instead of updating" bug.
"""

import asyncio
from datetime import datetime
from typing import Any, cast

import pytest
from aiogram.exceptions import TelegramBadRequest

from apps import schedule_functions
from config import cnfg


class FakeMessage:
    message_id = 42


class FakeBot:
    """Records calls; ``edit_message_media`` can be told to raise."""

    def __init__(self, edit_error: Exception | None = None):
        self.edit_error = edit_error
        self.send_calls = 0
        self.edit_calls = 0
        self.delete_calls = 0

    async def send_photo(self, *args, **kwargs):
        self.send_calls += 1
        return FakeMessage()

    async def edit_message_media(self, *args, **kwargs):
        self.edit_calls += 1
        if self.edit_error is not None:
            raise self.edit_error

    async def delete_message(self, *args, **kwargs):
        self.delete_calls += 1


@pytest.fixture
def digest_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    monkeypatch.setattr(cnfg, "DIGEST_CHAT_ID", -100)

    async def fake_render(now):
        return b"png-bytes", "caption"

    monkeypatch.setattr("apps.digest.render", fake_render)

    from bot import db

    db.init_db()
    return db


def test_post_then_update_uses_a_single_message(digest_db):
    """Happy path: post once, then refresh the same message — never a 2nd send."""
    bot = FakeBot()

    asyncio.run(schedule_functions._digest_tick(bot, datetime(2024, 3, 4, 6, 50)))
    assert bot.send_calls == 1
    assert digest_db.get_digest_state() == ("2024-03-04", -100, 42, 6)

    asyncio.run(schedule_functions._digest_tick(bot, datetime(2024, 3, 4, 7, 5)))
    assert bot.send_calls == 1  # still one message
    assert bot.edit_calls == 1
    assert digest_db.get_digest_state() == ("2024-03-04", -100, 42, 7)


def test_edit_not_modified_does_not_repost(digest_db):
    """The bug: an edit that fails with 'message is not modified' must NOT cause
    the digest to be re-posted on the following tick."""
    err = TelegramBadRequest(method=cast(Any, None), message="Bad Request: message is not modified")
    bot = FakeBot(edit_error=err)

    # Post the first message.
    asyncio.run(schedule_functions._digest_tick(bot, datetime(2024, 3, 4, 6, 50)))
    assert bot.send_calls == 1

    # Hour advances → UPDATE → edit raises "not modified".
    asyncio.run(schedule_functions._digest_tick(bot, datetime(2024, 3, 4, 7, 5)))
    assert bot.edit_calls == 1

    # State must survive so the next tick does NOT see ``None`` and re-post.
    assert digest_db.get_digest_state() is not None

    # Next tick inside the window: there must be no second message.
    asyncio.run(schedule_functions._digest_tick(bot, datetime(2024, 3, 4, 7, 6)))
    assert bot.send_calls == 1, "digest was re-posted instead of updated"
