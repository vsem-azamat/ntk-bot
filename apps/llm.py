"""Random LLM replies in the NTK chat via OpenRouter.

The system prompt and the reply probability are runtime settings stored in the
database (``bot.db`` settings), so they are edited from Telegram (/setprompt,
/llm) without a deploy and are covered by the database backups. The prompt is
not in the repository on purpose.
"""

import logging
import time
from pathlib import Path

import aiohttp

from config import cnfg

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_KEY = "llm_prompt"
PREVIOUS_PROMPT_KEY = "llm_prompt_previous"
PROBABILITY_KEY = "llm_answer_probability"

# Reasoning models spend part of max_tokens on hidden reasoning; 256 tokens cut
# replies mid-sentence or left them empty. Low effort keeps one-liners fast.
_MAX_TOKENS = 1024
_REASONING = {"effort": "low", "exclude": True}
_TIMEOUT = aiohttp.ClientTimeout(total=60)
# Telegram rejects messages over 4096 characters; keep a margin for UTF-16 counting.
MAX_REPLY_CHARS = 3500


class _Settings:
    """In-memory copy of the LLM settings.

    The random-reply handler runs on every chat message; reading the database
    there would cost a round-trip per message and, during a database outage,
    tie up worker threads the collector needs. The bot is the only writer, so
    the copy is loaded once (retried at most every ``_RELOAD_S`` while the
    database is unreachable) and updated in place by the setters."""

    _RELOAD_S = 60.0

    def __init__(self) -> None:
        self.loaded = False
        self.prompt = ""
        self.probability: float | None = None
        self._last_attempt = float("-inf")

    def ensure_loaded(self) -> None:
        if self.loaded or time.monotonic() - self._last_attempt < self._RELOAD_S:
            return
        self._last_attempt = time.monotonic()
        from bot import db

        try:
            if db.get_setting(PROMPT_KEY) is None:
                import_legacy_prompt(cnfg.LEGACY_INSTRUCTIONS_PATH)
            self.prompt = db.get_setting(PROMPT_KEY) or ""
            self.probability = _parse_probability(db.get_setting(PROBABILITY_KEY))
            self.loaded = True
        except Exception:
            logger.warning("LLM settings unavailable; retrying later", exc_info=True)

    def reset(self) -> None:
        self.__init__()


_settings = _Settings()


def ensure_settings() -> bool:
    """Load the settings if needed (does database I/O); returns whether loaded."""
    _settings.ensure_loaded()
    return _settings.loaded


def settings_loaded() -> bool:
    return _settings.loaded


def _parse_probability(stored: str | None) -> float | None:
    if stored is None:
        return None
    try:
        return float(stored)
    except ValueError:
        logger.warning("Bad stored %s=%r; using the default", PROBABILITY_KEY, stored)
        return None


def answer_probability() -> float:
    """Current reply probability: the stored setting, else the env default.
    A negative value means disabled."""
    _settings.ensure_loaded()
    if _settings.probability is None:
        return cnfg.LLM_ANSWER_PROBABILITY
    return _settings.probability


def set_answer_probability(value: str) -> None:
    from bot import db

    db.set_setting(PROBABILITY_KEY, value)
    _settings.probability = _parse_probability(value)


def system_prompt() -> str:
    """Current prompt; empty while it cannot be loaded (no replies then)."""
    _settings.ensure_loaded()
    return _settings.prompt


def set_system_prompt(text: str) -> bool:
    """Replace the prompt, keeping the previous one for a manual rollback.
    Returns False (and changes nothing) when ``text`` is already the prompt, so a
    repeated /setprompt cannot overwrite the rollback copy."""
    from bot import db

    current = db.get_setting(PROMPT_KEY)
    if current == text:
        return False
    updates = {PROMPT_KEY: text}
    if current is not None:
        updates[PREVIOUS_PROMPT_KEY] = current
    db.set_settings(updates)
    _settings.prompt = text
    return True


def import_legacy_prompt(path: str) -> bool:
    """Seed the stored prompt from the pre-database ``.instructions`` file once;
    returns True when it was imported. The file is left in place."""
    from bot import db

    file = Path(path)
    if db.get_setting(PROMPT_KEY) is not None or not file.exists():
        return False
    text = file.read_text(encoding="utf-8-sig").strip()
    if not text:
        return False
    db.set_setting(PROMPT_KEY, text)
    logger.info("Imported the LLM prompt from %s", path)
    return True


async def get_llm_response(message: str, prompt: str) -> str | None:
    """Ask the configured OpenRouter model to respond to ``message``."""
    if not cnfg.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set; skipping LLM response")
        return None

    headers = {
        "Authorization": f"Bearer {cnfg.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cnfg.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 1,
        "max_tokens": _MAX_TOKENS,
        "reasoning": _REASONING,
    }

    try:
        async with (
            aiohttp.ClientSession(timeout=_TIMEOUT) as session,
            session.post(OPENROUTER_URL, headers=headers, json=payload) as response,
        ):
            response.raise_for_status()
            data = await response.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            return None
        return content.strip()[:MAX_REPLY_CHARS]
    except aiohttp.ClientError, TimeoutError, KeyError, IndexError, TypeError:
        logger.exception("OpenRouter request failed")
        return None
