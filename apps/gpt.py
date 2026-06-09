import logging

import aiohttp

from config import cnfg

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def get_gpt_response(message: str) -> str | None:
    """Ask the configured OpenRouter model to respond to ``message``."""
    headers = {
        "Authorization": f"Bearer {cnfg.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cnfg.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": cnfg.INSTRUCTIONS},
            {"role": "user", "content": message},
        ],
        "temperature": 1,
        "max_tokens": 256,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(OPENROUTER_URL, headers=headers, json=payload) as response,
        ):
            response.raise_for_status()
            data = await response.json()
        return data["choices"][0]["message"]["content"]
    except (aiohttp.ClientError, KeyError, IndexError):
        logger.exception("OpenRouter request failed")
        return None
