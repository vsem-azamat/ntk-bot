import asyncio
import logging
import random

from aiogram import Bot, Router, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from apps.llm import (
    answer_probability,
    ensure_settings,
    get_llm_response,
    set_answer_probability,
    set_system_prompt,
    settings_loaded,
    system_prompt,
)
from bot.filters import ChatTypeFilter, NTKChatFilter, SuperAdmins

logger = logging.getLogger(__name__)

router = Router()

_PROBABILITIES = ("0.025", "0.05", "0.1", "0.15", "0.2")
_DISABLED = "-1"
# Telegram caps text messages at 4096 characters (counted in UTF-16); longer
# prompts are sent as a .txt file.
_MAX_MESSAGE_LEN = 3500
_MAX_PROMPT_BYTES = 64 * 1024
_DB_DOWN = "The database is unavailable right now; nothing changed. Try again later."


@router.message(Command("llm"), SuperAdmins())
async def choose_answer_probability(message: types.Message):
    """Pick how often the bot answers chat messages with the LLM."""
    builder = InlineKeyboardBuilder()
    for p in _PROBABILITIES:
        builder.add(
            types.InlineKeyboardButton(text=f"{float(p) * 100:g}%", callback_data=f"llm_{p}")
        )
    builder.add(types.InlineKeyboardButton(text="disable", callback_data=f"llm_{_DISABLED}"))
    builder.adjust(2)
    current = await asyncio.to_thread(answer_probability)
    state = "disabled" if current < 0 else f"{current * 100:g}%"
    await message.reply(
        text=f"LLM answer probability (now {state}):", reply_markup=builder.as_markup()
    )


@router.callback_query(lambda callback_query: callback_query.data.startswith("llm_"), SuperAdmins())
async def store_answer_probability(callback_query: types.CallbackQuery):
    """Store the chosen probability (survives restarts)."""
    value = callback_query.data[4:]
    if value not in (*_PROBABILITIES, _DISABLED):
        await callback_query.answer("Unknown option")
        return
    try:
        await asyncio.to_thread(set_answer_probability, value)
    except Exception:
        logger.exception("Could not store the LLM answer probability")
        await callback_query.answer(_DB_DOWN, show_alert=True)
        return
    await callback_query.answer()
    text = "disabled" if value == _DISABLED else f"{float(value) * 100:g}%"
    if isinstance(callback_query.message, types.Message):
        try:
            await callback_query.message.edit_text(
                text=f"LLM answer probability: {text}", reply_markup=None
            )
        except Exception:
            logger.debug("Probability message already updated", exc_info=True)


# The prompt is deliberately kept out of the public repository, so it is only
# shown and accepted in a private chat with the bot, never in the group.
@router.message(Command("prompt"), SuperAdmins(), ChatTypeFilter("private"))
async def show_prompt(message: types.Message):
    """Send the current system prompt (as a file when it is long)."""
    prompt = await asyncio.to_thread(system_prompt)
    if not prompt:
        await message.reply("The LLM prompt is empty (or the database is unavailable).")
    elif len(prompt) <= _MAX_MESSAGE_LEN:
        await message.reply(prompt)
    else:
        await message.reply_document(
            types.BufferedInputFile(prompt.encode("utf-8"), filename="llm_prompt.txt"),
            caption=f"LLM prompt, {len(prompt)} characters",
        )


async def _replied_text(message: types.Message, bot: Bot) -> str | None:
    """Text of the replied-to message or its .txt document; replies with the
    reason and returns None when it cannot be used."""
    source = message.reply_to_message
    if source is None:
        await message.reply(
            "Reply with /setprompt to a message (or a .txt file) with the new prompt."
        )
        return None
    if source.document is None:
        return source.text or ""
    document = source.document
    name = (document.file_name or "").lower()
    if not (name.endswith(".txt") or (document.mime_type or "").startswith("text/")):
        await message.reply("Send the prompt as text or as a .txt file.")
        return None
    if document.file_size is None or document.file_size > _MAX_PROMPT_BYTES:
        await message.reply("The file is too large for a prompt.")
        return None
    buffer = await bot.download(document)
    if buffer is None:
        await message.reply("Could not download the file.")
        return None
    try:
        return buffer.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        await message.reply("The file must be UTF-8 text.")
        return None


@router.message(Command("setprompt"), SuperAdmins(), ChatTypeFilter("private"))
async def update_prompt(message: types.Message, bot: Bot):
    """Replace the prompt with the replied-to message's text or .txt file."""
    text = await _replied_text(message, bot)
    if text is None:
        return
    text = text.strip()
    if not text:
        await message.reply("The new prompt is empty; nothing changed.")
        return
    try:
        changed = await asyncio.to_thread(set_system_prompt, text)
    except Exception:
        logger.exception("Could not store the LLM prompt")
        await message.reply(_DB_DOWN)
        return
    if changed:
        await message.reply(
            f"LLM prompt updated ({len(text)} characters). The previous one is kept."
        )
    else:
        await message.reply("This is already the current prompt; nothing changed.")


# IMPORTANT: This handler should be the last in the file
@router.message(NTKChatFilter())
async def random_llm_reply(message: types.Message):
    """Occasionally answer a chat message in the bot's persona."""
    text = message.text
    if not text:
        return
    if not settings_loaded():
        await asyncio.to_thread(ensure_settings)
    if random.random() >= answer_probability():
        return
    prompt = system_prompt()
    if not prompt:
        return
    response = await get_llm_response(text, prompt)
    if response:
        await message.reply(response)
