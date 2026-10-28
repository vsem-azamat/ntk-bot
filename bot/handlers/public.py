import logging

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from apps.parse_functions import get_ntk_quantity
from bot import db
from bot.filters import NTKChatFilter

logger = logging.getLogger(__name__)

router = Router()


@router.message(Command("ntk"), NTKChatFilter())
async def ntk(message: types.Message):
    """Send ntk quantity"""
    try:
        q = await get_ntk_quantity()
        text = f"📚<b>В NTK сейчас людей:</b> {q}"
    except Exception:
        logger.exception("Live NTK count failed")
        last = db.latest_occupancy()
        text = "📚 Сайт NTK сейчас не отвечает."
        if last:
            text += f"\nПоследнее значение: <b>{last[1]}</b> ({last[0]:%d.%m %H:%M})"
    await message.answer(
        text=text,
        parse_mode="HTML",
    )
    await message.delete()


@router.message(Command("help"))
async def help(message: types.Message):
    """Send help message"""
    text = (
        "🤖<b>Хай, я создан для чата @chat_ntk!</b>\n\n"
        "📋<b>Команды:</b>\n"
        "/ntk - Показать кол-во людей в NTK\n"
        "/graph - Показать график посещений NTK\n"
    )
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(text="📚NTK chat", url="https://t.me/chat_ntk"),
        types.InlineKeyboardButton(text="👨‍🎓Admin", url="t.me/vsem_azamat"),
        types.InlineKeyboardButton(text="🧑‍💻GitHub", url="github.com/vsem-azamat/ntk-bot"),
    )
    builder.adjust(1)
    await message.answer(
        text=text,
        reply_markup=builder.as_markup(),
        disable_web_page_preview=True,
        parse_mode="HTML",
    )
