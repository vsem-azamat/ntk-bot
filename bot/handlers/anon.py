from aiogram import Bot, Router, types
from aiogram.filters import Command

from config import cnfg
from bot.filters import SuperAdmins


router = Router()


@router.message(Command('anon_enable'), SuperAdmins())
async def anon_enable(message: types.Message):
    """Enable anon functionality"""
    cnfg.ANON_ENABLED = True
    await message.answer("🤖<b>Анон включен</b>\n")
    await message.delete()


@router.message(Command('anon_disable'), SuperAdmins())
async def anon_disable(message: types.Message):
    """Disable anon functionality"""
    cnfg.ANON_ENABLED = False
    await message.answer("🤖<b>Анон выключен</b>\n")
    await message.delete()


@router.message(Command('anon'))
async def anon(message: types.Message, bot: Bot):
    """Send anon message"""
    if not cnfg.ANON_ENABLED:
        await message.reply("💤<b>Анончик сейчас спит</b>💤")
        await message.delete()
        return

    text_head = "<b>💌Анон плс:</b>\n\n"
    text = message.text[6:].strip()
    if message.chat.id == message.from_user.id and text:
        member = await bot.get_chat_member(
            chat_id=cnfg.ID_NTK_BIG_CHAT,
            user_id=message.from_user.id
            )
        if member.status in ['creator', 'administrator', 'member']:
            await bot.send_message(
                chat_id=cnfg.ID_NTK_BIG_CHAT,
                text=text_head + text,
                parse_mode='HTML'
                )
    else:
        await message.delete()

