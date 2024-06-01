import random

from aiogram import Bot, Router, types
from aiogram.filters import Command

from config import cnfg
from bot.filters import SuperAdmins


router = Router()

# Constant variable to change probability of random display
REVEAL_ANON_PROBABILITY = 0.5
# Variably only to display probability for user
REVEAL_ANON_PROBABILITY_DISPLAY = int(REVEAL_ANON_PROBABILITY * 100)


@router.message(Command('anon_enable'), SuperAdmins())
async def anon_enable(message: types.Message):
    """Enable anon functionality"""
    cnfg.ANON_ENABLED = True
    await message.answer("🤖<b>Анон включен</b>\n", parse_mode='HTML')
    await message.delete()


@router.message(Command('anon_disable'), SuperAdmins())
async def anon_disable(message: types.Message):
    """Disable anon functionality"""
    cnfg.ANON_ENABLED = False
    await message.answer("🤖<b>Анон выключен</b>\n", parse_mode='HTML')
    await message.delete()


@router.message(Command('reveal_enable'), SuperAdmins())
async def reveal_enable(message: types.Message):
    """Enable reveal functionality"""
    cnfg.ANON_REVEAL_ENABLED = True
    await message.answer(f"🔎<b>Раскрытие анона с шансом: {REVEAL_ANON_PROBABILITY_DISPLAY}%</b>\n", parse_mode='HTML')
    await message.delete()


@router.message(Command('reveal_disable'), SuperAdmins())
async def reveal_disable(message: types.Message):
    """Disable reveal functionality"""
    cnfg.ANON_REVEAL_ENABLED = False
    await message.answer(f"🔎<b>Раскрытие анона выключено</b>\n", parse_mode='HTML')
    await message.delete()


@router.message(Command('anon'))
async def anon(message: types.Message, bot: Bot):
    """Send anon message"""
    if not cnfg.ANON_ENABLED:
        await message.reply("💤<b>Анончик сейчас спит</b>💤", parse_mode='HTML')
        await message.delete()
        return

    text = message.text[6:].strip()
    if message.chat.id == message.from_user.id and text:
        member = await bot.get_chat_member(
            chat_id=cnfg.ID_NTK_BIG_CHAT,
            user_id=message.from_user.id,
        )
        if member.status in ['creator', 'administrator', 'member']:
            if cnfg.ANON_REVEAL_ENABLED:
                reveal_identity = random.random() < REVEAL_ANON_PROBABILITY
                if reveal_identity:
                    text_head = f"<b>💌Анон плс, от: @{message.from_user.username} ({message.from_user.full_name}):</b>\n\n"
                else:
                    text_head = "<b>💌Анон плс:</b>\n\n"
            else:
                text_head = "<b>💌Анон плс:</b>\n\n"

            await bot.send_message(
                chat_id=cnfg.ID_NTK_BIG_CHAT,
                text=text_head + text,
                parse_mode='HTML'
            )
    else:
        await message.delete()
