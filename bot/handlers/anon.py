import random

from aiogram import Bot, Router, types
from aiogram.filters import Command
from aiogram.enums import ChatMemberStatus
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import cnfg
from bot.filters import SuperAdmins


router = Router()

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


@router.message(Command('reveal'))
async def reveal(message: types.Message):
    """Change reveal probability"""
    builder = InlineKeyboardBuilder()
    builder.row()
    builder.button(text='1%', callback_data='reveal:0.01')
    builder.button(text='5%', callback_data='reveal:0.05')
    builder.button(text='10%', callback_data='reveal:0.1')
    builder.button(text='20%', callback_data='reveal:0.2')
    builder.button(text='30%', callback_data='reveal:0.3')
    builder.button(text='40%', callback_data='reveal:0.4')
    builder.button(text='50%', callback_data='reveal:0.5')
    builder.button(text='Disable', callback_data='reveal:-1')
    builder.adjust(2)
    await message.answer("🤖<b>Вероятность раскрытия анона:</b>", reply_markup=builder.as_markup(), parse_mode='HTML')
    await message.delete()


@router.callback_query(lambda callback_query: callback_query.data.startswith('reveal:'))
async def set_reveal(callback_query: types.CallbackQuery, bot: Bot):
    """Set reveal probability"""
    probability = float(callback_query.data.split(':')[1])
    cnfg.REVEAL_ANON_PROBABILITY = probability
    await callback_query.message.edit_text(f"🤖<b>Вероятность раскрытия анона:</b> {int(probability*100)}%", parse_mode='HTML')
    await callback_query.answer()


@router.message(Command('anon'))
async def anon(message: types.Message, bot: Bot):
    """Send anon message"""
    # Check: User is member of chat
    member = await bot.get_chat_member(
        chat_id=cnfg.ID_NTK_BIG_CHAT,
        user_id=message.from_user.id,
    )
    if member.status not in [ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.MEMBER, ChatMemberStatus.RESTRICTED]:
        builder = InlineKeyboardBuilder()
        builder.button(text='📚Вступить в чат', url='https://t.me/chat_ntk')
        await message.reply(
            text="🚫<b>Только участники чата могут использовать анон</b>🚫", 
            parse_mode='HTML',
            reply_markup=builder.as_markup()
            )
        return
    
    # Check: User wrote message in private chat
    if message.chat.id == message.from_user.id:

        # Check: Anon isn't empty
        text = message.text[6:].strip()
        if not text:

            # Check: Anon is enabled
            if not cnfg.ANON_ENABLED:
                await message.reply("💤<b>Анончик сейчас спит</b>💤", parse_mode='HTML')
                await message.delete()
                return

            # Check: Random reveal identity
            reveal_identity = random.random() < cnfg.REVEAL_ANON_PROBABILITY
            username = user_link = message.from_user.username
            user_link = message.from_user.full_name
            if username: user_link = f'<a href="t.me/{username}">{message.from_user.full_name}</a>'
                
            text_head = "<b>💌Анон плс:</b>\n\n" if reveal_identity else f"<b>💌Анон плс, от {user_link}:</b>\n\n"

            await bot.send_message(
                chat_id=cnfg.ID_NTK_BIG_CHAT,
                text=text_head + text,
                parse_mode='HTML'
            )
        else:
            await message.reply("🚫<b>Анон не может быть пустым</b>🚫", parse_mode='HTML')

    else:
        await message.delete()
