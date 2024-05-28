from aiogram import Bot, Router, types
from aiogram.filters import Command

from bot.filters import NTKChatFilter, SuperAdmins

router = Router()


@router.message(Command('nick'), SuperAdmins(), NTKChatFilter())
async def admin(message: types.Message, bot: Bot):
    """Give nickname title to user in NTK chat"""
    bot_info = await bot.me()
    bot_chat_info = await bot.get_chat_member(
        chat_id=message.chat.id,
        user_id=bot_info.id
    )
    if isinstance(bot_chat_info, types.ChatMemberAdministrator):
        if bot_chat_info.can_promote_members:
            new_nick = message.text[6:].strip()
            if new_nick and message.reply_to_message:
                user_id = message.reply_to_message.from_user.id
                user_info = await bot.get_chat_member(
                    chat_id=message.chat.id,
                    user_id=user_id
                    )
                if not isinstance(user_info, types.ChatMemberAdministrator):
                    await message.chat.promote(
                        user_id=user_id, 
                        can_invite_users=True
                        )
                await message.chat.set_administrator_custom_title(
                    user_id=user_id,
                    custom_title=new_nick
                    )
            await message.delete()
            return

    await message.delete()


@router.message(Command('ban'), SuperAdmins())
async def ban_user(message: types.Message):
    """Ban user from chat"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.ban(user_id=user_id)
        await message.reply_to_message.delete()
        return
    await message.delete()


@router.message(Command('unban'), SuperAdmins())
async def unban_user(message: types.Message):
    """Unban user from chat"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.unban(user_id=user_id)
        return
    await message.delete()
