import datetime
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


@router.message(Command('mute'), SuperAdmins())
async def mute_user(message: types.Message):
    """Mute user in chat"""
    if message.reply_to_message:
        time_mute = message.text[6:].strip()
        if time_mute and time_mute.isdigit():
            time_mute = int(time_mute)
        else:
            time_mute = 5
        current_time = datetime.datetime.now() + datetime.timedelta(minutes=time_mute)
        user_id = message.reply_to_message.from_user.id
        await message.chat.restrict(
            user_id=user_id,
            permissions=types.ChatPermissions(),
            until_date=current_time
            )
        await message.reply_to_message.reply(f'User muted for {time_mute} minutes')


@router.message(Command('admin'), SuperAdmins())
async def make_admin(message: types.Message):
    """Make user admin in chat"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.promote(
            user_id=user_id,
            can_manage_chat=True,
            can_delete_messages=True,
            can_manage_video_chats=True,
            can_restrict_members=True,
            can_promote_members=True,
            can_change_info=True,
            can_invite_users=True,
            can_pin_messages=True,
            )
    await message.delete()


@router.message(Command('unadmin'), SuperAdmins())
async def unadmin(message: types.Message):
    """Remove admin rights from user"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.promote(user_id=user_id)
    await message.delete()

