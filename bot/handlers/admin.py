import datetime

from aiogram import Router, types
from aiogram.filters import Command

from bot.filters import SuperAdmins

router = Router()


@router.message(Command("ban"), SuperAdmins())
async def ban_user(message: types.Message):
    """Ban user from chat"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.ban(user_id=user_id)
        await message.reply_to_message.delete()
        return
    await message.delete()


@router.message(Command("unban"), SuperAdmins())
async def unban_user(message: types.Message):
    """Unban user from chat"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.unban(user_id=user_id)
        return
    await message.delete()


@router.message(Command("mute"), SuperAdmins())
async def mute_user(message: types.Message):
    """Mute user in chat"""
    if message.reply_to_message:
        raw_time_mute = message.text[6:].strip()
        time_mute = int(raw_time_mute) if raw_time_mute.isdigit() else 5
        current_time = datetime.datetime.now() + datetime.timedelta(minutes=time_mute)
        user_id = message.reply_to_message.from_user.id
        await message.chat.restrict(
            user_id=user_id, permissions=types.ChatPermissions(), until_date=current_time
        )
        await message.reply_to_message.reply(f"User muted for {time_mute} minutes")


@router.message(Command("unmute"), SuperAdmins())
async def unmute_user(message: types.Message):
    """Unmute user in chat"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.restrict(
            user_id=user_id,
            permissions=types.ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_other_messages=True,
                can_invite_users=True,
            ),
        )
        await message.reply_to_message.reply("User unmuted")


@router.message(Command("admin"), SuperAdmins())
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


@router.message(Command("unadmin"), SuperAdmins())
async def unadmin(message: types.Message):
    """Remove admin rights from user"""
    if message.reply_to_message:
        user_id = message.reply_to_message.from_user.id
        await message.chat.promote(user_id=user_id)
    await message.delete()
